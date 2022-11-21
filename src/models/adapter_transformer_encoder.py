import math
from transformers import RobertaTokenizer, RobertaModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class AdapterModel(torch.nn.Module):
    def __init__(self, base_dim, reduction_dim):
        super().__init__()
        self.base_dim = base_dim
        self.reduction_dim = reduction_dim

        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(base_dim, reduction_dim))
        self.layers.append(torch.nn.LayerNorm(reduction_dim))
        self.layers.append(torch.nn.GELU())
        self.layers.append(torch.nn.Dropout(p=0.1, inplace=False))

        self.layers.append(torch.nn.Linear(reduction_dim, reduction_dim))
        self.layers.append(torch.nn.LayerNorm(reduction_dim))
        self.layers.append(torch.nn.GELU())
        self.layers.append(torch.nn.Dropout(p=0.1, inplace=False))

        self.layers.append(torch.nn.Linear(reduction_dim, base_dim))
        self.layers.append(torch.nn.LayerNorm(base_dim))
        self.layers.append(torch.nn.Dropout(p=0.1, inplace=False))
    def forward(self, *input, **kwargs):
        z = input[0]
        attention = input[1]
        #print(attention.shape)
        remainder = input[2:]
        for layer in self.layers:
            z = layer(z)
        res = (z, attention, *remainder)
        return res

class TopicPredictorModel(torch.nn.Module):
    def __init__(
        self, 
        output_dim, 
        reduction_dim = 64, 
        base_model = "roberta-large", 
        learnable_token_count = True, 
        single_adapater = False, 
        finetune_base = False,
        use_adapters = True,
        threshold_value = 0.05,
        num_hidden_states = 4
        ):
        super().__init__()
        assert use_adapters or finetune_base
        self.num_hidden_states = num_hidden_states
        self.learnable_token_count = learnable_token_count
        self.base_model = RobertaModel.from_pretrained(base_model)
        self.encoder_intermediate_dim = self.base_model.encoder.layer[0].output.dense.out_features
        for param in self.base_model.parameters():
            param.requires_grad = finetune_base
        if (use_adapters):
            if (not single_adapater):
                num_adapaters = len(self.base_model.encoder.layer) - 1
                for i in range(num_adapaters, 0, -1):
                    self.base_model.encoder.layer.insert(i, AdapterModel(self.encoder_intermediate_dim, reduction_dim))
            else:
                shared_adapater = AdapterModel(self.encoder_intermediate_dim, reduction_dim)
                num_adapaters = len(self.base_model.encoder.layer) - 1
                for i in range(num_adapaters, 0, -1):
                    self.base_model.encoder.layer.insert(i, shared_adapater)
            self.base_model.config.num_hidden_layers = len(self.base_model.encoder.layer)

        if (self.learnable_token_count):
            self.ltcr_input_size = self.encoder_intermediate_dim * self.num_hidden_states
            self.ltcr_hidden_size = self.ltcr_input_size // 8
            self.ltcr_hidden_layers = 2
            self.ltcr_bidir = False
            self.ltcr_output_size = self.ltcr_hidden_size * (2 if self.ltcr_bidir else 1)
            self.learnable_token_reducer = torch.nn.LSTM(
                input_size = self.ltcr_input_size, 
                hidden_size = self.ltcr_hidden_size,
                num_layers = self.ltcr_hidden_layers,
                dropout = 0.1,
                batch_first = True,
                bidirectional = self.ltcr_bidir
                )
            self.learnable_token_reducer_dropout = torch.nn.Dropout(p=0.1, inplace=False)
            '''

            self.learnable_token_reducer = torch.nn.ModuleList()

            self.learnable_token_reducer.append(torch.nn.Linear(self.learnable_token_count, self.learnable_token_count // 2))
            self.learnable_token_reducer.append(torch.nn.LayerNorm(self.learnable_token_count // 2))
            self.learnable_token_reducer.append(torch.nn.GELU())
            self.learnable_token_reducer.append(torch.nn.Dropout(p=0.1, inplace=False))

            self.learnable_token_reducer.append(torch.nn.Linear(self.learnable_token_count // 2, self.learnable_token_count // 4))
            self.learnable_token_reducer.append(torch.nn.LayerNorm(self.learnable_token_count // 4))
            self.learnable_token_reducer.append(torch.nn.GELU())
            self.learnable_token_reducer.append(torch.nn.Dropout(p=0.1, inplace=False))

            self.learnable_token_reducer.append(torch.nn.Linear(self.learnable_token_count // 4, 1))
            '''

        self.output_head = torch.nn.ModuleList()
        if (self.learnable_token_count):
            self.output_head.append(torch.nn.Linear(self.ltcr_output_size, self.encoder_intermediate_dim  * self.num_hidden_states))
        else:
            self.output_head.append(torch.nn.Linear(self.encoder_intermediate_dim * self.num_hidden_states, self.encoder_intermediate_dim  * self.num_hidden_states))
        self.output_head.append(torch.nn.LayerNorm(self.encoder_intermediate_dim * self.num_hidden_states))
        self.output_head.append(torch.nn.GELU())
        self.output_head.append(torch.nn.Dropout(p=0.1, inplace=False))

        self.output_head.append(torch.nn.Linear(self.encoder_intermediate_dim * self.num_hidden_states, (self.encoder_intermediate_dim * self.num_hidden_states) // 2))
        self.output_head.append(torch.nn.LayerNorm((self.encoder_intermediate_dim * self.num_hidden_states) // 2))
        self.output_head.append(torch.nn.GELU())
        self.output_head.append(torch.nn.Dropout(p=0.1, inplace=False))

        self.output_head.append(torch.nn.Linear((self.encoder_intermediate_dim * self.num_hidden_states) // 2, output_dim))
        
        self.thresh_layer = torch.nn.Threshold(threshold_value, 0)
        self.soft_max = torch.nn.Softmax(dim=1)

    def parameter_counts(self):
        grad_params = 0
        no_grad_params = 0
        for param in self.parameters():
            if (param.requires_grad):
                grad_params += math.prod(param.shape)
            else:
                no_grad_params += math.prod(param.shape)
        return grad_params, no_grad_params
        
    def forward(self, inputs):
        #z = self.base_model(**inputs).last_hidden_state
        z = torch.cat(self.base_model(**inputs, output_hidden_states=True).hidden_states[-1 * self.num_hidden_states:], dim=2)
        if (self.learnable_token_count):
            h0 = torch.zeros(self.ltcr_hidden_layers * (2 if self.ltcr_bidir else 1), z.shape[0], self.ltcr_hidden_size).to(device = z.get_device())
            c0 = torch.zeros(self.ltcr_hidden_layers * (2 if self.ltcr_bidir else 1), z.shape[0], self.ltcr_hidden_size).to(device = z.get_device())
            out, (hn, cn) = self.learnable_token_reducer(z, (h0, c0))
            #res = torch.mean(hn, dim=0).squeeze(dim=0)
            #res = torch.mean(out, dim=1).squeeze(dim=1)
            res = out[:, -1, :]
            res = self.learnable_token_reducer_dropout(res)
            '''
            res = z.transpose(2, 1)
            for layer in self.learnable_token_reducer:
                res = layer(res)
            res = res.transpose(2, 1).squeeze(dim=1)
            '''
        else:
            res = torch.mean(z, dim=1).squeeze(dim=1)
        acts = []
        for layer in self.output_head:
            res = layer(res)
            acts.append(res)
        focus_acts = [acts[3].flatten(), acts[7].flatten()]
        #res = torch.where(res > 0, 1.0, 0.0)
        #res = torch.nn.functional.normalize(res)
        #res = self.thresh_layer(res)
        res = torch.nn.functional.normalize(res)
        #res = self.soft_max(res)
        return res, focus_acts, z
            

if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = TopicPredictorModel(482, 64, "roberta-large", 64, False, False).to(device = "cuda:0")
    inputs = tokenizer([" ".join(["Hi" for i in range(64)]) for i in range(64)], return_tensors="pt", truncation=True, max_length=64, padding=True).to(device = "cuda:0")
    outputs, _ = model(inputs)
    print(outputs.shape)
    grad_params, no_grad_params = model.parameter_counts()
    print(grad_params, no_grad_params, grad_params / (grad_params + no_grad_params), grad_params + no_grad_params)