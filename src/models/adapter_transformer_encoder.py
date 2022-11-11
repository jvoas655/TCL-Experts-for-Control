import math
from transformers import RobertaTokenizer, RobertaModel
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
        learnable_token_count = None, 
        single_adapater = False, 
        finetune_base = False
        ):
        super().__init__()
        self.learnable_token_count = learnable_token_count
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model)
        self.base_model = RobertaModel.from_pretrained(base_model)
        self.encoder_intermediate_dim = self.base_model.encoder.layer[0].output.dense.out_features
        for param in self.base_model.parameters():
            param.requires_grad = finetune_base
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

        if (self.learnable_token_count is not None):
            self.learnable_token_reducer = torch.nn.ModuleList()

            self.learnable_token_reducer.append(torch.nn.Linear(self.learnable_token_count, self.learnable_token_count // 2))
            self.learnable_token_reducer.append(torch.nn.LayerNorm(self.learnable_token_count // 2))
            self.learnable_token_reducer.append(torch.nn.GELU())
            self.learnable_token_reducer.append(torch.nn.Dropout(p=0.1, inplace=False))

            self.learnable_token_reducer.append(torch.nn.Linear(self.learnable_token_count // 2, 1))

        self.output_head = torch.nn.ModuleList()

        

        self.output_head.append(torch.nn.Linear(self.encoder_intermediate_dim, self.encoder_intermediate_dim))
        self.output_head.append(torch.nn.LayerNorm(self.encoder_intermediate_dim))
        self.output_head.append(torch.nn.GELU())
        self.output_head.append(torch.nn.Dropout(p=0.1, inplace=False))

        self.output_head.append(torch.nn.Linear(self.encoder_intermediate_dim, output_dim))
        self.output_head.append(torch.nn.LayerNorm(output_dim))
        self.output_head.append(torch.nn.GELU())
        self.output_head.append(torch.nn.Dropout(p=0.1, inplace=False))

        self.output_head.append(torch.nn.Linear(output_dim, output_dim))
    
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
        z = self.base_model(**inputs).last_hidden_state
        if (self.learnable_token_count is not None):
            res = z.transpose(2, 1)
            for layer in self.learnable_token_reducer:
                res = layer(res)
            res = res.transpose(2, 1).squeeze(dim=1)
        else:
            res = torch.mean(z, dim=1).squeeze(dim=1)
        for layer in self.output_head:
            res = layer(res)
        return res, z
            

if __name__ == "__main__":
    model = TopicPredictorModel(482, 64, "roberta-large", 64, False, False).to(device = "cuda:0")
    inputs = model.tokenizer([" ".join(["Hi" for i in range(64)]) for i in range(64)], return_tensors="pt", truncation=True, max_length=64, padding=True).to(device = "cuda:0")
    outputs, _ = model(inputs)
    print(outputs.shape)
    grad_params, no_grad_params = model.parameter_counts()
    print(grad_params, no_grad_params, grad_params / (grad_params + no_grad_params), grad_params + no_grad_params)