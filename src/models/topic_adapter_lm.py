import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class AdapterModel(torch.nn.Module):
    def __init__(self, base_dim, topic_dim, reduction_dim):
        super().__init__()
        self.base_dim = base_dim
        self.reduction_dim = reduction_dim

        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(base_dim + topic_dim, base_dim))
        self.layers.append(torch.nn.LayerNorm(base_dim))
        self.layers.append(torch.nn.GELU())
        self.layers.append(torch.nn.Dropout(p=0.1, inplace=False))

        self.layers.append(torch.nn.Linear(base_dim, reduction_dim))
        self.layers.append(torch.nn.LayerNorm(reduction_dim))
        self.layers.append(torch.nn.GELU())
        self.layers.append(torch.nn.Dropout(p=0.1, inplace=False))

        self.layers.append(torch.nn.Linear(reduction_dim, base_dim))
        self.layers.append(torch.nn.LayerNorm(base_dim))
        self.layers.append(torch.nn.Dropout(p=0.1, inplace=False))

        self.topic_emb = torch.zeros(topic_dim)
    def forward(self, *input, **kwargs):
        z = input[0]
        if (self.topic_emb.get_device() != z.get_device()):
            self.topic_emb = self.topic_emb.to(device = z.get_device())
        z = torch.cat([z, self.topic_emb.expand(z.shape[0], z.shape[1], -1)], dim=2)
        remainder = input[1:]
        for layer in self.layers:
            z = layer(z)
        res = (z,*remainder, *kwargs)
        return res

class TopicExpertLM(torch.nn.Module):
    def __init__(
        self, 
        reduction_dim = 64, 
        topic_dim = 1024,
        base_model = "gpt2", 
        single_adapater = False, 
        finetune_base = False,
        ):
        super().__init__()
        #assert use_adapters or finetune_base
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.unk_token})
        self.base_model = GPT2LMHeadModel.from_pretrained(base_model)
        #self.base_model.resize_token_embeddings(len(self.tokenizer))
        #self.base_model.transformer.wte.weight.data[-1, :] = torch.mean(self.base_model.transformer.wte.weight.data[:-1, :], dim=0)
        self.adapter_intermediate_dim = 768
        for param in self.base_model.parameters():
            param.requires_grad = finetune_base
        self.adapter_modules = []
        if (not single_adapater):
            num_adapaters = len(self.base_model.transformer.h) - 1
            for i in range(num_adapaters, 0, -1):
                self.adapter_modules.append(AdapterModel(self.adapter_intermediate_dim, topic_dim, reduction_dim))
                self.base_model.transformer.h.insert(i, self.adapter_modules[num_adapaters - i])
        else:
            self.adapter_modules.append(AdapterModel(self.adapter_intermediate_dim, topic_dim, reduction_dim))
            num_adapaters = len(self.base_model.transformer.h) - 1
            for i in range(num_adapaters, 0, -1):
                self.base_model.transformer.h.insert(i, self.adapter_modules[0])
        self.base_model.config.n_layer = len(self.base_model.transformer.h)

    def parameter_counts(self):
        grad_params = 0
        no_grad_params = 0
        for param in self.parameters():
            if (param.requires_grad):
                grad_params += math.prod(param.shape)
            else:
                no_grad_params += math.prod(param.shape)
        return grad_params, no_grad_params
        
    def forward(self, inputs, topic_emb):
        for adapter in self.adapter_modules:
            adapter.topic_emb = topic_emb
        res = self.base_model(**inputs, labels=inputs["input_ids"])
        return res
            

if __name__ == "__main__":
    
    model = TopicExpertLM(64, 1024, "gpt2").to(device = "cuda:0")
    tokenizer = model.tokenizer
    inputs = tokenizer(["Hi, I am a language model", "You are a language model"], return_tensors="pt", truncation=True, max_length=512, padding=True).to(device = "cuda:0")
    print(inputs)
    topic_emb = torch.ones(1024)
    outputs = model(inputs, topic_emb)
    print(outputs["loss"])
    #print(outputs["input_ids"].shape)