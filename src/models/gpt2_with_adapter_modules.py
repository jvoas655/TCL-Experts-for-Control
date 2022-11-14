# !pip install transformers
# !pip install datasets
# conda install h5py
# Note for Jordan: my workflow of importing data and creating model. Very rough initial implementation

from transformers import GPT2Tokenizer, GPT2Model
# from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
# import h5py
import torch
import torch.nn as nn
# import numpy as np    


class AdapterLayer(nn.Module):
    def __init__(self, input_dim, downsized_dim):
        super().__init__()
        self.input_dim = input_dim

        self.adapter_layer = nn.Sequential(
            nn.Linear(input_dim, downsized_dim),
            nn.ReLU(),
            nn.Linear(downsized_dim, input_dim)
        )
    def forward(self, x, topic_encoding):
        x_with_topic_encoding = torch.cat(x, topic_encoding)
        assert len(x_with_topic_encoding) == self.input_dim
        return self.adapter_layer(x_with_topic_encoding)


from transformers import GPT2Tokenizer, GPT2Model
class GPT_With_Adapter_Modules(nn.Module):
    def __init__(self):
        super(GPT_With_Adapter_Modules, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.module_list = nn.ModuleList(self.gpt2.modules())
        self.GPT2MLP_type = type(self.module_list[13])
        GPT2MLP_output_size = 768
        topic_encoding_size = 482 
        downsized_dim = 64 # TODO: decide number
        self.adapter_layer = AdapterLayer(input_dim = GPT2MLP_output_size+ topic_encoding_size, downsized_dim = downsized_dim)
    def forward(self, x, topic_encoding):
        #99.99% chance this is wrong
        for module in self.module_list[1::]:
            x = module(x)
            if isinstance(module,  self.GPT2MLP_type):
                x = torch.cat(x, topic_encoding)
        return x

# class GPT2Dataset(Dataset):

#     def __init__(self, data_path = "data/text_encoding_pairs.hdf5"):
#         raise NotImplementedError()
#         # with h5py.File(data_path, 'r') as f:
#         #     data = f['default']
            
#             # get the minimum value
#             # print(min(data))
            
#             # # get the maximum value
#             # print(max(data))
            
#             # # get the values ranging from index 0 to 15
#             # print(data[:15])
#         # self.sentence_topic_pairs = data["text"]
# #   def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):


#     # self.tokenizer = tokenizer
#     # self.input_ids = []
#     # self.attn_masks = []

#     # for txt in txt_list:

#     #   encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

#     #   self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
#     #   self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
#     def __len__(self):
#         raise NotImplementedError()
#         return len(self.sentence_topic_pair)

#     def __getitem__(self, idx):
#         raise NotImplementedError()
#         return self.input_ids[idx], self.attn_masks[idx] 

# class GPT_With_Adapter_Modules(nn.Module):
#     def __init__(self):
#         super(GPT_With_Adapter_Modules, self).__init__()
#         self.model = GPT2Model.from_pretrained('gpt2')
#         self.module_list = nn.ModuleList(self.model.modules())
#         # adapter layers
#         input_dim = None
#         downsized_dim = None
#         num_adapter_layers = None
# #         self.adapter_layers = [AdapterLayer(input_dim, downsized_dim) * num_adapter_layers]

#     def forward(self, x):
#         return self.model(x)
#           sequence_output, pooled_output = self.bert(
#                ids, 
#                attention_mask=mask)

#           # sequence_output has the following shape: (batch_size, sequence_length, 768)
#           linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings

#           linear2_output = self.linear2(linear2_output)
# def extract_data(file_path):
#     split = ["train", "test", "val"]
#     data = h5py.File(file_path)
#     # data = list(f) # TODO fix
#     # with h5py.File(file_path, 'r') as f:
#     #     data = f['default']
#     train_data = data["train"]
#     print(train_data.keys())
#     regex = "(cond|ind)_(tf|roverta)_(128|256|512)_encoding" # TODO: modify/use if necessary
#     rand = "ind_t5_large_512_encoding"
#     for ind in train_data["text"]: 
#         x = train_data["text"]["0"][()] #.decoder("utf-8")
            
#     raise NotImplementedError()

if __name__ == "__main__":
    
    # file_path = "data/text_encoding_pairs.hdf5" # TODO: ask jordan what best file for data
    # data = extract_data(file_path)
    model = GPT_With_Adapter_Modules() # You can pass the parameters if required to have more flexible model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # TODO: we are using different tokenizer that Jordan made? right?
    model.to(torch.device("cpu")) ## can be gpu
    # criterion = nn.CrossEntropyLoss() ## If required define your own criterion