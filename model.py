import torch.nn as nn
import transformers
from torch.utils.data import Dataset, DataLoader
import torch

# from transformers import BloomForCausalLM
# from transformers import BloomTokenizerFast
pretrained_weights='bert-base-uncased'
from transformers import BertConfig, BertModel
class modelNN(transformers.BertPreTrainedModel):
    def __init__(self, config, hidden_size, number_of_issues, model='bert' ):
        super(modelNN, self).__init__(config)
        if model =='bert':
          # self.model = transformers.BertModel.from_pretrained( pretrained_weights, output_attentions = True)
          self.model = transformers.BertModel(config)

        elif model =='bloom':
          self.model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b7", output_attentions = True)

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, number_of_issues)

    def forward(self, ex):
        # _, pooled_output, attentions = self.model(ex, return_dict = False)

        _, pooled_output = self.model(ex, return_dict = False)
        pooled_output = self.dropout(pooled_output)
        fc_out = self.fc(pooled_output)
        # return fc_out, attentions
        return fc_out



class Dataset(Dataset):
    def __init__(self, dataframe, max_len):
        pretrained_weights='bert-base-uncased'
        tokenizer_class = transformers.BertTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        self.dataframe = dataframe
        self.max_len = max_len
        self.sep_id = self.tokenizer.encode('[SEP]', add_special_tokens=False)
        self.pad_id = self.tokenizer.encode('[PAD]', add_special_tokens=False)
        # self.sep_id = tokenizer.encode(['[SEP]'], add_special_tokens=False)
        # self.pad_id = tokenizer.encode(['[PAD]'], add_special_tokens=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe[idx]
        text = row[0]
        # targets = torch.tensor(list(row[1:]))
        encoded = self.tokenizer.encode(text, add_special_tokens=True)[:self.max_len-1] #.tolist()
        if encoded[-1] != self.sep_id[0]:
            encoded = encoded + self.sep_id
        padded = encoded + self.pad_id * (self.max_len - len(encoded))
        padded = torch.tensor(padded)
        labels = torch.Tensor(list(row[1:]))
        return padded, labels