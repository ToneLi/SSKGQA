from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
from typing import List, Dict, Optional, Union, Tuple
import os
from update_feather import get_GCN_feather
import  torch
import copy
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.functional import softmax
class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)

        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):

        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class TransBERT(nn.Module):

    def __init__(self):
        super(TransBERT, self).__init__()
        self.config_keys = ['max_seq_length']
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        # self.bert = AutoModel.from_pretrained("bert-large-uncased")
        dim_model=768
        num_head=1
        hidden=3072
        dropout=0.5
        num_encoder=6
        self.encoder = Encoder(dim_model, num_head, hidden, dropout)
        # self.encoders = nn.ModuleList([
        #     copy.deepcopy(self.encoder)
        #     # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        #     for _ in range(num_encoder)])

        self.shared_lstm = nn.GRU(768, 768, batch_first=True, bidirectional=True)
        # self.bert = AutoModel.from_pretrained("xlm-r-distilroberta-base-paraphrase-v1/0_Transformer")
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 300)
        self.fc3 = nn.Linear(300, 1)
        self.dropout = nn.Dropout(0.7)
        dim=768
        W1 = torch.ones(dim, dim)
        W1 = torch.nn.init.xavier_normal_(W1)
        self.W1 = nn.Parameter(W1)

        W = torch.empty(768, 768)
        nn.init.normal_(W)
        self.W = nn.Parameter(W)
        # if torch.cuda.is_available():
        #     self.W1 = W1.cuda()

        v1 = torch.ones(dim, dim)
        v1 = torch.nn.init.xavier_normal_(v1)
        self.v1 = nn.Parameter(v1)
        # if torch.cuda.is_available():
        #     self.v1 = v1.cuda()


        self.soft_max=torch.nn.Softmax(dim=1)

        self.fc_Q = nn.Linear(768, 768)
        self.fc_K = nn.Linear(768, 768)
        self.fc_V = nn.Linear(768, 768)

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output # First element of model_output contains all token embeddings

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def max_pooling(self,model_output, attention_mask):
        # token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_over_time = torch.max(token_embeddings, 1)[0]
        return max_over_time

    def applyNonLinear(self, question_embedding):
        x = question_embedding
        x = self.tanh(x)
        # x = self.dropout(x)
        # x = self.fc1(x)
        # #
        # # # output layer
        # x = self.fc2(x)
        return x
    def self_attention(self,pos_gcn_feather):
        querys = self.fc_Q(pos_gcn_feather)
        keys = self.fc_K(pos_gcn_feather)
        values = self.fc_V(pos_gcn_feather)
        attn_scores = querys @ keys.T

        attn_scores_softmax = softmax(attn_scores, dim=-1)
        weighted_values = values[:, None] * attn_scores_softmax.T[:, :, None]
        outputs = weighted_values.sum(dim=0)
        # print(outputs.size())
        return  outputs


    def get_batch_token_type_ids(self,sentence_id):
        token_type_ids = []
        for sub in sentence_id:
            # print("len_sub",len(sub))
            i = -1
            index = [0]
            """
            get the index of 102 [SEP] in a sentence
            eg: index : [0, 4, 8, 11, 15, 18, 21, 25, 27, 29, 31]
            """
            for id in sub:
                i = i + 1
                if id == 102:
                    index.append(i)
            if index[-1] != len(sub) - 1:
                index.append(len(sub) - 1)

            """
            two flag:
            eg:D: [[0, 4], [4, 8], [8, 11], [11, 15], [15, 18], [18, 21], [21, 25], [25, 27], [27, 29], [29, 31]]
            """
            step = 2
            D = []
            for i in range(len(index)):
                if i < len(index):
                    if len(index[i:i + 2]) == 2:
                        D.append(index[i:i + 2])

            m = -1
            type_ids = []
            """
            add 1 or 0
            type_ids:[[0, 0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0], [1, 1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0, 0], [1, 1], [0, 0], [1, 1]]
            """
            for double_ in D:
                m = m + 1
                if m == 0:
                    type_ids.append([0] * (double_[1] + 1 - double_[0]))
                else:
                    if m % 2 == 0:
                        type_ids.append([0] * (double_[1] - double_[0]))
                    else:
                        type_ids.append([1] * (double_[1] - (double_[0])))

            token_type_ids_sub = [num for elem in type_ids for num in elem]
            # print("token_type_ids_sub",len(token_type_ids_sub))
            token_type_ids.append(token_type_ids_sub)

        return torch.tensor(token_type_ids)

    def forward(self, features):
        """
        feather:  after tokening
        """
        if torch.cuda.is_available():
            id = features["input_ids"].cuda()
            mask = features["attention_mask"].cuda()
            token_type_ids=self.get_batch_token_type_ids(id).cuda()
            # token_type_ids = features["token_type_ids"].cuda()
        else:
            id = features["input_ids"]
            mask = features["attention_mask"]
            # token_type_ids = features["token_type_ids"]
            token_type_ids = self.get_batch_token_type_ids(id)

        # pos_relation_output_states = self.bert(input_ids=id, attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        pos_relation_output_states = self.bert(input_ids=id, attention_mask=mask,
                                               return_dict=False)
        model="BERT_multi-attention"

        if model=="BERT":
            pos_gcn_feather=pos_relation_output_states[0]
            return pos_gcn_feather[:,0,:]


        if model=="BERT-GRU":
            pos_gcn_feather=pos_relation_output_states[0]
            pos_gcn_feather,_=self.shared_lstm(pos_gcn_feather)
            pos_gcn_feather=self.mean_pooling(pos_gcn_feather,mask)
            return pos_gcn_feather


        if model=="BERT_multi-attention":
            pos_gcn_feather=pos_relation_output_states[0]
            pos_gcn_feather=self.encoder(pos_gcn_feather)
            pos_gcn_feather=self.mean_pooling(pos_gcn_feather,mask)
            return pos_gcn_feather

        if model=="time_GRN":
            pos_gcn_feather=pos_relation_output_states[0]
            pos_gcn_feather=get_GCN_feather(pos_gcn_feather)
            pos_gcn_feather=self.mean_pooling(pos_gcn_feather,mask)
            return pos_gcn_feather

        # pos_gcn_feather1 = self.encoder(pos_gcn_feather)
        # pos_gcn_feather1=pos_gcn_feather1[:,0,:]

        # pos_gcn_feather=pos_gcn_feather[:,0,:]





