from transformers import BertModel
import  torch.nn as nn
import torch
class Bert_Model(nn.Module):
   def __init__(self, classes,entity_pre_embedding):
       super(Bert_Model, self).__init__()
       self.bert = BertModel.from_pretrained('bert-base-uncased')
       # self.out = nn.Linear(self.bert.config.hidden_size, classes)
       self.topic_entity_embedding_matrix = nn.Embedding.from_pretrained(torch.FloatTensor(entity_pre_embedding),
                                                                         freeze=False)
       self.loss = torch.nn.BCELoss(reduction='sum')
       self.dropout = nn.Dropout(0.1)

       # relu activation function
       self.relu = nn.ReLU()

       # dense layer 1
       self.fc1 = nn.Linear(768, 512)

       W1 = torch.ones(400, 768)
       W1 = nn.init.uniform_(W1)
       self.W1 = nn.Parameter(W1)
       self.fc1 = nn.Linear(768, 512)
       # dense layer 2 (Output layer)
       self.fc2 = nn.Linear(512, 3)

       # softmax activation function
       self.softmax =torch.nn.Softmax(dim=1)

   def applyNonLinear(self, question_embedding):
       x = self.fc1(question_embedding)
       x = self.relu(x)
       x = self.dropout(x)
       # output layer
       x = self.fc2(x)
       return x

   def get_triple_representation(self,question,head_entity):


       Topic_entity=torch.matmul(head_entity, self.W1)

       pi = 3.14159265358979323846
       re_head, im_head = torch.chunk(Topic_entity, 2, dim=1)
       re_relation, im_relation = torch.chunk(question, 2, dim=1)

       re_head=torch.cos(re_head)/pi
       im_head=torch.sin(im_head)/pi

       re_relation = torch.cos(re_relation)/pi
       im_relation = torch.sin(im_relation)/pi
       # print("re_relation",re_relation.size())
       # print("im_relation",im_relation.size())
       #        # print("re_head",re_head.size())
       #        # print("im_head",im_head.size())
       re_tail = re_relation * re_head - im_relation * im_head
       im_tail = im_head * re_relation + re_head * im_relation

       # print("re_score", re_score.size())
       # print("im_score", im_score.size())

       tail = torch.cat([re_tail, im_tail], dim=1)
       # score = score.norm(dim=0)


       triple_repre=Topic_entity+tail+question

       return  triple_repre
   def forward(self, input,attention_mask,labels,topic_entity_id):

       _, question_embedding= self.bert(input, attention_mask = attention_mask)
       # question_embedding= self.out(question_embedding)
       topic_entity_embedding = self.topic_entity_embedding_matrix(topic_entity_id)
       triple_representation = self.get_triple_representation(question_embedding, topic_entity_embedding)
       x=self.applyNonLinear(triple_representation)

       # apply softmax activation
       question_embedding= self.softmax(x)
       actual_r = labels

       # if self.label_smoothing:
       #     actual_r = ((1.0 - self.label_smoothing) * actual_r) + (1.0 / actual_r.size(1))


       loss = self.loss(question_embedding, actual_r)

       return question_embedding,loss

