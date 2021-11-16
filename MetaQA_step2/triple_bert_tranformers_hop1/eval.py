from util_ import load_data,load_test_data1,DatasetChenQA,DataLoaderChen,TripletLossC

from model import TransBERT
from transformers import AdamW,get_linear_schedule_with_warmup
import  torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
from sentence_transformers import util
import logging
import argparse
import heapq

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--model_path', type=str, default="bert-base-uncased", help='the model path from transformers')
parser.add_argument('--n_epochs', type=int, default=40, help='input batch size')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--num_warmup_steps', type=int, default=50, help='num_warmup_steps')
parser.add_argument('--patience', type=float, default=5, help='update times, up to it, down')
parser.add_argument('--test', type=int, default=0, help='1 for test, or for training')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")


def sentence_id_mask(sentence):
  encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
  return encoded_input

def set_model_logger(file_name):
  '''
  Write logs to checkpoint and console
  '''

  logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=file_name,
    filemode='w'
  )
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)


def TE_BERT(bert, test_data):
  """
  this is the method to test our model
  """

  right=0
  for doub in tqdm(test_data):
    question = doub[0]
    question_feather=sentence_id_mask(question)
    question_embedding=bert(question_feather)
    negs = doub[1]

    pos=doub[2]


    right_labels=[]
    for p in pos:
      l=negs.index(p)
      right_labels.append(l)

    d=[]
    i=-1
    for n in negs:
      i=i+1
      relation_feather = sentence_id_mask(n)
      relation_embedding = bert(relation_feather)
      cos_sim = util.pytorch_cos_sim(question_embedding, relation_embedding)

      d.append(cos_sim[0][0])

    end = [float(x) for x in d]
    predict = end.index(max(end))
    if predict in right_labels:
      right = right + 1
  print("the result is", right / len(test_data)) # becauce the all sample in the test is 1815, we can not /len(test_data)



def eval_(test_data_path):

  """----------load the train and test data----------"""
  T = load_test_data1(test_data_path)

  """----------model initialization----------"""
  model=TransBERT()
  if torch.cuda.is_available():
    model.cuda()
  """----------optimizer and some schedulers----------"""
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-6)

  fname = 'checkpoints/best_score_model.pt'

  model.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
  model.eval()
  global_results = TE_BERT(model, T)




if __name__=="__main__":
  test_data_path="text_match_1hops_test_SS.txt"
  eval_(test_data_path)
