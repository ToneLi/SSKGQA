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
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--model_path', type=str, default="bert-base-uncased", help='the model path from transformers')
parser.add_argument('--n_epochs', type=int, default=1, help='input batch size')
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
  return  right / len(test_data)



def train(train_data_path,test_data_path):

  """----------load the train and test data----------"""
  question, poss, negs = load_data(train_data_path)


  T = load_test_data1(test_data_path)

  train_data_web = []
  for idx in range(len(question)):
    train_data_web.append([question[idx], poss[idx], negs[idx]])
  train_data_web = DatasetChenQA(train_data_web)
  train_dataloader = DataLoaderChen(train_data_web, shuffle=True, batch_size=args.batch_size)
  print(" the data load is over")

  """----------model initialization----------"""
  model=TransBERT()

  """----------optimizer and some schedulers----------"""
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-6)


  n_epochs = args.n_epochs
  total_steps = len(train_dataloader) * n_epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,  # Default value in run_glue.py
                                              num_training_steps=total_steps)


  """----------define loss function----------"""
  LossF=TripletLossC()
  model.zero_grad() # this step is vert important,!!!! we must zero the weight of the model in the first
  if torch.cuda.is_available():
    model.cuda()

  patience = args.patience
  no_update = 0
  best_model = model.state_dict()
  best_score = -float("inf")


  """----------ok let's train it----------"""
  for epoch in range(n_epochs):
    print("the current epoch is:",epoch)
    model.train()  # this step is vert important  !!!!
    train_dataloader = tqdm(train_dataloader, total=len(train_dataloader), unit="batches")
    for step, batch in enumerate(train_dataloader):
      question_feather = batch[0]
      pos_relation = batch[1]
      neg_relation = batch[2]

      #
      question_embedding=model(question_feather) # if we want to calculate the question feather, input yes
      pos_embedding=model(pos_relation)
      neg_embedding=model(neg_relation)

      loss=LossF(question_embedding,pos_embedding,neg_embedding)


      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

    """----------ok let's test it----------"""
    model.eval()
    global_results = TE_BERT(model, T)

    eps = 0.0001
    if global_results > best_score + eps:
      best_score = global_results
      no_update = 0
      best_model = model.state_dict()
      logging.info(" accuracy %s increased from previous epoch" % (str(global_results)))
      # global_results=  validate(model=model, data_path= valid_data_path, word2idx= word2ix,device=device)
      logging.info('Test global accuracy %s for best valid so far:' % (str(global_results)))
      # writeToFile(answers, 'results_' + model_name + '_' + hops + '.txt')
      suffix = ''
      checkpoint_path = 'checkpoints/'
      checkpoint_file_name = checkpoint_path + suffix + ".pt"
      logging.info('Saving checkpoint to %s' % checkpoint_file_name)
      torch.save(model.state_dict(), checkpoint_file_name)
    elif (global_results < best_score + eps) and (no_update < patience):
      no_update += 1
      logging.info("Validation accuracy decreases to %s from %s, %d more epoch to check" % (
        global_results, best_score, patience - no_update))

    elif no_update == patience:
      logging.info("Model has exceed patience. Saving best model and exiting")
      torch.save(best_model, checkpoint_path + "best_score_model.pt")
      exit()
    if epoch == n_epochs - 1:
      logging.info("Final Epoch has reached. Stopping and saving model.")
      torch.save(best_model, checkpoint_path + "best_score_model.pt")
      exit()



if __name__=="__main__":
  train_data_path='text_match_train_SS.txt'
  test_data_path="step2data_metaQAhop1_after_step1.txt"
  # train_data_path='train_demo.txt'
  # test_data_path="train_demo.txt"

  train(train_data_path,test_data_path)

  # question, poss, negs = load_data(train_data_path)

