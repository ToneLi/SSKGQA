
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
import  numpy as np
from transformers import get_linear_schedule_with_warmup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
import  torch.nn as nn
import time
import datetime
import random
from model import  Bert_Model


MAX_LEN=15



def labeltoOneHot( indices):
    if indices == 0:
        one_hot = [0, 0, 1]
    elif indices == 1:
        one_hot = [0, 1, 0]
    elif indices == 2:
        one_hot = [1, 0, 0]

    return  one_hot

    # elif indices==6:
    #     one_hot = (torch.FloatTensor([1,0,0,0,0,0]))



def get_entity_2_idx():
  transe_entity_embedding=np.load("data/entity_vector.npy")
  with open("data/freebase_entity_mention_to_id.txt","r",encoding="utf-8") as fr:
      i=-1
      freebase_id=[]
      index_id=[]
      for line in fr.readlines():
          i=i+1
          topic_id=line.split("\t")[0]
          freebase_id.append(topic_id)
          index_id.append(i)
      entity2id=dict(zip(freebase_id,index_id))

      return entity2id,transe_entity_embedding




def get_data(file):


    entity_dict = "data/entities.dict"
    entity_path = "data/E.npy"
    entities = np.load(entity_path)
    e = preprocess_entities_relations(entity_dict, entities)
    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    sentences=[]
    labels=[]
    TE=[]
    with open(file,"r",encoding="utf-8") as fr:
        for line in fr.readlines():
            line=line.strip().split("\t")
            sentences.append(line[0])
            TE.append(entity2idx[line[1]])
            labels.append(labeltoOneHot(int(line[-1])))

    return  sentences,labels,TE





def get_input_and_mask(sentences):
    input_ids = []
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
    # Print sentence 0, now as a list of IDs.

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

    # Create attention masks
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return  input_ids, attention_masks




def get_input_id():
    train_sentences, train_labels,train_entity_mention_id = get_data("data/metaQA_classification_train.txt")
    # print(train_entity_mention_id)

    train_input_ids, train_attention_masks=get_input_and_mask(train_sentences)

    dev_sentences, dev_labels,test_entity_mention_id = get_data("data/metaQA_classification_1hops_test.txt")
    dev_input_ids, dev_attention_masks = get_input_and_mask(dev_sentences)
    #
    #
    #
    # # print(train_sentences)
    train_inputs = torch.tensor(train_input_ids)
    validation_inputs = torch.tensor(dev_input_ids)

    train_labels = torch.FloatTensor(train_labels)
    validation_labels = torch.FloatTensor(dev_labels)
    #
    train_masks = torch.tensor(train_attention_masks)
    validation_masks = torch.tensor(dev_attention_masks)

    train_entity_mention_id=torch.tensor(train_entity_mention_id)
    test_entity_mention_id=torch.tensor(test_entity_mention_id)



    batch_size = 32
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels,train_entity_mention_id)
    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels,test_entity_mention_id)
    validation_sampler = SequentialSampler(validation_data) 
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return  train_dataloader, validation_dataloader


def flat_accuracy(preds, labels):
    # pred_flat = np.argmax(preds, axis=1).flatten()
    # labels_flat = labels.flatten()
    pred_flat = preds
    labels_flat = labels
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
#

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def TE(model,validation_dataloader):
    model.eval()
    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.cuda() for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels,topic_entity_id = batch

        # speeding up validation
        with torch.no_grad():
            outputs, _ = model(input=b_input_ids,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               topic_entity_id=topic_entity_id)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs
        logits=torch.argmax(logits,dim=1)
        b_labels=torch.argmax(b_labels,dim=1)
        # Move logits and labels to CPU
        # print("logits",logits)
        # print("b_labels",b_labels)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy
        # Track the number of batches
        nb_eval_steps += 1
    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

    global_results=eval_accuracy / nb_eval_steps
    print("global_results",global_results)

    return  global_results

def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key.strip()] = i
        idx2entity[i] = key.strip()
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix


def preprocess_entities_relations(entity_dict, entities):
    e = {}

    f = open(entity_dict, 'r')
    for line in f:
        line = line.strip().split('\t')
        ent_id = int(line[0])
        ent_name = line[1].lower()
        e[ent_name] = entities[ent_id]
    f.close()


    return e


if __name__=="__main__":
    patience=5
    classes=3
    entity_dict="data/entities.dict"
    entity_path="data/E.npy"
    entities = np.load(entity_path)
    e= preprocess_entities_relations(entity_dict, entities)
    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)



    train_dataloader, validation_dataloader = get_input_id()

    model=Bert_Model(classes,embedding_matrix)
    #
    # model = BertForSequenceClassification.from_pretrained(
    #     "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    #     num_labels=5,  # The number of output labels--2 for binary classification.
    #     # You can increase this for multi-class tasks.
    #     output_attentions=False,  # Whether the model returns attentions weights.
    #     output_hidden_states=False,  # Whether the model returns all hidden-states.
    # )
    # Tell pytorch to run this model on the GPU.
    model.cuda()
    best_model = model.state_dict()
    optimizer = AdamW(model.parameters(), lr=5e-8, eps=1e-8)  # args.adam_epsilon  - default is 1e-8. )




    # Create the learning rate scheduler.
    epochs = 2
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    # For each epoch...
    best_score = -float("inf")

    model.zero_grad()
    for epoch_i in range(0, epochs):
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].cuda()#to(device)
            b_input_mask = batch[1].cuda()#to(device)
            b_labels = batch[2].cuda()#to(device)
            topic_entity_id = batch[3].cuda()

            score, loss = model(input=b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_labels,
                                topic_entity_id=topic_entity_id
                                )



            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
#
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print("")
        print("Running Validation...")
        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        eps = 0.0001
        global_results=TE(model, validation_dataloader)


        if global_results > best_score + eps:
            best_score = global_results
            no_update = 0
            best_model = model.state_dict()
            print(" accuracy %s increased from previous epoch" % (str(global_results)))
            checkpoint_path = 'checkpoints/'
            checkpoint_file_name = checkpoint_path + ".pt"
            torch.save(model.state_dict(), checkpoint_file_name)

        elif (global_results < best_score + eps) and (no_update < patience):
            no_update += 1
            print("Validation accuracy decreases to %s from %s, %d more epoch to check" % (
                global_results, best_score, patience - no_update))
        # elif no_update == patience:
        #     print("Model has exceed patience. Saving best model and exiting")
        #     torch.save(best_model, checkpoint_path + "best_score_model.pt")
        #     exit()

   

        if epoch_i == epochs - 1:
            print("Final Epoch has reached. Stopping and saving model.")
            torch.save(best_model, checkpoint_path + "best_score_model.pt")
            exit()
