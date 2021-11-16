import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
import numpy as np
from transformers import get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
import torch.nn as nn
import time
import datetime
import random
from model import Bert_Model

MAX_LEN = 15



def labeltoOneHot(indices):
    if indices == 0:
        one_hot = [0, 0, 1]
    elif indices == 1:
        one_hot = [0, 1, 0]
    elif indices == 2:
        one_hot = [1, 0, 0]

    return one_hot

    # elif indices==6:
    #     one_hot = (torch.FloatTensor([1,0,0,0,0,0]))


def get_entity_2_idx():
    transe_entity_embedding = np.load("data/entity_vector.npy")
    with open("data/freebase_entity_mention_to_id.txt", "r", encoding="utf-8") as fr:
        i = -1
        freebase_id = []
        index_id = []
        for line in fr.readlines():
            i = i + 1
            topic_id = line.split("\t")[0]
            freebase_id.append(topic_id)
            index_id.append(i)
        entity2id = dict(zip(freebase_id, index_id))

        return entity2id, transe_entity_embedding


def get_data(file):
    entity_dict = "data/entities.dict"
    entity_path = "data/E.npy"
    entities = np.load(entity_path)
    e = preprocess_entities_relations(entity_dict, entities)
    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    sentences = []
    labels = []
    TE = []
    with open(file, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line = line.strip().split("\t")
            sentences.append(line[0])
            TE.append(entity2idx[line[1]])
            labels.append(labeltoOneHot(int(line[-1])))

    return sentences, labels, TE


def get_input_and_mask(sentences):
    input_ids = []
    sentence_se_ids=[]

    # For every sentence...
    i=-1
    for sent in sentences:
        i=i+1
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
        sentence_se_ids.append(i)

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

    return sentence_se_ids, input_ids, attention_masks


def get_input_id():
    train_sentences, train_labels, train_entity_mention_id = get_data("data/metaQA_classification_train.txt")
    # print(train_entity_mention_id)

    _,train_input_ids, train_attention_masks = get_input_and_mask(train_sentences)

    dev_sentences, dev_labels, test_entity_mention_id = get_data("data/metaQA_classification_1hops_test.txt")
    sentence_ids,dev_input_ids, dev_attention_masks = get_input_and_mask(dev_sentences)
    #
    #
    validation_sentence_ids = torch.tensor(sentence_ids)
    # # print(train_sentences)
    train_inputs = torch.tensor(train_input_ids)


    validation_inputs = torch.tensor(dev_input_ids)

    train_labels = torch.FloatTensor(train_labels)
    validation_labels = torch.FloatTensor(dev_labels)
    #
    train_masks = torch.tensor(train_attention_masks)
    validation_masks = torch.tensor(dev_attention_masks)

    train_entity_mention_id = torch.tensor(train_entity_mention_id)
    test_entity_mention_id = torch.tensor(test_entity_mention_id)

    batch_size = 32
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_entity_mention_id)
    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, test_entity_mention_id,validation_sentence_ids)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader


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


def TE(model, validation_dataloader):
    model.eval()
    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    wrong_sentence_ids = []
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.cuda() for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, topic_entity_id,batch_sentence_ids = batch

        # speeding up validation
        with torch.no_grad():
            outputs, _ = model(input=b_input_ids,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               topic_entity_id=topic_entity_id)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs
        logits = torch.argmax(logits, dim=1)
        b_labels = torch.argmax(b_labels, dim=1)
        # Move logits and labels to CPU

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        batch_sentence_ids=batch_sentence_ids.cpu().numpy()

        for j in range(len(logits)):
            if int(logits[j])!=int(label_ids[j]):
                wrong_sentence_ids.append(batch_sentence_ids[j])

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy
        # Track the number of batches
        nb_eval_steps += 1
    # Report the final accuracy for this validation run.
    # print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    # print("  Validation took: {:}".format(format_time(time.time() - t0)))
    print("the eval_accuracy is",eval_accuracy)
    print("the nb_eval_steps is",nb_eval_steps)
    global_results = eval_accuracy / nb_eval_steps
    print("global_results", global_results)
    print("the wrong sentence id is:",set(wrong_sentence_ids))
    print("the wrong number is:" ,len(set(wrong_sentence_ids)))
    fw=open("wrong_sentence_id_hop1_in_struture_BERT.txt","w",encoding="utf-8")
    for id in set(wrong_sentence_ids):
        fw.write(str(id)+"\n")

    return global_results


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


if __name__ == "__main__":
    patience = 5
    classes = 3
    entity_dict = "data/entities.dict"
    entity_path = "data/E.npy"
    entities = np.load(entity_path)
    e = preprocess_entities_relations(entity_dict, entities)
    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)

    train_dataloader, validation_dataloader = get_input_id()

    model = Bert_Model(classes, embedding_matrix)
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
    fname = 'checkpoints/best_score_model.pt'
    model.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
    global_results = TE(model, validation_dataloader)