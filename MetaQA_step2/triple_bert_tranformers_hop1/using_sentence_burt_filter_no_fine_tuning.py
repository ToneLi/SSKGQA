# using sentence burt no fine tuning
from sentence_transformers import SentenceTransformer
import torch
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
import  torch.nn as nn
cos = nn.CosineSimilarity(dim=0)
def computer_vector(sentence):
    sentence_embeddings = model.encode(sentence)
    sentence_embeddings = torch.tensor(sentence_embeddings)
    return  sentence_embeddings

with open("text_match_3hops_test_SS.txt","r",encoding="utf-8") as fr:
    i=-1
    m=-1
    for line in fr.readlines():
        m=m+1
        print(m)
        line=line.strip().split("\t")
        sentence=line[0]
        pos_=line[2].split("@")
        candidate_relations=line[3].split("#")
        question_vector=computer_vector(sentence)

        right_labels = []
        for p in pos_:
            l = candidate_relations.index(p)
            right_labels.append(l)


        all_=[]
        for r in candidate_relations:
            r=r.replace("|"," ").replace("_"," ")
            relation_vector=computer_vector(r)
            sims=cos(question_vector,relation_vector)
            all_.append(sims)
        end = [float(x) for x in all_]
        if end.index(max(end)) in right_labels:
            i=i+1

    print(i/14274)

# hop1: 88.97