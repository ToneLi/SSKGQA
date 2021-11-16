import  torch

import  torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
dim=768
W0 = torch.ones(dim, dim)
W0 = torch.nn.init.xavier_normal_(W0)
W0 = nn.Parameter(W0)
if torch.cuda.is_available():
    W0=W0.cuda()
W1 = torch.ones(dim, dim)
W1 = torch.nn.init.xavier_normal_(W1)
W1 = nn.Parameter(W1)
if torch.cuda.is_available():
    W1=W1.cuda()

W2 = torch.ones(dim, dim)
W2 =torch.nn.init.xavier_normal_(W2)
W2 = nn.Parameter(W2)
if torch.cuda.is_available():
    W2=W2.cuda()

W3 = torch.ones(dim,dim)
W3 =torch.nn.init.xavier_normal_(W3)
W3 = nn.Parameter(W3)
if torch.cuda.is_available():
    W3=W3.cuda()



W4 = torch.ones(768*2,768)
W4 =torch.nn.init.xavier_normal_(W4)
W4 = nn.Parameter(W4)
if torch.cuda.is_available():
    W4=W4.cuda()


def creat_adjact(tempSentence):
    """
    input: [7,6,7,8,9,4,3]
    out: [[7, 6], [4, 3], [7, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 4], [9, 4, 3]]
    :param tempSentence:
    :return:
    """

    n_grams = []
    for j in range(2, 4):
        n_gram = []
        [n_gram.append((tempSentence[i:i + j])) if len(tempSentence[i:i + j]) == j else 0 for i in
         range(0, len(tempSentence))]
        n_grams.append(n_gram)
    end_gram = [num for elem in n_grams for num in elem]

    two_ = []
    three = []
    for gr in end_gram:
        if len(gr) == 2:
            two_.append(gr)
        else:
            three.append(gr)

    end_ = []
    end_.append(two_[0])

    for t in three:
        end_.append(t)
    last_ = (two_[-1])
    last_.reverse()
    end_.append(last_)
    return end_

LRELU=nn.LeakyReLU(0.1)
relu = nn.ReLU()
tanh=nn.Tanh()

fc1 = nn.Linear(768, 384)
fc2 = nn.Linear(768, 768)
if torch.cuda.is_available():
    fc1=fc1.cuda()
    fc2=fc2.cuda()
def upgrade_node_embedding( center_embedding, adj_node_embedding, weight1):
    center_embedding=center_embedding.unsqueeze(0) # (1,768)
    adj_node_embedding=adj_node_embedding.unsqueeze(0)
    W1_h = torch.matmul(center_embedding, weight1)  # (1,768)
    alpha = torch.mul(W1_h, adj_node_embedding)  ## (1,768)
    row = nn.Softmax(dim=1)

    soft_max_alpha = row(LRELU(alpha))  # ([1, 768])

    updata_head_embedding = torch.matmul(adj_node_embedding, weight1)  # (1,768)
    updata_head_embedding = torch.mul(updata_head_embedding, soft_max_alpha)

    return updata_head_embedding



def words_to_entity(hr_embedding):
    # print("---hr_embedding1",hr_embedding.size()) #torch.Size([3, 768])
    hr_embedding=hr_embedding.clone()
    # hr_embedding=torch.matmul(hr_embedding,W0)

    hr_embedding = torch.sum(hr_embedding, dim=0)
    # print("---hr_embedding", hr_embedding) #torch.Size([768])
    return relu(hr_embedding)

def combine_word_entity_infomration(word_vector,updata_entity_vector):
    word_vector=word_vector.clone()
    # combine_=word_vector+updata_entity_vector
    # print("combine_",combine_.size())
    # combine_=torch.matmul(combine_,W3)
    combine_=(word_vector*updata_entity_vector*(word_vector+updata_entity_vector))
    return relu(combine_)

def combine_entity_with_own_infomration(head_embeds, updata_head_embedding):
    #version1
    updata_head_embedding=relu(updata_head_embedding)
    # updata_head_embedding = updata_head_embedding
    last_head_embedding = head_embeds + updata_head_embedding
    last_head_embedding = torch.mul(last_head_embedding, torch.mul(head_embeds, updata_head_embedding))
    last_head_embedding = torch.matmul(last_head_embedding, W2)
    last_head_embedding = torch.relu(last_head_embedding)
    last_head_embedding = last_head_embedding.squeeze(0) #torch.Size([768])
    #version2
    # add_=head_embeds + updata_head_embedding
    # mul_=head_embeds * updata_head_embedding
    #
    # WA=torch.matmul(add_, W2)
    # WM=torch.matmul(mul_,W3)
    #
    # last_head_embedding=(relu(WA)+relu(WM))/2
    # last_head_embedding = last_head_embedding.squeeze(0)
    return last_head_embedding
shared_lstm = nn.GRU(768, 384, batch_first=True, bidirectional=True)
if torch.cuda.is_available():
    shared_lstm.cuda()
dropout = nn.Dropout(0.1)

def get_GCN_feather(question_output_states, pos_template_number, pos_words_position):
    """
    question_output_states: [6,21,768], the batch_size=6, the sentence length=21
    """

    if pos_words_position=="N" and pos_template_number=="N":
        # print("question_output_states",question_output_states)
        # question, _h = shared_lstm(question_output_states)
        # question = dropout(question)
        # question_output_states= torch.max(question, 1)[0]  # (bs, 2H)
        # last_representation, _ = torch.max(question_output_states, dim=1)
        # last_representation= torch.sum(question_output_states, dim=1)
        # return last_representation
        # return question_output_states[:, 0, :]
        return question_output_states
    else:
        batch_size = question_output_states.size(0)
        S=[]
        for i in range(batch_size):
            sentence_vector = question_output_states[i, :, :]
            words_postion = pos_words_position[i]

            if pos_template_number[i] == "1":
                # words_postion: [[0, 1], [2, 3, 4, 5], [6]]
                embes = []
                b_head_p = words_postion[0][0] + 1
                end_head_p = words_postion[0][-1] + 1  # becauce of the CLS label
                head_embedding = sentence_vector[b_head_p:end_head_p + 1, :]
                head_embedding = words_to_entity(head_embedding) # torch.Size([768])
                # print("head_embedding",head_embedding.size())

                b_relation_p = words_postion[1][0] + 1
                end_relation_p = words_postion[1][-1] + 1  # becauce of the CLS label
                relation_embedding = sentence_vector[b_relation_p:end_relation_p + 1, :]
                relation_embedding = words_to_entity(relation_embedding)

                b_tail_ = words_postion[2][0]+1
                tail_embedding = sentence_vector[b_tail_:b_tail_ + 1, :]
                tail_embedding = words_to_entity(tail_embedding)

                embes.append(head_embedding)
                embes.append(relation_embedding)
                embes.append(tail_embedding)
                #embes  [head_vector, relation_vector, tail_vector]
                # adjact_list: [[0, 1], [0, 1, 2], [2, 1]],  for triple (0,1,2)
                adjact_list = creat_adjact(list(range(len(words_postion))))

                m = -1
                for single_adj in adjact_list:
                    m = m + 1
                    if len(single_adj) == 2:
                        head = single_adj[0]
                        head_embeds = embes[head]
                        adj_nodes = single_adj[1]
                        adj_node_embeds = embes[adj_nodes]
                        # # sum the relation nodes embedding
                        updata_head_embedding = upgrade_node_embedding(head_embeds, adj_node_embeds, W1)
                        last_head_embedding=combine_entity_with_own_infomration(head_embeds,updata_head_embedding)

                        # entity relation add word
                        if m == 0:
                            for j in range(b_head_p, end_head_p + 1):

                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding).clone()
                        else:
                            for j in range(b_tail_, b_tail_ + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding).clone()
                    else:
                        # [0, 1, 2]  (left, head, right)
                        center = single_adj[1]
                        center_embedding = embes[center]
                        adj_nodes_left = single_adj[0]
                        left_node_embedding = embes[adj_nodes_left]

                        adj_nodes_right = single_adj[2]
                        right_node_embedding = embes[adj_nodes_right]


                        updata_head_embedding_left = upgrade_node_embedding(center_embedding, left_node_embedding, W1)
                        updata_head_embedding_right = upgrade_node_embedding(center_embedding,right_node_embedding, W1)
                        updata_head_embedding = updata_head_embedding_right + updata_head_embedding_left
                        # -----plus itsself
                        last_head_embedding = combine_entity_with_own_infomration(center_embedding, updata_head_embedding)

                        # entity relation add word
                        for j in range(b_relation_p, end_relation_p + 1):
                            sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding).clone()

            elif pos_template_number[i]=="2":
                #words_postion: [[0], [1, 2, 3, 4, 5, 6], [7], [8, 9, 10, 11, 12, 13, 14], [15]]
            #     print("ff")
                embes = []
                b_head1_p = words_postion[0][0] + 1
                end_head1_p = words_postion[0][-1] + 1  # becauce of the CLS label
                head_embedding = sentence_vector[b_head1_p:end_head1_p + 1, :]
                head_embedding = words_to_entity(head_embedding) # torch.Size([768])
                # print("head_embedding",head_embedding.size())

                b_relation1_p = words_postion[1][0] + 1
                end_relation1_p = words_postion[1][-1] + 1  # becauce of the CLS label
                relation_embedding1 = sentence_vector[b_relation1_p:end_relation1_p + 1, :]
                relation_embedding1 = words_to_entity(relation_embedding1)

                b_tail1_ = words_postion[2][0]+1
                tail_embedding1 = sentence_vector[b_tail1_:b_tail1_ + 1, :]
                tail_embedding1 = words_to_entity(tail_embedding1)


                b_relation2_p = words_postion[3][0] + 1
                end_relation2_p = words_postion[3][-1] + 1  # becauce of the CLS label
                relation_embedding2 = sentence_vector[b_relation2_p:end_relation2_p + 1, :]
                relation_embedding2 = words_to_entity(relation_embedding2)

                b_tail2_ = words_postion[4][0] + 1
                tail_embedding2 = sentence_vector[b_tail2_:b_tail2_ + 1, :]
                tail_embedding2 = words_to_entity(tail_embedding2)


                embes.append(head_embedding)
                embes.append(relation_embedding1)
                embes.append(tail_embedding1)
                embes.append(relation_embedding2)
                embes.append(tail_embedding2)

                adjact_list = creat_adjact(list(range(len(words_postion))))
                #adjact_list--[[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [4, 3]]

                m = -1
                for single_adj in adjact_list:
                    m = m + 1
                    if len(single_adj) == 2:
                        head = single_adj[0]
                        head_embeds = embes[head]
                        adj_nodes = single_adj[1]
                        adj_node_embeds = embes[adj_nodes]
                        # # sum the relation nodes embedding
                        updata_head_embedding = upgrade_node_embedding(head_embeds, adj_node_embeds, W1)
                        # # print(updata_head_embedding)
                        # # -----plus itsself
                        last_head_embedding = combine_entity_with_own_infomration(head_embeds,
                                                                                  updata_head_embedding)


                        # entity relation add word
                        if m == 0:
                            for j in range(b_head1_p, end_head1_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        else:
                            for j in range(b_tail2_, b_tail2_ + 1):
                                sentence_vector[j, :] =combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                    else:
                        # [0, 1, 2]  (left, head, right)
                        center = single_adj[1]
                        center_embedding = embes[center]
                        adj_nodes_left = single_adj[0]
                        left_node_embedding = embes[adj_nodes_left]

                        adj_nodes_right = single_adj[2]
                        right_node_embedding = embes[adj_nodes_right]


                        updata_head_embedding_left = upgrade_node_embedding(center_embedding, left_node_embedding,
                                                                            W1)
                        updata_head_embedding_right = upgrade_node_embedding(center_embedding,
                                                                             right_node_embedding, W1)
                        updata_head_embedding = updata_head_embedding_right + updata_head_embedding_left
                        # -----plus itsself
                        # last_head_embedding = torch.mul((center_embedding + updata_head_embedding),
                        #                                 torch.mul(center_embedding, updata_head_embedding))
                        last_head_embedding = combine_entity_with_own_infomration(center_embedding,updata_head_embedding)

                        # entity relation add word
                        if m==1:
                            for j in range(b_relation1_p, end_relation1_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        elif m==2:
                            for j in range(b_tail1_, b_tail1_ + 1):
                                sentence_vector[j, :] =combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        elif m==3:
                            for j in range(b_relation2_p, b_relation2_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)

            elif pos_template_number[i] == "3":
                # words_postion: [[0], [1, 2, 3], [4], [5, 6, 7], [8]]
                #     print("ff")
                embes = []
                b_head1_p = words_postion[0][0] + 1
                end_head1_p = words_postion[0][-1] + 1  # becauce of the CLS label
                head_embedding = sentence_vector[b_head1_p:end_head1_p + 1, :]
                head_embedding = words_to_entity(head_embedding)  # torch.Size([768])
                # print("head_embedding",head_embedding.size())

                b_relation1_p = words_postion[1][0] + 1
                end_relation1_p = words_postion[1][-1] + 1  # becauce of the CLS label
                relation_embedding1 = sentence_vector[b_relation1_p:end_relation1_p + 1, :]
                relation_embedding1 = words_to_entity(relation_embedding1)

                b_tail1_ = words_postion[2][0] + 1
                tail_embedding1 = sentence_vector[b_tail1_:b_tail1_ + 1, :]
                tail_embedding1 = words_to_entity(tail_embedding1)

                b_relation2_p = words_postion[3][0] + 1
                end_relation2_p = words_postion[3][-1] + 1  # becauce of the CLS label
                relation_embedding2 = sentence_vector[b_relation2_p:end_relation2_p + 1, :]
                relation_embedding2 = words_to_entity(relation_embedding2)

                b_tail2_ = words_postion[4][0] + 1
                end_tail2_ = words_postion[4][-1] + 1
                tail_embedding2 = sentence_vector[b_tail2_:end_tail2_ + 1, :]
                tail_embedding2 = words_to_entity(tail_embedding2)

                embes.append(head_embedding)
                embes.append(relation_embedding1)
                embes.append(tail_embedding1)
                embes.append(relation_embedding2)
                embes.append(tail_embedding2)

                adjact_list = creat_adjact(list(range(len(words_postion))))
                # print("-----d",[[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [4, 3]])

                m = -1
                for single_adj in adjact_list:
                    m = m + 1
                    if len(single_adj) == 2:
                        head = single_adj[0]
                        head_embeds = embes[head]
                        adj_nodes = single_adj[1]
                        adj_node_embeds = embes[adj_nodes]
                        # # sum the relation nodes embedding
                        updata_head_embedding = upgrade_node_embedding(head_embeds, adj_node_embeds, W1)
                        # # print(updata_head_embedding)
                        # # -----plus itsself
                        last_head_embedding = combine_entity_with_own_infomration(head_embeds,
                                                                                  updata_head_embedding)
                        # entity relation add word
                        if m == 0:
                            for j in range(b_head1_p, end_head1_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        else:
                            for j in range(b_tail2_, end_tail2_ + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                    else:
                        # [0, 1, 2]  (left, head, right)
                        center = single_adj[1]
                        center_embedding = embes[center]
                        adj_nodes_left = single_adj[0]
                        left_node_embedding = embes[adj_nodes_left]

                        adj_nodes_right = single_adj[2]
                        right_node_embedding = embes[adj_nodes_right]


                        updata_head_embedding_left = upgrade_node_embedding(center_embedding, left_node_embedding,
                                                                            W1)
                        updata_head_embedding_right = upgrade_node_embedding(center_embedding,
                                                                             right_node_embedding, W1)
                        updata_head_embedding = updata_head_embedding_right + updata_head_embedding_left
                        # -----plus itsself
                        # last_head_embedding = torch.mul((center_embedding + updata_head_embedding),
                        #                                 torch.mul(center_embedding, updata_head_embedding))
                        last_head_embedding = combine_entity_with_own_infomration(center_embedding,updata_head_embedding)
                        # entity relation add word
                        if m == 1:
                            for j in range(b_relation1_p, end_relation1_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        elif m == 2:
                            for j in range(b_tail1_, b_tail1_ + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        elif m == 3:
                            for j in range(b_relation2_p, b_relation2_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)

            elif pos_template_number[i] == "4":
                #[[0, 1], [2, 3, 4], [5], [6, 7, 8], [9], [10], [11, 12, 13], [14]]
                #ben stiller#film.actor.film#?y#film.performance.character#?x#?y#film.performance.film#megamind#4
                embes = []
                b_head1_p = words_postion[0][0] + 1
                end_head1_p = words_postion[0][-1] + 1  # becauce of the CLS label
                head_embedding = sentence_vector[b_head1_p:end_head1_p + 1, :]
                head_embedding = words_to_entity(head_embedding)  # torch.Size([768])
                # print("head_embedding",head_embedding.size())

                b_relation1_p = words_postion[1][0] + 1
                end_relation1_p = words_postion[1][-1] + 1  # becauce of the CLS label
                relation_embedding1 = sentence_vector[b_relation1_p:end_relation1_p + 1, :]
                relation_embedding1 =words_to_entity(relation_embedding1)

                b_tail1_ = words_postion[2][0] + 1
                tail_embedding1 = sentence_vector[b_tail1_:b_tail1_ + 1, :]
                tail_embedding1 = words_to_entity(tail_embedding1)

                b_relation2_p = words_postion[3][0] + 1
                end_relation2_p = words_postion[3][-1] + 1  # becauce of the CLS label
                relation_embedding2 = sentence_vector[b_relation2_p:end_relation2_p + 1, :]
                relation_embedding2 = words_to_entity(relation_embedding2)

                b_tail2_ = words_postion[4][0] + 1
                end_tail2_ = words_postion[4][-1] + 1
                tail_embedding2 = sentence_vector[b_tail2_:end_tail2_ + 1, :]
                tail_embedding2 =words_to_entity(tail_embedding2)

                b_head2_p = words_postion[5][0] + 1
                end_head2_p = words_postion[5][-1] + 1  # becauce of the CLS label
                head_embedding2= sentence_vector[b_head2_p:end_head2_p + 1, :]
                head_embedding2 = words_to_entity(head_embedding2)  # torch.Size([768])
                # print("head_embedding",head_embedding.size())

                b_relation3_p = words_postion[6][0] + 1
                end_relation3_p = words_postion[6][-1] + 1  # becauce of the CLS label
                relation_embedding3 = sentence_vector[b_relation3_p:end_relation3_p + 1, :]
                relation_embedding3 = words_to_entity(relation_embedding3)

                b_tail3_ = words_postion[7][0] + 1
                end_tail3_ = words_postion[7][-1] + 1
                tail_embedding3 = sentence_vector[b_tail3_:end_tail3_ + 1, :]
                tail_embedding3 = words_to_entity(tail_embedding3)


                embes.append(head_embedding)
                embes.append(relation_embedding1)
                embes.append(tail_embedding1)
                embes.append(relation_embedding2)
                embes.append(tail_embedding2)

                embes.append(head_embedding2)
                embes.append(relation_embedding3)
                embes.append(tail_embedding3)

                adjact_list = [[0, 1], [0, 1, 2], [2, 1, 3, 6], [2, 3, 4], [4, 3], [2, 6, 7], [7, 6]]
                m=-1
                for single_adj in adjact_list:
                    m=m+1
                    if len(single_adj) == 2:
                        head = single_adj[0]
                        head_embedding = embes[head]
                        adj_nodes = single_adj[1]
                        adj_node_embedding = embes[adj_nodes]
                        # sum the relation nodes embedding
                        updata_head_embedding = upgrade_node_embedding(head_embedding, adj_node_embedding, W1)

                        # -----plus itsself
                        last_head_embedding = combine_entity_with_own_infomration(head_embedding,updata_head_embedding)

                        if m == 0:
                            for j in range(b_head1_p, end_head1_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        elif m==4:
                            for j in range(b_tail2_, end_tail2_ + 1):
                                sentence_vector[j, :] =combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        else:
                            for j in range(b_tail3_, end_tail3_ + 1):
                                sentence_vector[j, :] =combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)



                    elif len(single_adj) == 3:
                        # [6, 7, 8]   (left, head, right)
                        center = single_adj[1]
                        center_embedding = embes[center]
                        adj_nodes_left = single_adj[0]
                        left_node_embedding = embes[adj_nodes_left]

                        adj_nodes_right = single_adj[2]
                        right_node_embedding = embes[adj_nodes_right]

                        # sum the relation nodes embedding
                        updata_head_embedding_left = upgrade_node_embedding(center_embedding, left_node_embedding,
                                                                                 W1)
                        updata_head_embedding_right = upgrade_node_embedding(center_embedding, right_node_embedding,
                                                                                  W1)
                        updata_head_embedding = updata_head_embedding_right + updata_head_embedding_left
                        # -----plus itsself
                        # last_head_embedding = torch.mul((center_embedding + updata_head_embedding),
                        #                                 torch.mul(center_embedding, updata_head_embedding))
                        last_head_embedding = combine_entity_with_own_infomration(center_embedding, updata_head_embedding)
                        # entity relation add word
                        if m == 1:
                            for j in range(b_relation1_p, end_relation1_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        elif m == 3:
                            for j in range(b_relation2_p, end_relation2_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        elif m == 5:
                            for j in range(b_head2_p, end_head2_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)


                    elif len(single_adj) == 4:
                        # [2,1,4,7]   (center,adj1,adj2,adj3)
                        center = single_adj[0]
                        center_embedding = embes[center]
                        adj_nodes1 = single_adj[1]
                        adj1_node_embedding = embes[adj_nodes1]

                        adj_nodes2 = single_adj[2]
                        adj2_node_embedding = embes[adj_nodes2]

                        adj_nodes3 = single_adj[3]
                        adj3_node_embedding = embes[adj_nodes3]

                        # sum the relation nodes embedding
                        updata_head_embedding_1 = upgrade_node_embedding(center_embedding, adj1_node_embedding,
                                                                              W1)
                        updata_head_embedding_2 = upgrade_node_embedding(center_embedding, adj2_node_embedding,
                                                                              W1)
                        updata_head_embedding_3 = upgrade_node_embedding(center_embedding, adj3_node_embedding,
                                                                              W1)

                        updata_head_embedding = updata_head_embedding_1 + updata_head_embedding_2 + updata_head_embedding_3
                        # -----plus itsself
                        last_head_embedding = combine_entity_with_own_infomration(center_embedding, updata_head_embedding)

                        if m == 2:
                            for j in range(b_tail1_, b_tail1_ + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)

            elif pos_template_number[i] == "5":
                # print("gg",words_postion)#[[0, 1], [2, 3, 4, 5], [6], [7, 8, 9, 10], [11], [12, 13, 14], [15]]
                #justin bieber#people.person.sibling_s#?y#people.sibling_relationship.sibling#?x#people.person.gender#Male#5
                embes = []
                b_head1_p = words_postion[0][0] + 1
                end_head1_p = words_postion[0][-1] + 1  # becauce of the CLS label
                head_embedding = sentence_vector[b_head1_p:end_head1_p + 1, :]
                head_embedding = words_to_entity(head_embedding)  # torch.Size([768])
                # print("head_embedding",head_embedding.size())

                b_relation1_p = words_postion[1][0] + 1
                end_relation1_p = words_postion[1][-1] + 1  # becauce of the CLS label
                relation_embedding1 = sentence_vector[b_relation1_p:end_relation1_p + 1, :]
                relation_embedding1 = words_to_entity(relation_embedding1)

                b_tail1_ = words_postion[2][0] + 1
                tail_embedding1 = sentence_vector[b_tail1_:b_tail1_ + 1, :]
                tail_embedding1 = words_to_entity(tail_embedding1)

                b_relation2_p = words_postion[3][0] + 1
                end_relation2_p = words_postion[3][-1] + 1  # becauce of the CLS label
                relation_embedding2 = sentence_vector[b_relation2_p:end_relation2_p + 1, :]
                relation_embedding2 = words_to_entity(relation_embedding2)

                b_tail2_ = words_postion[4][0] + 1
                end_tail2_ = words_postion[4][-1] + 1
                tail_embedding2 = sentence_vector[b_tail2_:end_tail2_ + 1, :]
                tail_embedding2 = words_to_entity(tail_embedding2)

                b_head2_p = words_postion[5][0] + 1
                end_head2_p = words_postion[5][-1] + 1  # becauce of the CLS label
                head_embedding2 = sentence_vector[b_head2_p:end_head2_p + 1, :]
                head_embedding2 = words_to_entity(head_embedding2)  # torch.Size([768])
                # print("head_embedding",head_embedding.size())

                b_relation3_p = words_postion[6][0] + 1
                end_relation3_p = words_postion[6][-1] + 1  # becauce of the CLS label
                relation_embedding3 = sentence_vector[b_relation3_p:end_relation3_p + 1, :]
                relation_embedding3 = words_to_entity(relation_embedding3)


                embes.append(head_embedding)
                embes.append(relation_embedding1)
                embes.append(tail_embedding1)

                embes.append(relation_embedding2)
                embes.append(tail_embedding2)
                embes.append(head_embedding2)
                embes.append(relation_embedding3)

                adjact_list = creat_adjact(list(range(len(words_postion))))
                # print("-----adjact_55list",adjact_list)
                #adjact_list[[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [6, 5]]

                m = -1
                for single_adj in adjact_list:
                    m = m + 1
                    if len(single_adj) == 2:
                        head = single_adj[0]
                        head_embeds = embes[head]
                        adj_nodes = single_adj[1]
                        adj_node_embeds = embes[adj_nodes]
                        # # sum the relation nodes embedding
                        updata_head_embedding = upgrade_node_embedding(head_embeds, adj_node_embeds, W1)
                        # # print(updata_head_embedding)
                        # # -----plus itsself

                        last_head_embedding = combine_entity_with_own_infomration(head_embeds, updata_head_embedding)
                        # entity relation add word
                        if m == 0:
                            for j in range(b_head1_p, end_head1_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        else:
                            for j in range(b_relation3_p, end_relation3_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                    else:
                        # [0, 1, 2]  (left, head, right)
                        center = single_adj[1]
                        center_embedding = embes[center]
                        adj_nodes_left = single_adj[0]
                        left_node_embedding = embes[adj_nodes_left]

                        adj_nodes_right = single_adj[2]
                        right_node_embedding = embes[adj_nodes_right]

                        updata_head_embedding_left = upgrade_node_embedding(center_embedding, left_node_embedding,
                                                                            W1)
                        updata_head_embedding_right = upgrade_node_embedding(center_embedding,
                                                                             right_node_embedding, W1)
                        updata_head_embedding = updata_head_embedding_right + updata_head_embedding_left
                        # -----plus itsself
                        # last_head_embedding = torch.mul((center_embedding + updata_head_embedding),
                        #                                 torch.mul(center_embedding, updata_head_embedding))
                        last_head_embedding = combine_entity_with_own_infomration(center_embedding, updata_head_embedding)
                        # entity relation add word
                        if m == 1:
                            for j in range(b_relation1_p, end_relation1_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        elif m == 2:
                            for j in range(b_tail1_, b_tail1_ + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)
                        elif m == 3:
                            for j in range(b_relation2_p, b_relation2_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)

                        elif m == 4:
                            for j in range(b_tail2_, end_tail2_ + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)

                        elif m == 5:
                            for j in range(b_head2_p, end_head2_p + 1):
                                sentence_vector[j, :] = combine_word_entity_infomration(sentence_vector[j, :],last_head_embedding)

            S.append(sentence_vector.unsqueeze(0))


        last_representation=(torch.cat(S, dim=0))
        # last_representation=torch.sum(last_representation,dim=1)
        # L=torch.cat([question_output_states[:,0,:],last_representation],dim=1)
        # L=torch.matmul(L,W4)
        # row = nn.Softmax(dim=1)
        # alpha_=row(L)
        # L=alpha_*L

        return  last_representation







