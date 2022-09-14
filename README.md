## SSKGQA: Semantic Structure based Query Graph Prediction for Question Answering over Knowledge Graph

### 1) Introduction

This is the source code of [SSKGQA](https://arxiv.org/abs/2204.10194). For the full project, including MetaQA and WSP datasets, please refer to [Google Drive](https://drive.google.com/drive/folders/18ZREtZq7d1XW_7IfNcsAq5NEoMLDIcK-?usp=sharing)

<img src="https://github.com/ToneLi/SSKGQA/blob/main/framework.png" width="500"/>

The overview of our proposed SSKGQA is depicted in this chart. Given a question **q** , we assume the topic entity of **q** has been obtained by preprocessing. Then the answer to **q** is generated by the following steps. 

**Step 1**： the semantic structure (we proposed six semantic structure as shown in Six-SS.pdf) of __q__ is predicted by a novel Structure-BERT classifier. For the example in the above, __q__ is a 2-hop question and the classifier predicts its semantic structure as __SS2__. 

**Step 2**: we retrieve all the candidate query graphs (CQG) of __q__ by enumeration, and use the predicted semantic structure __SS2__ as the constraint to filter out noisy candidate query graphs and keep the candidate query graphs with correct structure (CQG-CS). Afterwards, a BERT-based ranking model is used to score each candidate query graph in CQG-CS, and the top-1 highest scored candidate is selected as the query graph __g__ for question __q__. Finally, the selected query graph is issued to KG to retrieve the answer __Sergei Kozlov__.

Note: for the detail please check our paper.


### 2) Configuration

**Step1**:

1) Python>=3.6.5

2) Transformer: pip install transformers==3.0.0

3) Torch with 1.5.0: pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

4) Latest Tensorflow: pip install Tensorflow

**Step2**:

1) Python>=3.6.5

2) Transformer: pip install transformers==4.3.3

3) torch==1.7.1: pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html: 

### 3) How to run

For excample:

```markdown
the training file in step 1 and step2 all are train.py, just run it by
CUDA_VISIBLE_DEVICES=1 python train.py,  please use the default parameters.
```

### 4) Demo display

**Question**:   what is my timezone in louisiana ?

**groud truth query graph is**:  louisiana (topic entity)#location.location.time_zones (relation)#?x (the answer which to be got)#1  (SS1 for our proposed Semantic structure)

**candidate query graph**:
```
louisiana#base.biblioness.bibs_location.country#?x#1
louisiana#location.location.events#?x#1
louisiana#location.location.partially_contains#?x#1
**louisiana#location.location.time_zones#?x#1**
louisiana#government.political_district.representatives#?x#1
louisiana#location.administrative_division.country#?x#1
louisiana#base.aareas.schema.administrative_area.administrative_parent#?x#1
louisiana#location.administrative_division.first_level_division_of#?x#1
louisiana#location.location.partiallycontains#?x#1
louisiana#book.book_subject.works#?x#1
```
**Prediction results**:

<img src="https://github.com/ToneLi/SSKGQA/blob/main/demo.png" width="500"/>


### 5) Citation
Please cite this paper if you find the paper or code is useful.
```
@article{li2022semantic,
  title={Semantic Structure based Query Graph Prediction for Question Answering over Knowledge Graph},
  author={Li, Mingchen and Ji, Shihao},
  journal={International Conference on Computational Linguistics (COLING)},
  year={2022}
}
```
