# SSKGQA

### 1) Brief Introduction

This is the source code of SSKGQA. For the full project including MetaQA and WSP datasets, please refer to [Google Drive](https://drive.google.com/drive/folders/18ZREtZq7d1XW_7IfNcsAq5NEoMLDIcK-?usp=sharing)
. 

SSKGQA contains two steps:

**Step 1**： Use structure-BERT to predict the semantic strcture of a question.

**Step 2**： Use a BERT-based ranker to rank the candidate query graph.

If the query graph can be predicted correctly, the right answer can be retreved from KG.

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

cd metaQA_step1/hop1
CUDA_VISIBLE_DEVICES=1 python train.py
