# SSKGQA

### 1) Short Introduction

This is the source code about our project, the full project, please refer: [Google Drive](https://drive.google.com/drive/folders/18ZREtZq7d1XW_7IfNcsAq5NEoMLDIcK-?usp=sharing)
. 

There are two steps in our project,

**Step 1**： By using the structureBERT to predict the SS.

**Step 2**： By using the triple BERT to rank the candidate query graph. So, for for each dataset, we have two steps. For excample: MetaQA_step1 and MetaQA_step2.

So, if we can predict the query graph correctly, the answer will also be correctly answerd.

### 2) Configuration
**Step1**:

1)Python>=3.6.5

2)Transformer: pip install transformers==3.0.0

3)Torch with 1.5.0: pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html


4) The latest version about Tensorflow, you can just pip install Tensorflow

**Step2**:

1)Python>=3.6.5

2)Transformer: pip install transformers==4.3.3

3)torch==1.7.1: pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html: 

### 3) How to run

Enter each file, to find the train.py, just run  CUDA_VISIBLE_DEVICES=1 python train.py.

For excample:

enter: metaQA_step1/hop1,  run **CUDA_VISIBLE_DEVICES=1 python train.py**.
