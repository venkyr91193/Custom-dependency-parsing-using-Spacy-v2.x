# Custom-dependency-parsing-using-Spacy-v2.1
Train your own dependency parser for any language.

# Train your own parser
You can train your own custom dependency parser in any launguage.

# About pretrained weights
Pretrained weights are used from https://github.com/honnibal/spacy-pretrain-polyaxon/tree/master/weights

This model is trained on 2 billion words of text from Reddit (the January 2017 portion of the Reddit comments corpus).
The pre-trained CNN is very small: it's depth 4, width 96,and has only 2000 rows in the hash embeddings table.
Weights for the serialized model are only 3.2 MB.

# install the dependencies
pip install -r requirements.txt

# To run
Run train.py with the required parameters