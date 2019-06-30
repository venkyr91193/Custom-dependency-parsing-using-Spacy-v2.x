import os
import sys

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'src')
sys.path.append(BASE_DIR)

from load_and_convert_data import LoadData

def test_load_data():
  tmp = LoadData()
  data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data','train.conllu')
  data = tmp.load_data(data_path)
  for dat in data:
    assert(len(dat[1]['heads']) == len(dat[1]['deps']))

def test_load_data_tokens():
  tmp = LoadData()
  data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data','train.conllu')
  data = tmp.load_data_tokens(data_path)
  for dat in data:
    assert(len(dat[1]['heads']) == len(dat[1]['deps']))
    assert(len(dat[1]['heads']) == len(dat[0]))

def test_load_load_data_tokens_spacy():
  tmp = LoadData()
  data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data','train.conllu')
  data = tmp.conllu_to_json(data_path,'en')
  for dat in data:
    assert(len(dat[1]['heads']) == len(dat[1]['deps']))
    assert(len(dat[1]['heads']) == len(dat[0]))

if __name__ == "__main__":
  #test_load_data()
  #test_load_data_tokens()
  test_load_load_data_tokens_spacy()
  print('All tests passed')
