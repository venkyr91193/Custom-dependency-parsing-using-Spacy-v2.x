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


if __name__ == "__main__":
  test_load_data()
  print('All tests passed')
