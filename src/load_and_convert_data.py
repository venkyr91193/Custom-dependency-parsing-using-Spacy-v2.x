import os
from io import open

from conllu import parse_incr


class LoadData():

  @staticmethod
  def load_data(fpath):
    '''
    Converters the conllu data into spacy training data form.
    '''
    data = []
    f_pointer = open(fpath, "r", encoding="utf-8")
    for sentence in parse_incr(f_pointer):
      temp_dict = dict()
      temp_dict['heads'] = list()
      temp_dict['deps'] = list()
      for tok_info in sentence:
        temp_dict['heads'].append(tok_info['head'])
        temp_dict['deps'].append(tok_info['deprel'])
      data.append((sentence.metadata['text'],temp_dict))
    return data