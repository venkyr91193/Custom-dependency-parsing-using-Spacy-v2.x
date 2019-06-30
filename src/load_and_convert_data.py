import json
import os
from io import open

from conllu import parse_incr
from spacy.cli import convert


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

  @staticmethod
  def load_data_tokens(fpath):
    '''
    Converters the spacy json data into spacy training data form except
    it replaces the text information by tokens list.

    fpath : conllu format file
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
      data.append(([item['form'] for item in sentence],temp_dict))
    return data
  
  @staticmethod
  def load_data_tokens_spacy(fpath):
    '''
    Converters the spacy json data into spacy training data form except
    it replaces the text information by tokens list.

    fpath : json format file
    '''
    data = []
    with open(fpath, "r", encoding="utf-8") as f:
      temp_data = json.loads(f.read())
    for dat in temp_data:
      for para in dat['paragraphs']:
        for sent in para['sentences']:
          temp_dict = dict()
          temp_list = list()
          temp_dict['heads'] = list()
          temp_dict['deps'] = list()
          for tok in sent['tokens']:
            temp_list.append(tok['orth'])
            temp_dict['heads'].append(tok['head'])
            temp_dict['deps'].append(tok['dep'])
          data.append((temp_list,temp_dict))
      return data
          
  @staticmethod
  def conllu_to_json(fpath,lang):
    '''
    Converters the conllu data into spacy training data in json format
    
    fpath : conllu format file
    '''
    filename = fpath.rsplit('.',1)[0]
    dirname = os.path.dirname(fpath.rsplit('.',1)[0])
    convert(fpath,dirname,"json",1,False,"auto",lang)
    return LoadData.load_data_tokens_spacy(filename+'.json')
