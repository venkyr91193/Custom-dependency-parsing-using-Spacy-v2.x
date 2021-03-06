import argparse
import os
import random
from pathlib import Path

import spacy
from seqeval.metrics import classification_report, f1_score
from spacy.cli import train
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.util import compounding, decaying, minibatch

from load_and_convert_data import LoadData


class Train():
  def __init__(self,args):
    self.gpu = args.gpu
    self.model = args.model
    self.lang = args.lang
    self.train_path = args.train_path
    self.test_path = args.test_path
    self.output_dir = args.output_dir
    self.n_iter = args.n_iter
    self.todi = args.type_of_data_input
  
  def require_gpu(self,gpu):
    if gpu >= 0:
      spacy.prefer_gpu()

  def get_data(self,fpath):
    tmp = LoadData()
    if self.todi == 'conllu_2_spacy':
      data = tmp.conllu_to_json(fpath,self.lang)
    elif self.todi == 'conllu_2_text':
      data = tmp.load_data(fpath)
    elif self.todi == 'conllu_2_tokens':
      data = tmp.load_data_tokens(fpath)
    return data

  def prevent_sentence_boundary_detection(self,doc):
      for token in doc:
          # This will entirely disable spaCy's sentence detection
          token.is_sent_start = False
      return doc

  def train(self):
    '''
    Using spacy V2.1 to train the dependency parser
    '''
    # getting data in spacy required format
    train_data = self.get_data(self.train_path)
    test_data = self.get_data(self.test_path)

    # pretrained weights
    pretrain_weights_path = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)),\
      'pretrained_weights','pretrained_weights.bin'))

    train_filename = self.train_path.rsplit('.',1)[0] + '.json'
    dev_filename = self.test_path.rsplit('.',1)[0] + '.json'

    # call the train function
    train(self.lang,Path(self.output_dir),Path(train_filename),Path(dev_filename),raw_text=None,\
          base_model=None,pipeline="parser",vectors=None,n_iter=self.n_iter,\
          n_early_stopping=None,n_examples=0,use_gpu=self.gpu,version="0.0.0",\
          meta_path=None,init_tok2vec=pretrain_weights_path,parser_multitasks="",\
          entity_multitasks="",noise_level=0.0,eval_beam_widths="",gold_preproc=False,\
          learn_tokens=False,verbose=False,debug=False)

  def update(self):
    '''
    Using spacy V2.1 to update the dependency parser
    '''
    dropout = decaying(0.5, 0.2, 1e-4)

    self.require_gpu(self.gpu)

    # getting data in spacy required format
    data = self.get_data(self.train_path)

    random.seed(777)
    random.shuffle(data)

    if self.model is not None:
        nlp = spacy.load(self.model)  # load existing spaCy model
        print("Loaded model '%s'" % self.model)
    else:
        nlp = spacy.blank(self.lang)  # create blank Language class
        print("Created blank '%s' model" % self.lang)

    # add the parser to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "parser" not in nlp.pipe_names:
        parser = nlp.create_pipe("parser")
        nlp.add_pipe(parser, first=True)
    # otherwise, get it, so we can add labels to it
    else:
        parser = nlp.get_pipe("parser")

    # sentence segmentation diable
    #nlp.add_pipe(self.prevent_sentence_boundary_detection, name='prevent-sbd', before='parser')

    # change the tokens to spacy Doc
    new_data = list()
    for dat in data:
      assert(len(Doc(nlp.vocab, words=dat[0])) == len(dat[1]['deps']))
      assert(len(Doc(nlp.vocab, words=dat[0])) == len(dat[1]['heads']))
      doc = Doc(nlp.vocab, words=dat[0])
      new_data.append((doc,GoldParse(doc,heads=dat[1]['heads'],deps=dat[1]['deps'])))

    # add labels to the parser
    for _, annotations in data:
      for dep in annotations.get("deps", []):
        parser.add_label(dep)

    pretrain_weights_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),\
      'pretrained_weights','pretrained_weights.bin')
    if os.path.exists(pretrain_weights_path):
      # loading pretrained weights
      with open(pretrain_weights_path, "rb") as file_:
        nlp.from_bytes(file_.read())
        print('LOADED from PRETRAIN %s' % pretrain_weights_path)

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "parser"]
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(self.n_iter):
            random.shuffle(new_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(new_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                print(type(texts[0]))
                print(annotations[0])
                nlp.update(texts, annotations, sgd=optimizer, losses=losses, drop=next(dropout))
            print("Losses", losses)

    # save model to output directory
    if self.output_dir is not None:
        self.output_dir = Path(self.output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir()
        nlp.meta['name'] = "Custom-launguage-model"  # rename model
        with nlp.use_params(optimizer.averages):
          nlp.to_disk(self.output_dir)
          print("Saved model to", self.output_dir)

  def evaluate(self):
    '''
    To load and evaluate the model
    '''
    # Test the saved model
    print("Loading from", self.output_dir)
    nlp2 = spacy.load(self.output_dir)

    # Evaluating the model
    data = self.get_data(self.test_path)

    # pred dep
    pred = list()
    # true list
    true = list()

    for dat in data:
      temp_t = dat[1]['deps']

      doc = nlp2(dat[0])
      temp_p = list()
      temp_p = [t.dep_ for t in doc]
      pred.append(temp_p)
      true.append(temp_t)
    
    print("Validation F1-Score: {}".format(f1_score(pred, true)))
    print("Classification Report")
    print(classification_report(pred, true))    

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_path",
                        type=str,
                        required=True,
                        default=None,
                        help="Data source path for train data in conllu format")
  parser.add_argument("--test_path",
                        type=str,
                        required=True,
                        default=None,
                        help="Data source path for test data in conllu format")
  parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        default=None,
                        help="Directory to save the model")
  parser.add_argument("--lang",
                        type=str,
                        required=True,
                        default=None,
                        help="Language of the model")
  # optional
  parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="Whether to use GPU.")
  parser.add_argument("--n_iter",
                        type=int,
                        default=5,
                        help="Number of iterations required")
  parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="Base model to train on or blank model\nExample:  en_core_web_lg")
  parser.add_argument("--do_update",
                        action='store_true',
                        help="Set flag to update instead of train")
  parser.add_argument("--type_of_data_input",
                        type=str,
                        default='conllu_2_spacy',
                        help="'conllu_2_spacy' -> Converts .conull to .json format in spacy.\n\
                              'conllu_2_text' -> Converts .conull to spacy required form\n \
                              'conllu_2_tokens' -> Converts .conull to spacy required form but tokens instead of text")
  args = parser.parse_args()
  obj = Train(args)
  if args.do_update:
    obj.update()
  else:
    obj.train()
