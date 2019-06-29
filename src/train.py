import argparse
import os
import random
from pathlib import Path
from seqeval.metrics import classification_report, f1_score

import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
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
  
  def require_gpu(self,gpu):
    if gpu:
      spacy.prefer_gpu()

  def get_data(self,fpath):
    tmp = LoadData()
    data = tmp.load_data(fpath)
    return data
  
  def train(self):
    '''
    Using spacy V2.1 to train the dependency parser
    '''

    self.require_gpu(self.gpu)

    # getting data in spacy required format
    data = self.get_data(self.train_path)

    random.seed(770)
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
            random.shuffle(data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
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
                        help="Data source path for train")
  parser.add_argument("--test_path",
                        type=str,
                        required=True,
                        default=None,
                        help="Data source path for test")
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
                        action='store_true',
                        help="Whether to use GPU.")
  parser.add_argument("--n_iter",
                        type=int,
                        default=5,
                        help="Number of iterations required")
  parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="Base model to train on or blank model\nExample:  en_core_web_lg")
  args = parser.parse_args()
  obj = Train(args)
  obj.train()
