# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
import xml.etree.ElementTree as ET
import re
import itertools
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMultipleChoice
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
          datefmt = '%m/%d/%Y %H:%M:%S',
          level = logging.INFO)
logger = logging.getLogger(__name__)


class RaceExample(object):
  """A single training/test example for the RACE dataset."""
  '''
  For RACE dataset:
  race_id: data id
  context_sentence: article
  start_ending: question
  ending_0/1/2/3: option_0/1/2/3
  label: true answer
  '''
  def __init__(self,
         race_id,
         context_sentence,
         start_ending,
         ending_0,
         ending_1,
         ending_2,
         ending_3,
         label = None):
    self.race_id = race_id
    self.context_sentence = context_sentence
    self.start_ending = start_ending
    self.endings = [
      ending_0,
      ending_1,
      ending_2,
      ending_3,
    ]
    self.label = label

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    l = [
      f"id: {self.race_id}",
      f"article: {self.context_sentence}",
      f"question: {self.start_ending}",
      f"option_0: {self.endings[0]}",
      f"option_1: {self.endings[1]}",
      f"option_2: {self.endings[2]}",
      f"option_3: {self.endings[3]}",
    ]

    if self.label is not None:
      l.append(f"label: {self.label}")

    return ", ".join(l)



class InputFeatures(object):
  def __init__(self,
         example_id,
         choices_features,
         label

  ):
    self.example_id = example_id
    self.choices_features = [
      {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids
      }
      for _, input_ids, input_mask, segment_ids in choices_features
    ]
    self.label = label



## paths is a list containing all paths
def read_race_examples(paths):
  examples = []
  for path in paths:
    filenames = glob.glob(path+"/*txt")
    for filename in filenames:
      with open(filename, 'r', encoding='utf-8') as fpr:
        data_raw = json.load(fpr)
        article = data_raw['article']
        ## for each qn
        for i in range(len(data_raw['answers'])):
          truth = ord(data_raw['answers'][i]) - ord('A')
          question = data_raw['questions'][i]
          options = data_raw['options'][i]
          examples.append(
            RaceExample(
              race_id = filename+'-'+str(i),
              context_sentence = article,
              start_ending = question,

              ending_0 = options[0],
              ending_1 = options[1],
              ending_2 = options[2],
              ending_3 = options[3],
              label = truth))
   
  return examples 

## paths is a list containing all paths
def read_center_examples(input_list):
  examples = []
  for dic in input_list:
    article = ''.join(dic['body'])
    question = ''.join(dic['data'])
    options = dic['choices']
    answer = dic['answer']
    anscol = dic['anscol']
    anscol = anscol.replace('A', '')
    options_cat = []
    for option in options:
      option_cat = re.sub('[①-⑨] ', '', option)
      options_cat.append(option_cat)
      
    if re.search(anscol,question):
      qa_cat = re.sub(anscol, '_', question)
    else:
      qa_cat = question
    examples.append(
      RaceExample(
        race_id = 'center',
        context_sentence = article,
        start_ending = qa_cat,

        ending_0 = options_cat[0],
        ending_1 = options_cat[1],
        ending_2 = options_cat[2],
        ending_3 = options_cat[3],
        label = (int(answer) -1 )))
  return examples 

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                 is_training):
  """Loads a data file into a list of `InputBatch`s."""

  # RACE is a multiple choice task. To perform this task using Bert,
  # we will use the formatting proposed in "Improving Language
  # Understanding by Generative Pre-Training" and suggested by
  # @jacobdevlin-google in this issue
  # https://github.com/google-research/bert/issues/38.
  #
  # The input will be like:
  # [CLS] Article [SEP] Question + Option [SEP]
  # for each option 
  # 
  # The model will output a single value for each input. To get the
  # final decision of the model, we will run a softmax over these 4
  # outputs.
  features = []
  for example_index, example in enumerate(examples):
    context_tokens = tokenizer.tokenize(example.context_sentence)
    start_ending_tokens = tokenizer.tokenize(example.start_ending)

    choices_features = []
    for ending_index, ending in enumerate(example.endings):
      # We create a copy of the context tokens in order to be
      # able to shrink it according to ending_tokens
      context_tokens_choice = context_tokens[:]
      ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
      # Modifies `context_tokens_choice` and `ending_tokens` in
      # place so that the total length is less than the
      # specified length. Account for [CLS], [SEP], [SEP] with
      # "- 3"
      _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

      tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
      segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding = [0] * (max_seq_length - len(input_ids))
      input_ids += padding
      input_mask += padding
      segment_ids += padding

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      choices_features.append((tokens, input_ids, input_mask, segment_ids))

    label = example.label
    ## display some example
    if example_index < 1:
      logger.info("*** Example ***")
      logger.info(f"race_id: {example.race_id}")
      for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
        logger.info(f"choice: {choice_idx}")
        logger.info(f"tokens: {' '.join(tokens)}")
        logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
        logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
        logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
      if is_training:
        logger.info(f"label: {label}")

    features.append(
      InputFeatures(
        example_id = example.race_id,
        choices_features = choices_features,
        label = label
      )
    )

  return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def accuracy(out, labels):
  outputs = np.argmax(out, axis=1)
  return np.sum(outputs == labels)

def select_field(features, field):
  return [
    [
      choice[field]
      for choice in feature.choices_features
    ]
    for feature in features
  ]

def warmup_linear(x, warmup=0.002):
  if x < warmup:
    return x/warmup
  return 1.0 - x

def xml_parser(input_file):
  tree = ET.parse(input_file)
  root = tree.getroot()
  part_number = 0
  PART_NUMBER = 5
  dic_question = {}
  dic_body = {}
  list_question = []

  for parts in root:
    # 対象とする大問番号を選択する
    if parts.tag == 'question':
      part_number += 1
      if part_number != PART_NUMBER:
        continue
    for middle_parts in parts:
      # print(middle_parts.tag,middle_parts.attrib)
      dic_question = middle_parts.attrib
      for question_element in middle_parts: 
        # print(question_element.tag,question_element.attrib)
        for question_info in question_element.iter('paragraph'):
          dic_body.setdefault('body', []).append(data_parser(question_info))
        if question_element.tag == 'data':
          dic_question.setdefault('data', []).append(data_parser(question_element))
          # print(data_parser(question_element)) 
        elif question_element.tag == 'choices':
          for choice in question_element:
            dic_question.setdefault('choices', []).append(data_parser(choice))
          dic_question.update(dic_body)
          list_question.append(dic_question)
          dic_question = {}
        else :
          continue
  return list_question

def json_parser(input_file):
  with open(input_file) as f:
    list_answer = json.load(f)
  return list_answer

def make_test_list(list_question, list_answer):
  if len(list_question) != len(list_answer):
    print('list_question length is = ' + str(len(list_question)) + ' but,list_answer length is = ' + str(len(list_answer)))
    return -1
  for i in range(len(list_answer)):
    list_question[i].update(list_answer[i])
  return list_question

def data_parser(node):
  data = ''.join(node.itertext()).strip()
  data = re.sub('[ 　]+', ' ', data)
  data = data.replace('\n', '')
  return data


def main():
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the .csv files (or other data files) for the task.")
  parser.add_argument("--bert_model", default=None, type=str, required=True,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
               "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
  parser.add_argument("--output_dir",
            default=None,
            type=str,
            required=True,
            help="The output directory where the model checkpoints will be written.")

  ## Other parameters
  parser.add_argument('--question_file', 
            default='data/center-2017-2015/dev/Center-2017--Main-Eigo_hikki.xml',
            type=str, 
            help='test file')
  parser.add_argument('--answer_file', 
            type=str, 
            default='data/center-2017-2015/answer/Center-2017--Main-Eigo_hikki_part5.json',
            help='answer file')
  parser.add_argument('--result_file', 
            default='result.txt',
            type=str, 
            help='result file')
  parser.add_argument('--answer_file', 
            type=str, 
            default='data/center-2017-2015/answer/Center-2017--Main-Eigo_hikki_part5.json',
            help='answer file')
  parser.add_argument("--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
               "Sequences longer than this will be truncated, and sequences shorter \n"
               "than this will be padded.")
  parser.add_argument("--do_eval",
            default=False,
            action='store_true',
            help="Whether to run eval on the dev set.")
  parser.add_argument("--do_lower_case",
            default=False,
            action='store_true',
            help="Set this flag if you are using an uncased model.")
  parser.add_argument("--train_batch_size",
            default=32,
            type=int,
            help="Total batch size for training.")
  parser.add_argument("--eval_batch_size",
            default=8,
            type=int,
            help="Total batch size for eval.")
  parser.add_argument("--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.")
  parser.add_argument("--num_train_epochs",
            default=3.0,
            type=float,
            help="Total number of training epochs to perform.")
  parser.add_argument("--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
               "E.g., 0.1 = 10%% of training.")
  parser.add_argument("--no_cuda",
            default=False,
            action='store_true',
            help="Whether not to use CUDA when available")
  parser.add_argument("--local_rank",
            type=int,
            default=-1,
            help="local_rank for distributed training on gpus")
  parser.add_argument('--seed',
            type=int,
            default=42,
            help="random seed for initialization")
  parser.add_argument('--gradient_accumulation_steps',
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument('--fp16',
            default=False,
            action='store_true',
            help="Whether to use 16-bit float precision instead of 32-bit")
  parser.add_argument('--loss_scale',
            type=float, default=0,
            help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
               "0 (default value): dynamic loss scaling.\n"
               "Positive power of 2: static loss scaling value.\n")

  args = parser.parse_args()

  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
  else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
  logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args.local_rank != -1), args.fp16))

  if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                          args.gradient_accumulation_steps))

  args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

  os.makedirs(args.output_dir, exist_ok=True)

  tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

  if args.local_rank != -1:
    try:
      from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model)
  elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

  output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
  model_state_dict = torch.load(output_model_file)
  model = BertForMultipleChoice.from_pretrained(args.bert_model,
    state_dict=model_state_dict,
    cache_dir=None,
    num_choices=4)
  model.to(device)

  
  ## test middle
  list_question = xml_parser(args.question_file)
  list_answer = json_parser(args.answer_file)
  list_test = make_test_list(list_question, list_answer)
  eval_examples = read_center_examples(list_test)
  eval_features = convert_examples_to_features(
    eval_examples, tokenizer, args.max_seq_length, True)
  logger.info("***** Running evaluation: test middle *****")
  logger.info(" Num examples = %d", len(eval_examples))
  logger.info(" Batch size = %d", args.eval_batch_size)
  all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
  all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
  all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
  all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
  eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
  # Run prediction for full data
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

  model.eval()
  middle_eval_loss, middle_eval_accuracy = 0, 0
  middle_nb_eval_steps, middle_nb_eval_examples = 0, 0
  print('pre','ans')
  for step, batch in enumerate(eval_dataloader):
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, segment_ids, label_ids = batch

    with torch.no_grad():
      tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
      logits = model(input_ids, segment_ids, input_mask)

    logits = logits.detach().cpu().numpy()
    label_ids = label_ids.to('cpu').numpy()
    tmp_eval_accuracy = accuracy(logits, label_ids)
    
    print(np.argmax(logits, axis=1), label_ids)

    middle_eval_loss += tmp_eval_loss.mean().item()
    middle_eval_accuracy += tmp_eval_accuracy

    middle_nb_eval_examples += input_ids.size(0)
    middle_nb_eval_steps += 1

  eval_loss = middle_eval_loss / middle_nb_eval_steps
  eval_accuracy = middle_eval_accuracy / middle_nb_eval_examples

  result = {'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy}

  # with open(args.result_file, "w+") as writer:
  #   for key in sorted(result.keys()):
  #     logger.info(" %s = %s", key, str(result[key]))
  #     writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
  main()