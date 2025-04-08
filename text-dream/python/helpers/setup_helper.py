# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Provides the setup for the experiments."""
from pytorch_pretrained_bert import modeling
from pytorch_pretrained_bert import tokenization
import torch
import embeddings_helper


def setup_uncased(model_config):
  """Setup the uncased bert model.

  Args:
    model_config: The model configuration to be loaded.

  Returns:
    tokenizer: The tokenizer to be used to convert between tokens and ids.
    model: The model that has been initialized.
    device: The device to be used in this run.
    embedding_map: Holding all token embeddings.
  """
  # Load pre-trained model tokenizer (vocabulary)
  tokenizer = tokenization.BertTokenizer.from_pretrained(model_config)
  # Load pre-trained model (weights)
  model = modeling.BertModel.from_pretrained(model_config)
  _ = model.eval()
  # Set up the device in use
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print('device : ', device)
  model = model.to(device)
  # Initialize the embedding map
  embedding_map = embeddings_helper.EmbeddingMap(device, model)
  return tokenizer, model, device, embedding_map


def setup_bert_vanilla(model_config):
  """Setup the uncased bert model without embedding maps.

  Args:
    model_config: The model configuration to be loaded.

  Returns:
    tokenizer: The tokenizer to be used to convert between tokens and ids.
    model: The model that has been initialized.
    device: The device to be used in this run.
  """
  # Load pre-trained model tokenizer (vocabulary)
  tokenizer = tokenization.BertTokenizer.from_pretrained(model_config)
  # Load pre-trained model (weights)
  model = modeling.BertModel.from_pretrained(model_config)
  _ = model.eval()
  # Set up the device in use
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print('device : ', device)
  model = model.to(device)
  return tokenizer, model, device


def setup_bert_mlm(model_config):
  """Setup the uncased bert model with classification head.

  Args:
    model_config: The model configuration to be loaded.

  Returns:
    tokenizer: The tokenizer to be used to convert between tokens and ids.
    model: The model that has been initialized.
    device: The device to be used in this run.
  """
  # Load pre-trained model tokenizer (vocabulary)
  tokenizer = tokenization.BertTokenizer.from_pretrained(model_config)
  # Load pre-trained model (weights)
  model = modeling.BertForMaskedLM.from_pretrained('bert-base-uncased')
  _ = model.eval()
  # Set up the device in use
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print('device : ', device)
  model = model.to(device)
  # Initialize the embedding map
  embedding_map = embeddings_helper.EmbeddingMap(device, model.bert)
  return tokenizer, model, device, embedding_map
