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

"""Reconstructs the activation of sentence shifted towards a different context."""
import json
import os
from absl import app
from absl import flags
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(1, 'helpers')
import activation_helper
import attention_mask_helper
import embeddings_helper
import folder_helper
import inference_helper
import one_hots_helper
import optimization_helper
import output_helper
import setup_helper
import tokenization_helper

# Command Line Arguments
FLAGS = flags.FLAGS
flags.DEFINE_string('sentence', u'i hate kickshaws',
                    'the sentence to start with')
flags.DEFINE_string('sentence2', u'', 'an optional sencond sentence')
flags.DEFINE_string('output_dir', None,
                    'the output directory where the results will be'
                    'written.')
flags.DEFINE_string('shift_vector', None, 'the file that holds the vector'
                    'indicating which direction to shift to')
flags.DEFINE_string('model_config', 'bert-base-uncased', 'the name of the model'
                    'configuration to load')
flags.DEFINE_string('target', None, 'target of the shifted activation process')
flags.DEFINE_integer('num_iterations', 10, 'number of optimization steps')
flags.DEFINE_integer('layer_id', 5, 'layer to optimize activation for')
flags.DEFINE_integer('word_id', None, 'word to optimize activation for')
flags.DEFINE_integer('neuron_id', None, 'neuron to optimize activation for')
flags.DEFINE_integer('shift_start', 1, 'first token that is to be shifted in'
                     'the sentence')
flags.DEFINE_integer('shift_end', 1, 'last token that is to be shifted in the'
                     'sentence')
flags.DEFINE_integer('dream_start', 1, 'first token that is to be changed in'
                     'the sentence')
flags.DEFINE_integer('dream_end', 0, 'last token that is to be changed in the'
                     'sentence')
flags.DEFINE_integer('warmup', 200, 'how long before the temperature of the'
                     'softmax gets adjusted')
flags.DEFINE_integer('k', 10, 'number of top ranked words to store for each'
                     'iteration')
flags.DEFINE_integer('metrics_frequency', 250, 'frequency in which results are'
                     'saved')
flags.DEFINE_float('start_temp', 2.0, 'start-temperature of the softmax')
flags.DEFINE_float('end_temp', 0.1, 'end-temperature of the softmax')
flags.DEFINE_float('anneal', 0.9995, 'annealing factor for the temperature')
flags.DEFINE_float('learning_rate', 0.1, 'learning rate of the optimizer')
flags.DEFINE_float('shift_magnitude', 0.1, 'the magnitude with which to'
                   'change the activation vector')
flags.DEFINE_bool('gumbel', False, 'use gumbel noise with the softmax')
flags.DEFINE_bool('write_top_k', False, 'write top words for each iteration')


def shift_target_activation(target_activation, device):
  """Shift the target activation in the desired direction.

  Args:
    target_activation: The old target activation to be changed.
    device: Device to load variables to.

  Returns:
    target_activation: The new, changed target activation to optimize for.
  """
  shift_path = os.path.join(FLAGS.shift_vector, str(FLAGS.layer_id),
                            'final_weights.np')
  shift_file = open(shift_path, 'rb')
  shift = np.load(shift_file)
  shift_tensor = torch.tensor(shift)
  shift_tensor = shift_tensor.to(device)
  shift_tensor = shift_tensor.reshape(-1)
  shift_tensor = shift_tensor / torch.max(shift_tensor)
  shift_tensor = shift_tensor * FLAGS.shift_magnitude
  for i in range(FLAGS.shift_start, FLAGS.shift_end+1):
    target_activation[i] = target_activation[i] + shift_tensor
  return target_activation


def deep_dream(data, results, params, device, tokenizer, embedding_map, model):
  """Deep dream to a target activation.

  Args:
    data: Holds the top-k values.
    results: Holds the results of the run.
    params: Holds the parameters of the run.
    device: Where to place new variables.
    tokenizer: Used to convert between ids and tokens.
    embedding_map: Holding all BERT token embeddings.
    model: The model used for this dream.
  """
  # An embedding for the tokens is obtained
  tokens = tokenization_helper.tokenize_input_sentence(
      tokenizer, FLAGS.sentence, FLAGS.sentence2)
  tokens_tensor, segments_tensor = tokenization_helper.tensors_from_tokens(
      tokenizer, tokens, device)
  _, pos_embeddings, sentence_embeddings = embeddings_helper.get_embeddings(
      tokens_tensor, segments_tensor, model)
  # Correct the end of the dream if necessary
  if FLAGS.dream_end == 0:
    FLAGS.dream_end = len(tokens) - 2
  # Write the parameters to a file
  output_helper.get_params(params, FLAGS, tokens)
  # Get the smooth one-hot vector that is to be optimized, split into static and
  # modifiable parts
  before, modify, after = one_hots_helper.get_one_hots(
      tokens_tensor.data.cpu().numpy(), FLAGS.dream_start, FLAGS.dream_end,
      device)
  modify = torch.randn(modify.shape, device=device, requires_grad=True)
  # Obtain the default attention mask to be able to run the model
  att_mask = attention_mask_helper.get_attention_mask(tokens_tensor)
  # The optimizer used to modify the input embedding
  optimizer = torch.optim.Adam([modify], lr=FLAGS.learning_rate)
  # Init temperature for Gumbel
  temperature = torch.tensor(FLAGS.start_temp, device=device,
                             requires_grad=False)
  # Obtain the target activation we try to optimize towards.
  target_ids = tokens_tensor.data.cpu().numpy()[0]
  target_activation = activation_helper.get_ids_activation(
      target_ids, pos_embeddings, sentence_embeddings, att_mask,
      FLAGS.dream_start, FLAGS.dream_end, FLAGS.word_id, FLAGS.neuron_id,
      FLAGS.layer_id, False, embedding_map, model, device)
  target_activation = shift_target_activation(target_activation, device)
  target_activation = target_activation.clone().detach().requires_grad_(False)
  # Obtain the properties of the initial embedding
  one_hots_sm = one_hots_helper.softmax_one_hots(modify, temperature,
                                                 FLAGS.gumbel)
  max_values, token_ids = one_hots_helper.get_tokens_from_one_hots(
      torch.cat([before, one_hots_sm, after], dim=1))
  numpy_max_values = max_values.data.cpu().numpy()
  ids = token_ids.data.cpu().numpy()[0]
  tokens = tokenizer.convert_ids_to_tokens(ids)
  ids_activation = activation_helper.get_ids_activation(
      ids, pos_embeddings, sentence_embeddings, att_mask, FLAGS.dream_start,
      FLAGS.dream_end, FLAGS.word_id, FLAGS.neuron_id, FLAGS.layer_id, False,
      embedding_map, model, device)
  # Write the initial stuff for the results file
  output_helper.init_results(results)

  # Optimize the embedding for i iterations and update the properties to
  # evaluate the result in each step
  for i in range(FLAGS.num_iterations):
    # Do an optimization step
    max_vals, token_ids, loss = optimization_helper.step_towards_activation(
        optimizer, before, modify, after, pos_embeddings,
        sentence_embeddings, att_mask, temperature, i, FLAGS.gumbel,
        FLAGS.write_top_k, FLAGS.k, data, FLAGS.word_id, FLAGS.neuron_id,
        FLAGS.layer_id, FLAGS.dream_start, FLAGS.dream_end, tokenizer,
        embedding_map, model, target_activation)
    # Write the properties of the last step
    ids_loss = F.mse_loss(ids_activation, target_activation)
    if (i % FLAGS.metrics_frequency) == 0:
      output_helper.get_metrics(
          tokens, i, temperature, numpy_max_values, results,
          loss=loss, ids_loss=ids_loss)
    # Set the numpy max values
    numpy_max_values = max_vals.data.cpu().numpy()
    # Obtain the activation property for the id-array that would result from the
    # optimization
    ids = token_ids.data.cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    # Calculate the activation using the highest scoring words
    ids_activation = activation_helper.get_ids_activation(
        ids, pos_embeddings, sentence_embeddings, att_mask, FLAGS.dream_start,
        FLAGS.dream_end, FLAGS.word_id, FLAGS.neuron_id, FLAGS.layer_id, False,
        embedding_map, model, device)
    # Check if the temperature needs to decrease
    if i > FLAGS.warmup:
      temperature = torch.clamp(temperature * FLAGS.anneal, FLAGS.end_temp)

  # Calculate the final activation just as before, but without backprop
  if (FLAGS.num_iterations % FLAGS.metrics_frequency) == 0:
    with torch.no_grad():
      one_hots_sm = one_hots_helper.softmax_one_hots(modify, temperature,
                                                     FLAGS.gumbel)
      fused_one_hots = torch.cat([before, one_hots_sm, after], dim=1)
      if FLAGS.write_top_k:
        output_helper.write_top_ks(fused_one_hots, FLAGS.k,
                                   FLAGS.num_iterations, data,
                                   FLAGS.dream_start, FLAGS.dream_end,
                                   tokenizer)
      layers = inference_helper.run_inference(before, one_hots_sm, after,
                                              pos_embeddings,
                                              sentence_embeddings, att_mask,
                                              embedding_map, model)
      activation = activation_helper.get_activations(
          layers, FLAGS.word_id, FLAGS.neuron_id, FLAGS.layer_id)
      loss = F.mse_loss(activation, target_activation)
      ids_loss = F.mse_loss(ids_activation, target_activation)
      output_helper.get_metrics(
          tokens, FLAGS.num_iterations, temperature, numpy_max_values, results,
          loss=loss, ids_loss=ids_loss)


def reconstruct_shifted_activation(device, tokenizer, emb_map, model):
  """Reconstruct the activation for a given sentence after they have been shifted.

  Args:
    device: The device to use for training the model.
    tokenizer: Used to convert between sentences, tokens, and ids.
    emb_map: Map containing all the pretrained embeddings of the model.
    model: BERT model used for the dreaming process.
  """
  data = []
  results = {}
  params = {}
  # Create a folder for this experiment
  layer_dir = os.path.join(FLAGS.output_dir, str(FLAGS.layer_id))
  folder_helper.make_folder_if_not_exists(layer_dir)
  magnitude_dir = os.path.join(layer_dir, str(FLAGS.shift_magnitude))
  folder_helper.make_folder_if_not_exists(magnitude_dir)
  # Actually do the optimization
  deep_dream(data, results, params, device, tokenizer, emb_map, model)
  # If the top k file is to be written, write it
  if FLAGS.write_top_k:
    for i in range(len(data)):
      top_k_path = os.path.join(magnitude_dir, 'top_k' + str(i) + '.json')
      top_k_file = open(top_k_path, 'w')
      json.dump(data[i], top_k_file)
      top_k_file.close()
  output_helper.write_results(magnitude_dir, results, params,
                              'reconstruct_shifted')


def main(_):
  tokenizer, model, device, emb_map = setup_helper.setup_uncased(
      FLAGS.model_config)
  # Make a directory for the current run
  folder_helper.make_folder_if_not_exists(FLAGS.output_dir)
  # Start the run
  reconstruct_shifted_activation(device, tokenizer, emb_map, model)

if __name__ == '__main__':
  flags.mark_flag_as_required('output_dir')
  flags.mark_flag_as_required('shift_vector')
  flags.mark_flag_as_required('target')
  app.run(main)
