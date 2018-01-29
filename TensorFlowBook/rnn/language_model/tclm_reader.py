# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

MAX_CHAR = 256
BOF = MAX_CHAR + 1
EOF = MAX_CHAR + 2
VOCAB_SIZE = MAX_CHAR + 3


def get_file_suffix(file):
    index = file.rfind('.')
    return file[index + 1:] if index >= 0 else ''


def read_source_code_data(code_files):
    data = []
    for code_file in code_files:
        file_r = open(code_file, 'r')
        curr_data = []
        curr_data.append(BOF)
        for dataline in file_r:
            for c in dataline:
                curr_data.append(ord(c))
        curr_data.append(EOF)
        data.extend(curr_data)
        file_r.close()
    return data


def tensorflow_code_data(data_path=None):
    # find all python source code
    tensorflow_code_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            code_file = os.path.join(root, file)
            file_suffix = get_file_suffix(code_file)
            if file_suffix == 'py':
                tensorflow_code_files.append(code_file)

    train_code_file_count = int(len(tensorflow_code_files) * 0.5)
    valid_code_file_count = int(len(tensorflow_code_files) * 0.3)
    # test_code_file_count = int(len(tensorflow_code_files) * 0.2)
    train_data = read_source_code_data(
        tensorflow_code_files[0: train_code_file_count])
    valid_data = read_source_code_data(tensorflow_code_files[
                                       train_code_file_count: train_code_file_count + valid_code_file_count])
    test_data = read_source_code_data(
        tensorflow_code_files[train_code_file_count + valid_code_file_count:])
    return train_data, valid_data, test_data


def tensorflow_code_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.
        name: the name of this operation (optional).

    Returns:
        A pair of Tensors, each shaped [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the right by one.

    Raises:
        tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "TensorflowCodeProducer",
                       [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data",
                                        dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
        y.set_shape([batch_size, num_steps])
        return x, y
