# coding=utf-8
#
# Modifications Copyright 2021 by Michalis Titsias, Jiaxin Shi
# from https://github.com/alekdimi/arms
# and https://github.com/google-research/google-research/tree/master/disarm/binary
#
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Dataset for DisARM experiments."""
import os
import scipy.io
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp


tfd = tfp.distributions


default_omniglot_url = (
    "https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat")


def get_continuous_mnist_batch(batch_size, seed, fashion_mnist=False):
  """Get MNIST that has continuous values between [-1, 1]."""
  def _preprocess(x):
    return 2 * (tf.cast(x["image"], tf.float32) / 255.) - 1.

  if fashion_mnist:
    dataset_name = "fashion_mnist"
  else:
    dataset_name = "mnist:3.*.*"
  train, valid, test = tfds.load(
      dataset_name,
      split=["train[:50000]", "train[50000:]", "test"],
      shuffle_files=False)

  train = (train.map(_preprocess)
           .shuffle(1024, seed=seed)
           .repeat()
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  valid = (valid.map(_preprocess)
           .shuffle(1024, seed=seed)
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  test = (test.map(_preprocess)
          .shuffle(1024, seed=seed)
          .batch(batch_size)
          .prefetch(tf.data.experimental.AUTOTUNE))
  return train, valid, test


def get_binarized_mnist_batch(batch_size, seed):
  """Get MNIST that is binarized by tf.cast(x > .5, tf.float32)."""
  def _preprocess(x):
    return tf.cast(
        (tf.cast(x["image"], tf.float32) / 255.) > 0.5,
        tf.float32)

  train, valid, test = tfds.load(
      "mnist:3.*.*",
      split=["train[:50000]", "train[50000:]", "test"],
      shuffle_files=False)

  train = (train.map(_preprocess)
           .shuffle(1024, seed=seed)
           .repeat()
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  valid = (valid.map(_preprocess)
           .shuffle(1024, seed=seed)
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  test = (test.map(_preprocess)
          .shuffle(1024, seed=seed)
          .batch(batch_size)
          .prefetch(tf.data.experimental.AUTOTUNE))
  return train, valid, test


def get_dynamic_mnist_batch(batch_size, seed, fashion_mnist=False):
  """Transforms data based on args (assumes images in [0, 255])."""

  def _preprocess(x):
    """Sample dynamic image."""
    return tfd.Bernoulli(probs=tf.cast(x["image"], tf.float32) / 255.).sample(seed=seed)

  if fashion_mnist:
    dataset_name = "fashion_mnist"
  else:
    dataset_name = "mnist:3.*.*"
  train, valid, test = tfds.load(
      dataset_name,
      split=["train[:50000]", "train[50000:]", "test"],
      shuffle_files=False)

  train = (train.map(_preprocess)
           .shuffle(1024, seed=seed)
           .repeat()
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  valid = (valid.map(_preprocess)
           .shuffle(1024, seed=seed)
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  test = (test.map(_preprocess)
          .shuffle(1024, seed=seed)
          .batch(batch_size)
          .prefetch(tf.data.experimental.AUTOTUNE))
  return train, valid, test


def get_static_mnist_batch(batch_size, seed):
  """Get static MNIST dataset with tfds."""
  preprocess = lambda x: tf.cast(x["image"], tf.float32)
  mnist_dataset = tfds.load("binarized_mnist")
  train_ds, valid_ds, test_ds = [
      mnist_dataset[tag].map(preprocess)
      for tag in ["train", "validation", "test"]]
  train_ds = train_ds.repeat().shuffle(1024, seed=seed).batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)
  valid_ds = valid_ds.shuffle(1024, seed=seed).batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)
  test_ds = test_ds.shuffle(1024, seed=seed).batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)
  return train_ds, valid_ds, test_ds


def get_continuous_omniglot_batch(
    batch_size,
    seed,
    data_dir,
    omniglot_url=default_omniglot_url):
  """Load omnigload (assumes images in [0., 1.])."""
  def _preprocess(x):
    """Sample dynamic image."""
    return 2 * tf.cast(x, tf.float32) - 1.

  try:
    data_path = os.path.join(data_dir, "omniglot", "chardata.mat")
    with open(data_path, "rb") as f:
      omni_raw = scipy.io.loadmat(f)
  except:
    with open(omniglot_url, "rb") as f:
      omni_raw = scipy.io.loadmat(f)

  num_valid = 1345  # number of validation sample
  train_data, test = omni_raw["data"], omni_raw["testdata"]
  train_data = train_data.T.reshape([-1, 28, 28, 1])
  test = test.T.reshape([-1, 28, 28, 1])
  train, valid = train_data[:-num_valid], train_data[-num_valid:]

  train = (tf.data.Dataset.from_tensor_slices(train)
           .map(_preprocess)
           .shuffle(1024, seed=seed)
           .repeat()
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  valid = (tf.data.Dataset.from_tensor_slices(valid)
           .map(_preprocess)
           .shuffle(1024, seed=seed)
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  test = (tf.data.Dataset.from_tensor_slices(test)
          .map(_preprocess)
          .shuffle(1024, seed=seed)
          .batch(batch_size)
          .prefetch(tf.data.experimental.AUTOTUNE))
  return train, valid, test


def get_omniglot_batch(
    batch_size,
    seed,
    data_dir,
    omniglot_url=default_omniglot_url):
  """Load omnigload (assumes images in [0., 1.])."""
  def _preprocess(x):
    """Sample dynamic image."""
    return tfd.Bernoulli(probs=x).sample()

  try:
    data_path = os.path.join(data_dir, "omniglot", "chardata.mat")
    with open(data_path, "rb") as f:
      omni_raw = scipy.io.loadmat(f)
  except:
    with open(omniglot_url, "rb") as f:
      omni_raw = scipy.io.loadmat(f)

  num_valid = 1345  # number of validation sample
  train_data, test = omni_raw["data"], omni_raw["testdata"]
  train_data = train_data.T.reshape([-1, 28, 28, 1])
  test = test.T.reshape([-1, 28, 28, 1])
  train, valid = train_data[:-num_valid], train_data[-num_valid:]

  train = (tf.data.Dataset.from_tensor_slices(train)
           .map(_preprocess)
           .shuffle(1024, seed=seed)
           .repeat()
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  valid = (tf.data.Dataset.from_tensor_slices(valid)
           .map(_preprocess)
           .shuffle(1024, seed=seed)
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  test = (tf.data.Dataset.from_tensor_slices(test)
          .map(_preprocess)
          .shuffle(1024, seed=seed)
          .batch(batch_size)
          .prefetch(tf.data.experimental.AUTOTUNE))
  return train, valid, test
