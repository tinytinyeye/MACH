from pathlib import Path
import os
import time
import string
import random
import numpy as np
import tensorflow as tf

def get_meta(filename):
    """
    obtain number of samples. features and classes from given input file

    parameters:
    filename - input dataset, in the format of .vw.gz

    returns:
    num_points - number of samples in the dataset
    num_features - number of features per samples
    num_labels - number of classes in the dataset
    """
    num_points = 0
    label_dict = {}
    feature_dict = {}
    # max_feature = 0
    with open(filename, 'r') as f:
        # lines = [x.decode('utf8').strip() for x in f.readlines()]
        for line in f:
            line = line.strip()
            num_points += 1
            if num_points % 10000 == 0:
                print("read {} lines/r".format(num_points))
            data = line.split('|')
            label = data[0].strip().split(' ')[0]
            if (label not in label_dict):
                label_dict[int(label)] = 1
            features_str = data[1].strip()
            features = features_str.split(' ')
            for i in range(len(features)):
                feature = features[i].split(':')[0]
                feature_dict[int(feature)] = 1
                # if int(feature) > max_feature:
                #     max_feature = int(feature)
    sorted_label = sorted(label_dict.keys())
    sorted_feature = sorted(feature_dict.keys())

    num_labels = len(label_dict.keys())
    min_label = sorted_label[0]
    max_label = sorted_label[-1]

    min_feature = sorted_feature[0]
    max_feature = sorted_feature[-1]
    num_features = max_feature + 1
    print(num_points, "points")
    print(num_labels, "labels, min", min_label, "max", max_label)
    print(num_features, "features, min", min_feature, "max", max_feature)
    return num_points, num_features, num_labels

def save_to_tfrecords_sparse(filename, save_path):
    """
    save dataset to tfrecord, with format (index, value, label)

    parameters:
    filename - input dataset, in the format of .vw.gz
    save_path - the path you save tfrecords to
    """
    _float_feature = lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v))
    _int_feature = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    writer = tf.python_io.TFRecordWriter(save_path)
    num_points = 0
    with open(filename, 'r') as f:
        # lines = [x.decode('utf8').strip() for x in f.readlines()]
        for line in f:
            line = line.strip()
            data = line.split('|')
            # label starts at 1 in dataset
            label = int(data[0].strip().split(' ')[0]) - 1
            features_str = data[1].strip()
            features = features_str.split(' ')

            indices = []
            values = []

            for feature in features:
                index, value = feature.split(':')
                # feature starts at 1 in dataset
                indices.append(int(index))
                values.append(float(value))

            label_ = _int_feature([label])
            example = tf.train.Example(
                  features=tf.train.Features(
                  feature={
                      'label': label_,
                      'index': _int_feature(indices),
                      'value': _float_feature(values)
                  }))
            writer.write(example.SerializeToString())
            num_points += 1
            if num_points % 10000 == 0:
                print('%d lines done' % num_points)

def save_to_tfrecords_dense(filename, save_path):
    _float_feature = lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v))
    _int_feature = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    writer = tf.python_io.TFRecordWriter(save_path)
    num_points = 0
    num_features = 6144
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split('|')
            # label starts at 1 in dataset
            label = int(data[0].strip().split(' ')[0]) - 1
            features_str = data[1].strip().split(' ')
            # initialize a feature vector
            features = np.zeros(num_features)
            for feature in features_str:
                index, value = feature.split(':')
                features[int(index)] = float(value)

            label_ = _int_feature([label])
            features_ = _float_feature(features)
            example = tf.train.Example(
                  features=tf.train.Features(
                  feature={
                      'label': label_,
                      'features': features_
                  }))
            writer.write(example.SerializeToString())
            num_points += 1
            if num_points % 10000 == 0:
                print('%d lines done' % num_points)

def save_batches(P, output_path, chunck_size=10000):
    """
    P: probability matrix (R * num_test * R)
    output_path: save path
    chunck_size: how large is a batch

    returns: a list of paths representing batch binary files
    """
    size = chunck_size
    num_test = P[0].shape[0]
    R = P.shape[0]

    num_batches = num_test // size
    if num_test % size != 0:
        num_batches += 1

    paths = []

    for i in range(num_batches):
        x_batch = None
        if i != num_batches - 1:
            x_batch = P[:, i*size:(i+1)*size, :]
        else:
            x_batch = P[:, i*size:, :]
        print("x_shape", x_batch.shape)
        batch = []
        for r in range(R):
            batch.append(x_batch[r].flatten(order='F'))
        probs = np.concatenate(batch)
        print("probs_shape", probs.shape)

        output_path_prefix = output_path + "/prob_chunck_"
        suffix = ".dat"
        filename = output_path_prefix + str(i) + suffix
        paths.append(filename)

        with open(filename, 'w') as f:
            probs.astype(dtype=np.float32).tofile(f)
        print("done", i)

    return paths

def save_hash(a, b, output_path):
    """
    save hash paramaters to output path in binary format
    a: parameter a
    b: parameter b
    output_path: path to hash binary files

    returns: file paths for two binary files
    """
    a_path = output_path + "/hash_params_a"
    b_path = output_path + "/hash_params_b"
    with open(a_path, 'w') as f:
        a.tofile(f)
    with open(b_path, 'w') as f:
        b.tofile(f)
    return a_path, b_path

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

class Timer():
  def __init__(self):
    self.start_time = time.time()

  def elapsed(self):
    end_time = time.time()
    duration = end_time - self.start_time
    self.start_time = end_time
    return duration
