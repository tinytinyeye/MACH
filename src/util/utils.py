import json
import os
import numpy as np

def load_config(filename):
    """
    load config json file and read parameters

    parameters:
    filename: config json filename

    returns:
    train_file: training data filename
    test_file: testing data filename
    num_classes: number of labels in training file
    num_features: number of features in testing file
    """
    with open(filename) as f:
        config = json.load(f)
        train_file = config['train_file']
        test_file = config['test_file']
        num_classes = config['num_classes']
        num_features = config['num_features']
        return train_file, test_file, num_classes, num_features

def load_log(filename):
    with open(filename) as f:
        train_times = []
        predict_times = []
        evaluate_time = None
        accuracy = None
        for line in f:
            if len(line.split()) > 1:
                (tag, b, r, time) = line.split()
                if tag == "TRAIN":
                    train_times.append(float(time.split("=")[1]))
                if tag == "PREDICT":
                    predict_times.append(float(time.split("=")[1]))
                if tag == "EVALUATE":
                    evaluate_time = float(time.split("=")[1])
            else:
                accuracy = float(line.split("=")[1])
        max_train_time = np.array(train_times).max()
        average_train_time = np.array(train_times).mean()
        max_predict_time = np.array(predict_times).max()
        average_predict_time = np.array(predict_times).mean()
        print("max train time:", max_train_time)
        print("average train time:", average_train_time)
        print("max predict time:", max_predict_time)
        print("average predict time:", average_predict_time)
        print("evaluation time:", evaluate_time)
        print("accuracy:", accuracy)
