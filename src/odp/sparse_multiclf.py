from concurrent.futures import ProcessPoolExecutor, wait
import sys, os, errno
import time
import tensorflow as tf
import numpy as np
import hash_util
import util

################## constants #########################
BATCH_SIZE = 64
NUM_EPOCHES = 5
NUM_PREPROCESS_THREADS = 12
LEARNING_RATE = 0.01
MIN_AFTER_DEQUEUE = 10000
THR = 0.01
STOP_THR = 12000
################## helper functions ##################
def matmul(X, W):
    """
    general purpose matrix multiplication
    params:
    X - dense tensor or sparse tensor in the form of (indices, values)
    W - weights
    """
    if type(X) == tf.Tensor:
        return tf.matmul(X, W)
    else:
        return tf.nn.embedding_lookup_sparse(W, X[0], X[1], combiner="sum")

def init_weights(shape, stddev=0.01, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def init_bias(shape, val=0.0, name=None):
    init = tf.constant(val, shape=shape)
    return tf.Variable(init, name=name)

def decode(batch_serialized_examples):
    features = tf.parse_example(
        batch_serialized_examples,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'index': tf.VarLenFeature(tf.int64),
            'value': tf.VarLenFeature(tf.float32),
        }
    )
    labels = features['label']
    indices = features['index']
    values = features['value']

    return labels, indices, values

################## classifier class ##################
class MultiClassifier(object):
    def __init__(self,
                    R,
                    B,
                    num_features,
                    num_classes,
                    seed=0,
                    tag=None,
                    save_path=None,
                    load_hash=False):
        """
        Parameters:

        R - number of sub-classifiers
        B - number of merged classes in each sub-classifier
        num_features - number of features in the input dataset
        num_classes - number of classes in the input dataset
        tag - unique tag to identify the directory that contains models, if not
                specified, a new tag will be generated and a new directory
                named with this tag and parameters R, B will be created.
        save_path - path to save the directory that contains models, hash
                        parameters and precomputed probabilities. If tag is
                        provided, the program will try to load the data from
                        the directory specified by tag.

        """
        self.R = R;
        self.B = B;
        self.num_features = num_features
        self.num_classes = num_classes
        self.save_path = save_path
        self.seed = seed
        self.tag = tag
        self.root_dir = None
        self.model_dir = None
        self.probs_dir = None
        self.hash_path = None
        self.complete_probs_path = None
        # hash parameters
        self.a = None
        self.b = None
        self.p = hash_util.PRIME
        # training variables
        self.weights = []
        self.bias = []

        if self.tag is None:
            # create a 6 characters id to save models and hash parameters
            self.tag = util.id_generator()

        if save_path is None:
            self.save_path = "./"
        elif not save_path.endswith("/"):
            self.save_path = save_path + "/"

        self.root_dir = self.save_path + self.tag + \
                                "_B" + str(self.B) + \
                                "_R" + str(self.R)
        self.model_dir = self.root_dir + "/models"
        self.probs_dir = self.root_dir + "/probs"
        self.hash_path = self.root_dir + "/hash_" + self.tag + \
                                "_B" + str(self.B) + "_R" + str(self.R) + ".npz"

        # create new directory if the root directory does not exist
        if not os.path.isdir(self.root_dir):
            try:
                os.makedirs(self.root_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        if not os.path.isdir(self.model_dir):
            try:
                os.makedirs(self.model_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        if not os.path.isdir(self.probs_dir):
            try:
                os.makedirs(self.probs_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        h = hash_util.HashGenerator(seed=self.seed)
        h_a, h_b = h.get_hash_params(self.num_classes, self.R, self.B)
        self.a = h_a
        self.b = h_b
        self.p = h.get_p()

        if load_hash is True:
            print("loading hash parameters")
            hash_params = np.load(self.hash_path)
            self.a = hash_params['a']
            self.b = hash_params['b']
        else:
            np.savez(self.hash_path, a=h_a, b=h_b)

        self.complete_probs_path = self.probs_dir + "/complete_probs_" + \
                                   self.tag + \
                                   "_B" + str(self.B) + \
                                   "_R" + str(self.R) + ".npz"
        print("current tag is", self.tag)

    def get_model_path(self, model_id):
        return self.model_dir + "/model_" + self.tag + \
                                    "_B" + str(self.B) + \
                                    "_R" + str(self.R) + \
                                    "_" + str(model_id) + ".npz"

    def get_probs_path(self, probs_id):
        return self.probs_dir + "/probs_" + self.tag + \
                                    "_B" + str(self.B) + \
                                    "_R" + str(self.R) + \
                                    "_" + str(probs_id) + ".npz"

    def get_tag(self):
        return self.tag

    def load_sparse_shuffle(self, filename, graph,
                            batch_size=BATCH_SIZE,
                            num_epochs=NUM_EPOCHES):
        """
        load sparse matrix data from tfrecord file and save to X and y
        used in loading training data
        """
        with graph.as_default():
            filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=num_epochs)
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            batch_serialized_examples = tf.train.shuffle_batch(
                [serialized_example],
                batch_size=batch_size,
                num_threads=NUM_PREPROCESS_THREADS,
                capacity=MIN_AFTER_DEQUEUE+(NUM_PREPROCESS_THREADS+1)*batch_size,
                min_after_dequeue=MIN_AFTER_DEQUEUE
            )

            return decode(batch_serialized_examples)

    def load_sparse(self, filename, graph, batch_size=100):
        """
        load testing data
        """
        with graph.as_default():
            filename_queue = tf.train.string_input_producer([filename],
                                                            num_epochs=1)
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            batch_serialized_examples = tf.train.batch(
                [serialized_example],
                batch_size=batch_size,
                num_threads=1,
                capacity=batch_size+(4+1)*batch_size,
            )

            return decode(batch_serialized_examples)

    def train_single(self, filename, clf_id, num_epochs=5, gpu_option='0'):
        """
        filename - input file, in the format of TFRecords
        clf_id - classifier id, used to identify which hash parameter to use
        num_epochs - number of epochs running for each classifier
        """
        # return value
        ret_weight = None
        ret_bias = None
        graph = tf.Graph()
        with graph.as_default():
            # build input pipeline
            labels, indices, values = self.load_sparse_shuffle(filename,
                                                        graph,
                                                        num_epochs=num_epochs)
            X = (indices, values)
            y = labels
            # hash labels to the parameters for given classifier
            y_h = tf.map_fn(hash_util.hash_factory(self.a[clf_id],
                                                   self.b[clf_id],
                                                   self.B), y)
            W = init_weights([self.num_features, self.B], name="weights")
            b = init_bias([self.B], name="bias")

            # build graph
            y_p = matmul(X, W) + b
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=y_p,
                                    labels=y_h)
                                    )
            prediction = tf.nn.in_top_k(y_p, y_h, 1)
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
            train_op = tf.train.AdamOptimizer().minimize(loss)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            sess = None
            # only use 1 gpu, 0 or 1
            if gpu_option == '1' or gpu_option == '0':
                config = tf.ConfigProto()
                config.gpu_options.visible_device_list = gpu_option
                sess = tf.Session(config=config)
            else:
                sess = tf.Session()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            path = "./" + "_".join((self.tag, str(self.B), str(self.R), "time")) + ".out"
            log = open(path, 'a')
            train_timer = util.Timer()
            try:
              step = 0
              timer = util.Timer()
              min_loss = float("inf")
              stabilized_steps = 0
              while not coord.should_stop():
                  _, loss_, accuracy_ = sess.run([train_op, loss, accuracy])
                  if min_loss - loss_ > THR:
                      stabilized_steps = 0
                      min_loss = loss_
                  else:
                      stabilized_steps += 1
                  # manually set early stop value
                #   if stabilized_steps > STOP_THR and abs(loss_ - min_loss) < 10 * THR:
                #       break
                  if step % 100 == 0:
                      print('step:', step,
                            'train precision@1:', accuracy_,
                            'loss:', loss_,
                            'min_loss', min_loss,
                            'duration:', timer.elapsed())
                  step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' %
                        (num_epochs, step))
            finally:
                coord.request_stop()

            log.write("TRAIN B={} R={} time={}\n".format(self.B, self.R,
                                                    train_timer.elapsed()))
            log.close()
            ret_weight = sess.run(W)
            ret_bias = sess.run(b)
            coord.join(threads)
            sess.close()

            return ret_weight, ret_bias

    def train(self, filename, num_epochs=5, start=0, end=None, gpu_option='0'):
        """
        filename - input file, in the format of TFRecords
        num_epochs - number of epochs running for each classifier
        resume_from - resume training from a specific classifier
        """
        if end is None:
            end = self.R

        for i in range(start, end):
            path = self.get_model_path(i)
            weight, bias = self.train_single(filename, i, num_epochs=num_epochs,
                                                            gpu_option=gpu_option)
            np.savez_compressed(path, weight=weight, bias=bias)

    def predict_single(self, filename, clf_id, gpu_option='0'):
        graph = tf.Graph()
        probs = []
        path = "./" + "_".join((self.tag, str(self.B), str(self.R), "time")) + ".out"
        log = open(path, 'a')
        predict_timer = util.Timer()
        with graph.as_default():
            # build input pipeline
            labels, indices, values = self.load_sparse(filename, graph)
            X = (indices, values)
            y = labels
            W = tf.constant(self.weights[clf_id])
            b = tf.constant(self.bias[clf_id])
            # build graph
            y_p = tf.nn.softmax(matmul(X, W) + b)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess = None
            if gpu_option == '1' or gpu_option == '0':
                print("single gpu")
                config = tf.ConfigProto()
                config.gpu_options.visible_device_list = gpu_option
                sess = tf.Session(config=config)
            else:
                sess = tf.Session()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
              timer = util.Timer()
              while not coord.should_stop():
                probs.append(sess.run(y_p))
            except tf.errors.OutOfRangeError:
                print("Done predicting")
            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close()
            log.write("PREDICT B={} R={} time={}\n".format(self.B, self.R,
                                                    predict_timer.elapsed()))
            log.close()
            return np.concatenate(probs, axis=0)

    def predict(self, filename, gpu_option='0', start=0, end=None):
        """
        get the probability matrix for each sub-classifier
        """
        prob_list = []
        if end is None:
            end = self.R
        # load all models from disk to memory
        if (len(self.weights) == 0 or len(self.bias) == 0):
            print("start loading weights and bias")
            for i in range(self.R):
                path = self.get_model_path(i)
                params = np.load(path)
                w = params['weight']
                b = params['bias']
                self.weights.append(w)
                self.bias.append(b)
                print("finished loading model", i)
            print("finished loading weights and bias")
        for i in range(start, end):
            probs = self.predict_single(filename, i, gpu_option=gpu_option)
            path = self.get_probs_path(i)
            np.savez_compressed(path, probs=probs)
            prob_list.append(probs)
            print("finished predicting model", i)

        print("merging probabilities from all sub-classifiers")
        # garbage collect
        self.weights = None
        self.bias = None

    def get_complete_probs(self):
        prob_list = []
        for i in range(self.R):
            path = self.get_probs_path(i)
            probs = np.load(path)['probs']
            prob_list.append(probs)
        print("merging probabilities from all sub-classifiers")
        # P = np.dstack(prob_list)
        # merge all probabilities to one giant matrix
        # P = np.rollaxis(P, -1) # P is now (R * num_test * B)
        P = np.stack(prob_list)
        return P

    def get_test_labels(self, filename, gpu_option='0'):
        label_list = []
        label_array = None
        graph = tf.Graph()
        with graph.as_default():
            labels, indices, values = self.load_sparse(filename, graph)
            init_op = tf.local_variables_initializer()
            sess = None
            if gpu_option == '1' or gpu_option == '0':
                print("single gpu")
                config = tf.ConfigProto()
                config.gpu_options.visible_device_list = gpu_option
                sess = tf.Session(config=config)
            else:
                sess = tf.Session()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
              step = 0
              while not coord.should_stop():
                  step += 1
                  labels_ = sess.run(labels)
                  label_list.append(labels_)
            except tf.errors.OutOfRangeError:
                print("done")
            finally:
                coord.request_stop()
            label_array = np.concatenate(label_list, axis=0)
            # np.save(save_path, label_array)
            coord.join(threads)
            sess.close()
            return label_array

    def evaluate_chunck(self, idx, P, y_test):
        """
        idx - index of a chunck
        P - chunck of probabilities to evaluate
        y_test - correct labels chunck
        """
        k = self.num_classes
        size = y_test.shape[0]
        GP = np.zeros((size, k))
        for i in range(k):
            probs_i = np.zeros((size, self.R))
            for j in range(self.R):
                probs_i[:, j] = P[j][:, hash_util.H(i, self.a[j], self.b[j], self.B, self.p)]
            GP[:, i] = np.mean(probs_i, axis=1)
            if (i % 10000 == 0):
                print("chunck {} finished class {}".format(idx, i))
        y_pred_chunck = np.argmax(GP, axis=1)
        y_test_chunck = y_test
        correct = np.sum(y_pred_chunck == y_test_chunck)
        accuracy = correct / size
        print("chunck {} done, accuracy is {}".format(idx, accuracy))
        return correct

    def evaluate(self, filename, chunck_size=10000, gpu_option='0', load_probs=False):
        y_test = self.get_test_labels(filename, gpu_option=gpu_option)
        # P = np.load(self.complete_probs_path)['probs']
        P = None
        if load_probs is False:
            self.predict(filename, gpu_option=gpu_option)
        path = "./" + "_".join((self.tag, str(self.B), str(self.R), "time")) + ".out"
        log = open(path, 'a')
        eval_timer = util.Timer()
        # Get complete probs
        P = self.get_complete_probs() # R * num_test * B

        k = self.num_classes
        num_test = P[0].shape[0]
        print("num_test", num_test)
        size = chunck_size
        print("start merging")

        pool = ProcessPoolExecutor()
        futures = []

        for i in range(num_test // size):
            x_batch = P[:, i*size:(i+1)*size, :]
            y_batch = y_test[i*size:(i+1)*size]
            futures.append(pool.submit(self.evaluate_chunck, i, x_batch, y_batch))

        if (num_test // size) * size < num_test:
            i = num_test // size
            x_batch = P[:, i*size:, :]
            y_batch = y_test[i*size:]
            futures.append(pool.submit(self.evaluate_chunck, i, x_batch, y_batch))

        wait(futures)

        correct = 0
        for fut in futures:
            correct += fut.result()

        print("total accuracy is", str(correct / num_test))

        log.write("EVALUATE B={} R={} time={}\n".format(self.B, self.R,
                                                eval_timer.elapsed()))
        log.write("ACCURACY={}\n".format(correct / num_test))
        log.close()
