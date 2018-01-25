from dense_multiclf import *
import sys

NUM_FEATURES = 6144
NUM_CLASSES = 21841
TRAIN_FILE = "./data/tfrecords/train.tfrecords"
TEST_FILE = "./data/tfrecords/test.tfrecords"

def main():
    parser = argparse.ArgumentParser(description='Predict Imagenet dataset')
    parser.add_argument('-b', '--bucket', type=int, required=True, help='Number of buckets for hashing')
    parser.add_argument('-r', '--repetition', type=int, required=True, help='Number of meta classifiers')
    parser.add_argument('-g', '--gpu', default='0', type=str, help='gpu id (0 or 1)')
    parser.add_argument('-t', '--tag', default=None, type=str, help='tag to continue training')
    parser.add_argument('--load_probs', action='store_true')
    parser.add_argument('--start', default=0, type=int, help='training start location')
    parser.add_argument('--end', default=0, type=int, help='training end location')

    params = P()
    args = parser.parse_args(namespace=params)
    
    B = params.bucket
    R = params.repetition
    gpu_option = params.gpu
    tag = params.tag
    start = params.start
    end = params.end
    load_probs = params.load_probs

    if end == 0:
        end = R

    clf = MultiClassifier(R=R,
                          B=B,
                          num_features=NUM_FEATURES,
                          num_classes=NUM_CLASSES,
                          seed=0,
                          tag=tag,
                          save_path=None,
                          load_hash=False
                          )
    clf.train("./data/tfrecords/training.tfrecords", gpu_option=gpu_option,
                                                        start=start, end=end)
    clf.evaluate("./data/tfrecords/testing.tfrecords", gpu_option=gpu_option)
    sys.exit()

if __name__ == '__main__':
  main()
