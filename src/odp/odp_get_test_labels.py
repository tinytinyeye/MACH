from sparse_multiclf import *
import sys
import numpy as np

NUM_FEATURES = 422713
NUM_CLASSES = 105033
TRAIN_FILE = "./data/tfrecords/train.tfrecords"
TEST_FILE = "./data/tfrecords/test.tfrecords"

def main():
    argv = sys.argv
    if len(argv) < 3:
        print("usage: python3 main.py [B] [R] [gpu_option] [tag] [start] [end]")
        return
    B = int(argv[1])
    R = int(argv[2])
    gpu_option = '0'
    if len(argv) >= 4:
        gpu_option = argv[3]

    tag=None

    if len(argv) >= 5:
        tag = argv[4]

    start = 0
    end = R
    if len(argv)>= 7:
        start = int(argv[5])
        end = int(argv[6])

    clf = MultiClassifier(R=R,
                          B=B,
                          num_features=NUM_FEATURES,
                          num_classes=NUM_CLASSES,
                          seed=0,
                          tag=tag,
                          save_path=None,
                          load_hash=False
                          )
    # clf.train(TRAIN_FILE, gpu_option=gpu_option, start=start, end=end)
    # clf.evaluate(TEST_FILE, gpu_option=gpu_option, load_probs=True)
    label_array = clf.get_test_labels(TEST_FILE)
    print("labels get")
    np.save("./y_test.npy", label_array)
    print("save done")
    sys.exit()

if __name__ == '__main__':
  main()
