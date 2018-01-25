import numpy as np
import tensorflow as tf

def probs2byte(input_path, output_path, limit=1000):
    probs = np.load(input_path)['probs'][limit, :]
    with open(output_path, 'w') as f:
        probs.astype(dtype=np.float32).flatten(order='F').tofile(f)
    print("done")

def hashparams2byte(input_path, output_path):
    hash_params = np.load(input_path)
    a = hash_params['a']
    b = hash_params['b']
    a_path = output_path + "_a"
    b_path = output_path + "_b"
    with open(a_path, 'w') as f:
        a.tofile(f)
    with open(b_path, 'w') as f:
        b.tofile(f)

def label2byte(input_path, output_path, limit=1000):
    labels = np.load(input_path)[limit, :]
    with open(output_path, 'w') as f:
        f.tofile(f)

# def test():
#     probs = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
#     input_path = "./test.npz"
#     np.savez(input_path, probs=probs)
#     output_path = "./test_byte"
#     probs2byte(input_path, output_path)
#     arr = np.fromfile(output_path).astype(dtype=np.float32)
#     print arr

# test()
def save_hash(output_path, tag, B, R):
    root_dir = output_path + "/" + tag + "_B" + str(B) + "_R" + str(R)
    output_dir = "./bin/"
    hash_dir = root_dir + "/hash_" + tag + "_B" + str(B) + "_R" + str(R) + ".npz"
    hashparams2byte(hash_dir, output_dir + "hash_params")
    print("hash parameters done")

def make_batches(P, output_path, chunck_size=10000):
    """
    create batches for (R * num_test * R)
    """
    size = chunck_size
    num_test = P[0].shape[0]
    num_batches = num_test // size
    if num_test % size != 0:
        num_batches += 1
    print("num_batches", num_batches)
    R = 50
    for i in range(num_batches):
        x_batch = None
        if i != num_batches - 1:
            x_batch = P[:, i*size:(i+1)*size, :]
        else:
            x_batch = P[:, i*size:, :]
        batch = []
        for r in range(R):
            batch.append(x_batch[r].flatten(order='F'))
        probs = np.concatenate(batch)
        output_path_prefix = "./prob_chunck_"
        suffix = ".dat"
        filename = output_path_prefix + str(i) + suffix
        with open(filename, 'w') as f:
            probs.astype(dtype=np.float32).tofile(f)
        print("done", i)

def byte2numpy(input_path):
    arr = np.fromfile(input_path, dtype=np.float32)
    length = arr.shape
    print(length)
    # assert(105033000 == length)
    k = 105033
    n = 1000
    reshape = arr.reshape((k, n))
    # arr = arr.reshape((k, n)).T
    #
    # print(arr.shape)
    # print(arr[0])

    with tf.Graph().as_default():
        # reshape = tf.reshape(arr, [k, n])
        global_prob = tf.placeholder(dtype=tf.float32, shape=[k, None])
        argmax = tf.argmax(global_prob, axis=0)

        sess = tf.Session()
        result = sess.run(argmax, feed_dict={global_prob: reshape})
        print(result)

# create_test_set()
# input_path = "./cpp/global_prob.dat"
# byte2numpy(input_path)

# save_hash("204B9E", 32, 50)
