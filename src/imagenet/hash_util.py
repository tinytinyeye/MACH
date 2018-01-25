import random
import numpy as np

PRIME = 105943

def H(label, a, b, B, p=PRIME):
    # label is an int and B number of partitions
    return ((a * label + b) % p) % B

def hash_factory(a, b, B, p=PRIME):
    """
    generate a hash function based on parameters a, b, B and p
    """
    return lambda label : ((a * label + b) % p) % B

def hash_vector(y, a, b, B, p):
    mapper = lambda x: ((a * x + b) % p) % B
    vfunc = np.vectorize(mapper)
    y_h = vfunc(y)
    return y_h

class HashGenerator(object):
    def __init__(self, seed=0, p=PRIME):
        random.seed(seed)
        self.p = p

    def generate_ab(self, R):
        a = []
        b = []
        for i in range(R):
            tmp = random.randrange(1, self.p)
            while tmp in a:
                tmp = random.randrange(1, self.p)
            a.append(tmp)
            tmp = random.randrange(1, self.p)
            while tmp in b:
                tmp = random.randrange(0, self.p)
            b.append(tmp)

        return a, b

    def get_a(self):
        return random.randrange(1, self.p)

    def get_b(self):
        return random.randrange(0, self.p)

    def get_p(self):
        return self.p

    def get_hash_params(self, num_labels, R, B):
        """
        generate R pairs of hash parameters a and b and ensure that the hashed
        result will have B different kinds.

        params:
        num_labels - total number of classes
        R - paris of hash parameters
        B - number of buckets
        p - a large prime number

        returns:
        a, b - hash parameters
        """
        print("in get hash params")
        a, b = self.generate_ab(R)
        print(a)
        print(b)
        y = np.array(list(range(num_labels)))
        print(len(y))
        for i in range(R):
            y_hash = hash_vector(y, a[i], b[i], B, self.p)
            print("in model", i)
            print("y_hash unique is", len(np.unique(y_hash)))
            print("B is", B)
            # hash function may lead to less than B categories
            while len(np.unique(y_hash)) != B:
                print("hash failed, rehashing")
                tmp_a = self.get_a()
                tmp_b = self.get_b()
                while tmp_a in a:
                    tmp_a = self.get_a()
                while tmp_b in b:
                    tmp_b = self.get_b()
                a[i] = tmp_a
                b[i] = tmp_b
                y_hash = hash_vector(y, a[i], b[i], B, self.p)
            # print("in model %d the number of unique labels after hash is %d" % (i, len(np.unique(y_hash))))
        return a, b
