from __future__ import print_function
import pickle
import numpy as np

 
np.random.seed(42)


def prepare_data(seqs, labels):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences

    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples),dtype=np.int32)
    # x_mask = np.ones((maxlen, n_samples))
    for idx, s in enumerate(seqs):
        # x[maxlen-lengths[idx]:, idx] = s
        x[:lengths[idx], idx] = s

    x_mask =1.0 - (x == 0)
    # x,x_mask (step*bat),

    return x, x_mask, lengths,labels


def load_data(path_train,path_test,maxlen,valid_portion=0, sort_by_len=False):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here RSC2015)
    :type n_items: int
    :param n_items: The number of items.
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    path_train_data = path_train
    path_test_data = path_test

    f1 = open(path_train_data, 'rb')
    train_set = pickle.load(f1)
    f1.close()

    f2 = open(path_test_data, 'rb')
    test_set = pickle.load(f2)
    f2.close()

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
            else:
                new_train_set_x.append(x[:maxlen])
                # new_train_set_x.append(x[len(x)-maxlen:])
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

        new_test_set_x = []
        new_test_set_y = []
        for xx, yy in zip(test_set[0], test_set[1]):
            if len(xx) < maxlen:
                new_test_set_x.append(xx)
                new_test_set_y.append(yy)
            else:
                new_test_set_x.append(xx[:maxlen])
                # new_test_set_x.append(x[len(x)-maxlen:])
                new_test_set_y.append(yy)
        test_set = (new_test_set_x, new_test_set_y)
        del new_test_set_x, new_test_set_y

    test_set_x, test_set_y = test_set
    # split training set into validation set
    if valid_portion > 0 :
        train_set_x, train_set_y = train_set
        n_samples = len(train_set_x)
        sidx = np.arange(n_samples, dtype='int32')
        np.random.shuffle(sidx)
        n_train = int(np.round(n_samples * (1. - valid_portion)))
        valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
        valid_set_y = [train_set_y[s] for s in sidx[n_train :]]
        train_set_x = [train_set_x[s] for s in sidx[:n_train]]
        train_set_y = [train_set_y[s] for s in sidx[:n_train]]

        train_set = (train_set_x, train_set_y)
        valid_set = (valid_set_x, valid_set_y)
        valid_set_x, valid_set_y = valid_set
    #train data
    train_set_x, train_set_y = train_set

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

        if valid_portion > 0 :
            sorted_index = len_argsort(valid_set_x)
            valid_set_x = [valid_set_x[i] for i in sorted_index]
            valid_set_y = [valid_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    test = (test_set_x, test_set_y)
    if valid_portion > 0:
        valid = (valid_set_x, valid_set_y)
    else:
        valid = None

    return train, valid, test

def get_minibatches_idx(n, minibatch_size, shuffle=True):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def print_metric(test_evaluation):
    print('\nTest Recall@20', test_evaluation[0][0], '   Test Mrr@20:', test_evaluation[1][0],
          '\nTest Recall@10', test_evaluation[0][1], '   Test Mrr@10:', test_evaluation[1][1],
          '\nTest Recall@5', test_evaluation[0][2], '   Test Mrr@5:', test_evaluation[1][2],
          '\nTest Recall@3', test_evaluation[0][3], '   Test Mrr@3:', test_evaluation[1][3], '\n')
