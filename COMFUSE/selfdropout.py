import numpy as np
import random
import torch


# array: the input array to drop out, in torch tensor form
# drop_rate: if drop rate is 0.3, 30% words will be set 0
# type=1: drop out selected words with all dims to 0
# type=2: drop out selected dims of all words to 0
# type=3: drop out selected dims of selected words to 0
def rnddrop_2(inputarray, drop_rate, drtype, verbose=False):
    # 为了不改变输入矩阵的值
    array = inputarray.clone()
    assert array.dim() == 5
    num_of_words = array.size(-2)
    num_of_dims = array.size(-1)
    if verbose:
        print("#drtype: %d, dropout rate: %f, num of words: %d, num of dims: %d" % (
            drtype, drop_rate, num_of_words, num_of_dims))

    # drop out selected words
    if drtype == 1:
        num_of_drops = round(num_of_words * drop_rate)
        if verbose:
            print("num of drops: %d" % (num_of_drops))
        for i in range(array.size(0)):
            for j in range(array.size(1)):
                rndIdxes = random.sample(range(0, num_of_words), num_of_drops)
                for rndIdx in rndIdxes:
                    if verbose:
                        print("drop index: %d" % rndIdx)
                    array[i, j, :, rndIdx, :] = 0

    # drop out selected dims
    elif drtype == 2:
        num_of_drops = round(num_of_dims * drop_rate)
        if verbose:
            print("num of drops: %d" % (num_of_drops))
        for i in range(array.size(0)):
            for j in range(array.size(1)):
                rndIdxes = random.sample(range(0, num_of_dims), num_of_drops)
                for rndIdx in rndIdxes:
                    if verbose:
                        print("drop index: %d" % rndIdx)
                    array[i, j, :, :, rndIdx] = 0

    # drop out selected dims of selected words
    elif drtype == 3:
        num_of_worddrops = round(num_of_words * drop_rate)
        num_of_dimdrops = round(num_of_dims * drop_rate)
        if verbose:
            print("num of word drops: %d" % (num_of_worddrops))
            print("num of dim drops: %d" % (num_of_dimdrops))

        for i in range(array.size(0)):
            for j in range(array.size(1)):
                rndIdxes = random.sample(range(0, num_of_words), num_of_worddrops)
                for rndIdx in rndIdxes:
                    if verbose:
                        print("drop word index: %d" % rndIdx)
                    rndDimIdxes = random.sample(range(0, num_of_dims), num_of_dimdrops)
                    for rndDimIdx in rndDimIdxes:
                        if verbose:
                            print("drop dim index: %d" % rndIdx)
                        array[i, j, :, rndIdx, rndDimIdx] = 0
    return array






def rnddrop(inputarray, drop_rate, drtype, verbose=False):
    # 为了不改变输入矩阵的值
    array = inputarray.clone()
    assert array.dim() == 5
    num_of_words = array.size(-2)
    num_of_dims = array.size(-1)
    if verbose:
        print("#drtype: %d, dropout rate: %f, num of words: %d, num of dims: %d" % (
            drtype, drop_rate, num_of_words, num_of_dims))

    # drop out selected words
    if drtype == 1:
        num_of_drops = round(num_of_words * drop_rate)
        if verbose:
            print("num of drops: %d" % (num_of_drops))
        rndIdxes = random.sample(range(0, num_of_words), num_of_drops)
        for rndIdx in rndIdxes:
            if verbose:
                print("drop index: %d" % rndIdx)
            array[:, :, :, rndIdx, :] = 0

    # drop out selected dims
    elif drtype == 2:
        num_of_drops = round(num_of_dims * drop_rate)
        if verbose:
            print("num of drops: %d" % (num_of_drops))
        rndIdxes = random.sample(range(0, num_of_dims), num_of_drops)
        for rndIdx in rndIdxes:
            if verbose:
                print("drop index: %d" % rndIdx)
            array[:, :, :, :, rndIdx] = 0

    # drop out selected dims of selected words
    elif drtype == 3:
        num_of_worddrops = round(num_of_words * drop_rate)
        num_of_dimdrops = round(num_of_dims * drop_rate)
        if verbose:
            print("num of word drops: %d" % (num_of_worddrops))
            print("num of dim drops: %d" % (num_of_dimdrops))
        rndIdxes = random.sample(range(0, num_of_words), num_of_worddrops)

        for rndIdx in rndIdxes:
            if verbose:
                print("drop word index: %d" % rndIdx)
            rndDimIdxes = random.sample(range(0, num_of_dims), num_of_dimdrops)
            for rndDimIdx in rndDimIdxes:
                if verbose:
                    print("drop dim index: %d" % rndIdx)
                array[:, :, :, rndIdx, rndDimIdx] = 0
    return array


# array: the input array to drop out, in numpy form
def rnddrop_numpy(inputarray, drop_rate, drtype):
    # 为了不改变输入矩阵的值
    array = inputarray.copy()
    num_of_words = array.shape[-2]
    num_of_dims = array.shape[-1]
    print("#drtype: %d, dropout rate: %f, num of words: %d, num of dims: %d" % (
        drtype, drop_rate, num_of_words, num_of_dims))

    # drop out selected words
    if drtype == 1:
        num_of_drops = round(num_of_words * drop_rate)
        print("num of drops: %d" % (num_of_drops))
        rndIdxes = random.sample(range(0, num_of_words), num_of_drops)
        for rndIdx in rndIdxes:
            print("drop index: %d" % rndIdx)
            array[:, :, rndIdx, :] = 0

    # drop out selected dims
    elif drtype == 2:
        num_of_drops = round(num_of_dims * drop_rate)
        print("num of drops: %d" % (num_of_drops))
        rndIdxes = random.sample(range(0, num_of_dims), num_of_drops)
        for rndIdx in rndIdxes:
            print("drop index: %d" % rndIdx)
            array[:, :, :, rndIdx] = 0

    # drop out selected dims of selected words
    elif drtype == 3:
        num_of_worddrops = round(num_of_words * drop_rate)
        num_of_dimdrops = round(num_of_dims * drop_rate)
        print("num of word drops: %d" % (num_of_worddrops))
        print("num of dim drops: %d" % (num_of_dimdrops))
        rndIdxes = random.sample(range(0, num_of_words), num_of_worddrops)

        for rndIdx in rndIdxes:
            print("drop word index: %d" % rndIdx)
            rndDimIdxes = random.sample(range(0, num_of_dims), num_of_dimdrops)
            for rndDimIdx in rndDimIdxes:
                print("drop dim index: %d" % rndIdx)
                array[:, :, rndIdx, rndDimIdx] = 0
    return array

# array = np.ones( (1,1,6,5) )
# print(array)
# array = rnddrop(array, 0.3, 2)
# print(array)
