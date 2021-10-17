import numpy as np


def transform_cm_for_lda(data):
    return np.absolute(data)
    # data = data.reshape([data.shape[0], 64, 3])
    # print(data.shape)
    # for x in range(len(data)):
    #     d = data[x]
    #     m = np.amin(d, axis=0)
    #     for y in d:
    #         y[2] = y[2] - m[2]
    # return data.reshape(data.shape[0], 192)
