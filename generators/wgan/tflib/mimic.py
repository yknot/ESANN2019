import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def mimic_generator(data, batch_size):
    def get_epoch():
        np.random.shuffle(data)

        # trim to size divisible by batch_size
        data_batches = data[:(data.shape[0] // batch_size) * batch_size]
        data_batches = data_batches.reshape(-1, batch_size, data.shape[1])

        for i in range(len(data_batches)):
            yield np.copy(data_batches[i])

    return get_epoch


def load(df, batch_size, test_batch_size):
    # split 70-30
    train_data, big_test = train_test_split(
        df, test_size=0.3, random_state=100)
    # split 30 again in half
    dev_data, test_data = train_test_split(
        big_test, test_size=.5, random_state=100)

    return (
        mimic_generator(train_data, batch_size),
        mimic_generator(dev_data, test_batch_size),
        mimic_generator(test_data, test_batch_size)
    )
