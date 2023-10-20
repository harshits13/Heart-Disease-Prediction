import math


def train_test_spilt(spilt_factor, x, y):
    n_train = math.floor(spilt_factor * x.shape[0])
    X_train = x[:n_train]
    X_test = x[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test


def standardization(data):
    standardized_data = (data - data.mean()) / data.std()
    return standardized_data
