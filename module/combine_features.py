import numpy as np

def combine_features(image_features, tabular_features):
    return np.concatenate((image_features, tabular_features), axis=1)

