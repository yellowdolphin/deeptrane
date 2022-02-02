import numpy as np


def get_class_ids(s):
    "Return list of class_ids from prediction_string `s`"
    a = np.array(s.split()).reshape(-1, 6)
    return set(a[:, 0].tolist())


def get_single_label_class_ids(s):
    s = get_class_ids(s)
    assert len(s) == 1
    return s.pop()
