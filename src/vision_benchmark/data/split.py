"""Fixed, seeded split of image ids into test / val / train / unused."""

import random


def split_ids(ids, n_test, n_val, n_train=None, seed=42):
    """Shuffle ids with a fixed seed, then take n_test for test, n_val for val, n_train for train.

    n_train=None puts every remaining id in train; otherwise the rest is "unused".
    Returns a dict {id: split}.
    """
    ids = sorted(ids)  # same order on every machine before shuffling
    random.Random(seed).shuffle(ids)
    out = {}
    for i, image_id in enumerate(ids):
        if i < n_test:
            out[image_id] = "test"
        elif i < n_test + n_val:
            out[image_id] = "val"
        elif n_train is None or i < n_test + n_val + n_train:
            out[image_id] = "train"
        else:
            out[image_id] = "unused"
    return out
