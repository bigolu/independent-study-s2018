"""
An SVC for road types.
"""
from typing import List, Tuple
import sys

from sklearn import svm
import numpy as np


CACHE_SIZE = 1000


def debug() -> None:
    """
    for testing
    """
    training_features = np.ascontiguousarray([[0, 0], [1, 1]],
                                             dtype=np.float32)
    training_labels = np.ascontiguousarray([0, 1], dtype=np.float32)

    # feature scale features

    assert training_labels.flags['C_CONTIGUOUS'], \
        ("training labels aren't C_CONTIGUOUS")
    assert training_features.flags['C_CONTIGUOUS'], \
        ("training features aren't C_CONTIGUOUS")

    road_classifier.fit(training_features, training_labels)

    print(road_classifier.predict([[2., 2.]]))

def load_data(filenames: List[str]) -> List[Tuple[str]]:
    """
    Load data from csv file(s). Returns a list of tuples where each tuple
    represents a record. The structure of tuple is as follows:
    0:
    1:
    2:
    3:

    Args:
    files - a list of filenames for CSVs containing data
    """
    records = []
    return records

def make_feature_vectors() -> List[List[float]]:
    pass

def train():
    feature_vectors = []
    labels = []

def test():
    pass

def main():
    """
    Classify roads.
    """
    raw_data = load_data(sys.argv[1:])
    feature_vectors  = make_feature_vectors(raw_data)
    labels = make_labels(raw_data)
    road_classifier = svm.SVC(CACHE_SIZE)
    train()
    # test()

if __name__ == '__main__':
    main()
    # debug()

