from sklearn import svm
import numpy as np

CACHE_SIZE = 1000

road_classifier = None


def debug():
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


def train():
    pass


def test():
    pass


def predict():
    pass


def classifier_init():
    global road_classifier
    road_classifier = svm.SVC(CACHE_SIZE)


if __name__ == '__main__':
    classifier_init()
    debug()
    # train()
    # test()
    # predict()
