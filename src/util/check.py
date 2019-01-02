
from util import py as pu
import numbers

def check_y(true_targets, predictions):
    if len(true_targets) != len(predictions):
        raise ValueError("True targets do not equal length of self.predictions, True targets:", len(true_targets),
                         "self.predictions", len(predictions))
    if (pu.isList(predictions[0]) and not pu.isList(true_targets[0])) or (
        not pu.isList(predictions[0]) and pu.isList(true_targets[0])):
        raise ValueError("One of the targets/self.predictions are a 2d array, when both of them are expected to be.")
    if pu.isList(predictions[0]):
        if len(predictions[0]) != len(true_targets[0]):
            raise ValueError("Lengths of internal arrays do not match, self.true_targets", len(true_targets[0]),
                             len(predictions[0]))
def check_x(features):
    if len(features[0]) <= 1:
        raise ValueError("X input is not 2d array, example", features[0])
    if isinstance(features[0][0], numbers.Integral):
        raise TypeError("X input is int not float, example", features[0])
    if len(features) < len(features[0]):
        raise ValueError("Very low X sample size compared to dimension size", len(features), "dimension size", len(features[0]))

def check_splits(x_train=None, y_train=None, x_test=None, y_test=None):
    if len(x_train)!= len(y_train):
        if len(y_train[0]) == len(x_train):
            print("Transposed y_train to fit x_train")
            y_train = y_train.transpose()
            check_splits(x_train, y_train, x_test, y_test)
        raise ValueError("Sizes do not match for x_train", len(x_train), "and y_train", len(y_train))
    if not pu.isList(x_train[0]):
        raise ValueError("X input is not 2d array, example", x_train[0])
    if isinstance(x_train[0][0], numbers.Integral):
        raise TypeError("X input is int not float, example", x_train[0])
    if isinstance(y_train[0][0], float):
        raise TypeError("Y input is float not int", y_train[0])
    if isinstance(y_train[0][0], str) or isinstance(x_train[0][0], str):
        raise TypeError("Strings found", y_train[0][0], x_train[0][0])
    if len(x_train) < len(x_train[0]):
        raise ValueError("Very low X sample size compared to dimension size", len(x_train), "dimension size", len(x_train[0]))
    if len(y_train) < len(y_train[0]):
        raise ValueError("Very low Y sample size compared to dimension size", len(y_train), "dimension size", len(y_train[0]))
    if x_test is not None:
        if len(x_test[0]) != len(x_train[0]):
            raise ValueError("X test dim != X train dim", len(x_test[0]), "train", len(x_train[0]))
    if y_test is not None:
        if len(y_test[0]) != len(y_train[0]):
            raise ValueError("Y test dim != Y train dim", len(x_test[0]), "train", len(x_train[0]))
    print("All checked, seems good.")