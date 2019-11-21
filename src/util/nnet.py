
from keras.models import Sequential
import numpy as np
def getFirstLayer(model, x):
    amt_to_subtract = 1
    if len(model.layers) == 1:
        amt_to_subtract = 0
    for l in range(0, len(model.layers) - amt_to_subtract):
        print("Writing", l, "layer")
        truncated_model = Sequential()
        for a in range(l + 1):
            truncated_model.add(model.layers[a])
        truncated_model.compile(loss="binary_crossentropy", optimizer="sgd")
        return truncated_model.predict(x)


def probaToBinary(y_pred, threshold=0.5):
    y_pred = np.asarray(y_pred)
    for y in range(len(y_pred)):
        y_pred[y][y_pred[y] >= threshold] = 1
        y_pred[y][y_pred[y] < threshold] = 0
    return y_pred


if __name__ == '__main__':
    print(probaToBinary([[0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6], [0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6], [0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6]]))