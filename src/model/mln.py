from common import Method
from common.SaveLoadPOPO import SaveLoadPOPO
#http://www.jmlr.org/papers/volume18/16-174/16-174.pdf
# This paper states that a 2-fold procedure for resampling/cross-validation is the best for datasets with 1000 or more datapoints
# 2-fold cross validation is where the data-set is split into two, and the model is trained twice, once with the training set being
# The first "fold" and test set being the second fold, and the second where the training set is the second fold.
# Tune C parameters for linear, gammas and C parameters for rbf
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from common.SaveLoadPOPO import SaveLoadPOPO
from keras.optimizers import SGD, Adagrad, Adam, RMSprop, Adadelta, Adamax, Nadam
from keras.callbacks import TensorBoard
from keras.initializers import Identity, Zeros, Ones, Constant, Orthogonal
from keras.utils.vis_utils import plot_model
from util import nnet
import os
class MultiLabelNetwork(Method.ModelMethod):
    max_features = None
    class_weight = None
    verbose = None

    max_depth = None
    min_samples_leaf = None
    min_samples_split = None
    hidden_layer_rep = None
    feature_names = None
    class_names = None
    tree_image_fn = None
    model = None
    epoch = None
    space = None
    batch_size = None
    get_rep = None

    def __init__(self, x_train, y_train, x_test, y_test, space, file_name, save_class, epoch=0,  class_weight=None, activation_function=None,
                 dropout=None, hidden_layer_size=None, verbose=False, feature_names=None, class_names=None, batch_size=None, get_rep=False):
        self.activation_function = activation_function
        self.verbose = verbose
        self.class_weight = class_weight
        self.dropout = dropout
        self.hidden_layer_size = hidden_layer_size
        self.feature_names = feature_names
        self.class_names = class_names
        self.epoch = epoch
        self.space = space
        self.batch_size = batch_size
        self.get_rep = get_rep
        # Probability is set to true
        super().__init__(x_train, y_train, x_test, y_test, file_name, save_class, True, None)

    def makePopoArray(self):
        super().makePopoArray()
        if self.get_rep:
            self.popo_array.append(self.hidden_layer_rep)

    def makePopos(self):
        super().makePopos()
        self.hidden_layer_rep = SaveLoadPOPO(self.hidden_layer_rep,self.file_name + "rep.npy", "npy")


    def process(self):
        self.model = Sequential()

        if self.hidden_layer_size < 100:
            hidden_size = int(len(self.x_train[0]) * self.hidden_layer_size)
        else:
            hidden_size = self.hidden_layer_size
        print("Hidden layer")
        self.model.add(
            Dense(output_dim=hidden_size, input_dim=len(self.x_train[0]), activation=self.activation_function,
                  init="glorot_uniform"))

        #self.model.add(Dropout(rate=self.dropout))

        print("Output no init")
        self.model.add(Dense(output_dim=len(self.y_train[0]), input_dim=hidden_size, activation="sigmoid"))


        self.model.compile(loss="binary_crossentropy", optimizer=Adagrad(lr=0.01, epsilon=None, decay=0.0))

        if self.verbose is True:
            print("Model plot")
            print(os.listdir())
            plot_model(self.model, to_file='plots/model_plot.png', show_shapes=True, show_layer_names=True)

        # self.ppmi_boc.transpose()
        self.model.fit(self.x_train, self.y_train, nb_epoch=self.epoch, batch_size=self.batch_size, verbose=1)
        self.hidden_layer_rep.value = nnet.getFirstLayer(self.model, self.space)

        self.test_proba.value = self.model.predict(self.x_test)
        self.test_predictions.value = nnet.probaToBinary(self.test_proba.value)

        super().process()


