import numpy as np
from util import io
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
from keras.optimizers import SGD, Adagrad, Adam, RMSprop, Adadelta, Adamax, Nadam
from keras.callbacks import TensorBoard
from keras.initializers import Identity, Zeros, Ones, Constant, Orthogonal
from util import nnet
class FineTuneNetwork(Method):
    output_ranks = None
    output_folder = None
    multidim_array = None
    normalized_output = None
    rankings = None
    epoch = None
    activation = None
    model = None
    loss = "mse"
    trainer = None
    layer_space = None
    use_hidden = None
    use_weights = None

    def __init__(self, file_name, output_folder, space, directions, rankings, ppmi_boc, log_dir, hidden_layer_size,  activation, epoch, save_class, use_hidden, use_weights):

        self.space = space
        self.directions = directions
        self.rankings = rankings
        self.ppmi_boc = ppmi_boc
        self.activation = activation
        self.log_dir = log_dir
        self.output_folder = output_folder
        self.hidden_layer_size = int(len(space[0]) * hidden_layer_size)
        self.epoch = epoch
        self.use_hidden = use_hidden
        self.use_weights = use_weights
        self.trainer = Adagrad(lr=0.01, epsilon=None, decay=0.0)

        super().__init__(file_name, save_class)

    def makePopos(self):
        self.output_ranks = SaveLoadPOPO(self.output_ranks, self.output_folder + self.file_name + "_output_ranks.npy", "npy")
        self.layer_space = SaveLoadPOPO(self.layer_space, self.output_folder + self.file_name + "_layer_space.npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.output_ranks, self.layer_space]

    def process(self):

        for a in range(len(self.directions)):
            self.directions[a] = np.around(self.directions[a], decimals=6)

        for a in range(len(self.ppmi_boc)):
            self.ppmi_boc[a] = np.around(self.ppmi_boc[a], decimals=6)

        # The bias term
        fine_tune_weights = []
        fine_tune_weights.append(self.directions.transpose())
        fine_tune_weights.append(np.zeros(shape=len(self.directions), dtype="float64"))
        print("FINETUNER")
        """
        tensorboard = TensorBoard(log_dir=self.log_dir,
                                  histogram_freq=0,
                                  write_graph=True, write_images=True)
        """
        self.model = Sequential()
        if self.use_hidden is True:
            print("Hidden layer")
            self.model.add(Dense(output_dim=self.hidden_layer_size, input_dim=len(self.space[0]), activation=self.activation, init="glorot_uniform"))
        elif self.use_hidden is "identity":
            print("Identity layer")
            self.model.add(
                Dense(output_dim=self.hidden_layer_size, input_dim=len(self.space[0]), activation=self.activation, init=Identity()))

        if self.use_weights:
            print("Output init with directions")
            self.model.add(Dense(output_dim=len(self.ppmi_boc), input_dim=self.hidden_layer_size,  activation="linear",
                            weights=fine_tune_weights))
        else:
            print("Output no init")
            self.model.add(Dense(output_dim=len(self.ppmi_boc), input_dim=self.hidden_layer_size,  activation="linear"))


        self.model.compile(loss=self.loss, optimizer=Adagrad(lr=0.01, epsilon=None, decay=0.0))

        orig_ranks = self.model.predict(self.space)

        self.layer_space = nnet.getFirstLayer(self.model, self.space)
        #self.ppmi_boc.transpose()
        self.model.fit(self.space, self.ppmi_boc.transpose(), nb_epoch=self.epoch, batch_size=200, verbose=1)

        self.output_ranks.value = self.model.predict(self.space)


        print("completed")
        super().process()




    def getRanks(self):
        if self.processed is False:
            self.output_ranks.value = self.save_class.load(self.output_ranks)
        return self.output_ranks.value

