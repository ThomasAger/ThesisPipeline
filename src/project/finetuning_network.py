import numpy as np
from util import io
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from common.SaveLoadPOPO import SaveLoadPOPO
from common.Method import Method
from keras.optimizers import SGD, Adagrad, Adam, RMSprop, Adadelta, Adamax, Nadam
from keras.callbacks import TensorBoard

class FineTuneNetwork(Method):
    output_ranks = None
    output_folder = None
    multidim_array = None
    normalized_output = None
    epoch = None

    def __init__(self, file_name, output_folder, space, directions, ppmi_boc, log_dir, hidden_layer_size, epoch, save_class):

        self.space = space
        self.directions = directions
        self.ppmi_boc = ppmi_boc
        self.log_dir = log_dir
        self.output_folder = output_folder
        self.hidden_layer_size = int(len(space[0]) * hidden_layer_size)
        self.epoch = epoch

        super().__init__(file_name, save_class)

    def makePopos(self):
        self.output_ranks = SaveLoadPOPO(self.output_ranks, self.output_folder + self.file_name + ".npy", "npy")

    def makePopoArray(self):
        self.popo_array = [self.output_ranks]

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
        model = Sequential()

        model.add(Dense(output_dim=self.hidden_layer_size, input_dim=len(self.space[0]), activation="tanh", init="glorot_uniform"))

        model.add(Dense(output_dim=len(self.ppmi_boc),input_dim=self.hidden_layer_size,  activation="linear",
                        init="glorot_uniform",
                        weights=fine_tune_weights))

        model.compile(loss="mse", optimizer=Adagrad(lr=0.01, epsilon=None, decay=0.0))

        model.fit(self.space, self.ppmi_boc.transpose(), nb_epoch=self.epoch, batch_size=200, verbose=1)

        self.output_ranks.value = model.predict(self.space)
        print("completed")
        super().process()

    def getRanks(self):
        if self.processed is False:
            self.output_ranks.value = self.save_class.load(self.output_ranks)
        return self.output_ranks.value

