# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad, Adam, RMSprop, Adadelta
from keras.regularizers import l2
from sklearn.metrics import f1_score, accuracy_score

from util import proj as dt


class NeuralNetwork:

    # Shared variables
    training_data = None
    class_path = None
    learn_rate = None
    epochs = None
    loss = None
    batch_size = None
    hidden_activation = None
    layer_init = None
    output_activation = None
    hidden_layer_size = None
    file_name = None
    vector_path = None
    optimizer = None
    dropout_noise = None
    reg = 0.0
    activity_reg = 0.0
    finetune_size = 0
    save_outputs = False
    amount_of_hidden = 0
    identity_swap = None
    deep_size = None
    from_ae = None
    is_identity = None
    randomize_finetune_weights = None
    corrupt_finetune_weights = None
    deep_size = None
    fine_tune_weights_fn = None
    finetune_weights = None
    past_weights = None
    identity_activation = None

    def __init__(self,  class_path=None, get_scores=False,  randomize_finetune_weights=False, dropout_noise = None,
                 amount_of_hidden=0,
                 epochs=1,  learn_rate=0.01, loss="mse", batch_size=1, past_model_bias_fn=None, identity_swap=False,
                 reg=0.0, amount_of_finetune=[], output_size=25,
                 hidden_activation="tanh", layer_init="glorot_uniform", output_activation="tanh", deep_size = None,
                 corrupt_finetune_weights = False, split_to_use=-1,
                   hidden_layer_size=100, file_name="unspecified_filename", vector_path=None, is_identity=False,
                  finetune_size=0, data_type="movies",
                 optimizer_name="rmsprop", noise=0.0, fine_tune_weights_fn=None, past_model_weights_fn=None,
                 from_ae=True, save_outputs=False, label_names_fn="",
                 rewrite_files=False, cv_splits=1,cutoff_start=0.2, development=False,
                 class_weight=None, csv_fn=None, tune_vals=False, get_nnet_vectors_path=None, classification_name="all",
                 limit_entities=False, limited_label_fn="", vector_names_fn="", identity_activation="linear", loc="../data/",
                 lock_weights_and_redo=False):

        weights_fn =loc + data_type + "/nnet/weights/" + file_name + "L0.txt"
        bias_fn = loc + data_type + "/nnet/bias/" + file_name +"L0.txt"
        rank_fn =loc+ data_type + "/nnet/clusters/" + file_name + ".txt"

        all_fns = [weights_fn, bias_fn, rank_fn]
        if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
            print("Skipping task", "nnet")
            return
        else:

            print("Running task", "nnet")

        self.class_path = class_path
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.loss = loss
        self.batch_size = batch_size
        self.hidden_activation = hidden_activation
        self.layer_init = layer_init
        self.output_activation = output_activation
        self.hidden_layer_size = hidden_layer_size
        self.file_name = file_name
        self.vector_path = vector_path
        self.dropout_noise = dropout_noise
        self.finetune_size = finetune_size
        self.get_scores = get_scores
        self.reg = reg
        self.amount_of_finetune = amount_of_finetune
        self.amount_of_hidden = amount_of_hidden
        self.output_size = output_size
        self.identity_swap = identity_swap
        self.deep_size = deep_size
        self.from_ae = from_ae
        self.is_identity = is_identity
        self.randomize_finetune_weights = randomize_finetune_weights
        self.corrupt_finetune_weights = corrupt_finetune_weights
        self.deep_size = deep_size
        self.fine_tune_weights_fn = fine_tune_weights_fn
        self.identity_activation = identity_activation
        self.lock_weights_and_redo = lock_weights_and_redo

        print(data_type)

        if optimizer_name == "adagrad":
            self.optimizer = Adagrad()
        elif optimizer_name == "sgd":
            self.optimizer = SGD()
        elif optimizer_name == "rmsprop":
            self.optimizer = RMSprop()
        elif optimizer_name == "adam":
            self.optimizer = Adam()
        elif optimizer_name == "adadelta":
            self.optimizer = Adadelta()
        else:
            print("optimizer not found")
            exit()

        entity_vectors = np.asarray(dt.import2dArray(self.vector_path))
        print("Imported vectors", len(entity_vectors), len(entity_vectors[0]))

        if get_nnet_vectors_path is not None:
            nnet_vectors = np.asarray(dt.import2dArray(get_nnet_vectors_path))
            print("Imported vectors", len(entity_vectors), len(entity_vectors[0]))

        entity_classes = np.asarray(dt.import2dArray(self.class_path))
        print("Imported classes", len(entity_classes), len(entity_classes[0]))


        if fine_tune_weights_fn is None:
            vector_names = dt.import1dArray(vector_names_fn)
            limited_labels = dt.import1dArray(limited_label_fn)
            entity_vectors = np.asarray(dt.match_entities(entity_vectors, limited_labels, vector_names))

        if fine_tune_weights_fn is not None:
            if len(entity_vectors) != len(entity_classes):
                entity_classes = entity_classes.transpose()
                print("Transposed classes, now in form", len(entity_classes), len(entity_classes[0]))
                """
                # IF Bow
                if len(entity_vectors[0]) != len(entity_classes[0]):
                    entity_vectors = entity_vectors.transpose()
                    print("Transposed vectors, now in form", len(entity_vectors), len(entity_vectors[0]))
                """
        elif len(entity_vectors) != len(entity_classes):
            entity_vectors = entity_vectors.transpose()
            print("Transposed vectors, now in form", len(entity_vectors), len(entity_vectors[0]))


        self.input_size = len(entity_vectors[0])
        self.output_size = len(entity_classes[0])

        if fine_tune_weights_fn is not None:
            model_builder = self.fineTuneNetwork
            weights = []
            if from_ae:
                self.past_weights = []
                past_model_weights = []
                for p in past_model_weights_fn:
                    past_model_weights.append(np.asarray(dt.import2dArray(p), dtype="float64"))
                past_model_bias = []
                for p in past_model_bias_fn:
                    past_model_bias.append(np.asarray(dt.import1dArray(p, "f"), dtype="float64"))

                for p in range(len(past_model_weights)):
                    past_model_weights[p] = np.around(past_model_weights[p], decimals=6)
                    past_model_bias[p] = np.around(past_model_bias[p], decimals=6)

                for p in range(len(past_model_weights)):
                    self.past_weights.append([])
                    self.past_weights[p].append(past_model_weights[p])
                    self.past_weights[p].append(past_model_bias[p])
            for f in fine_tune_weights_fn:
                weights.extend(dt.import2dArray(f))

            r = np.asarray(weights, dtype="float64")
            r = np.asarray(weights, dtype="float64")

            for a in range(len(r)):
                r[a] = np.around(r[a], decimals=6)

            for a in range(len(entity_classes)):
                entity_classes[a] = np.around(entity_classes[a], decimals=6)

            self.fine_tune_weights = []
            self.fine_tune_weights.append(r.transpose())
            self.fine_tune_weights.append(np.zeros(shape=len(r), dtype="float64"))
        else:
            model_builder = self.classifierNetwork


        # Converting labels to categorical
        f1_scores = []
        accuracy_scores = []
        f1_averages = []
        accuracy_averages = []

        original_fn = file_name
        x_train, y_train, x_test, y_test, x_dev, y_dev = split_data.splitData(vectors, labels[l], data_type)

        if development:
            x_test = x_dev
            y_test = y_dev
        
        model = model_builder()
        
        if get_scores:
            test_pred = model.predict(x_train).transpose()
            print(test_pred)
            highest_vals = [0.5]*len(test_pred) # Default 0.5
            y_pred = model.predict(x_test).transpose()
            y_test = np.asarray(y_test).transpose()
            for y in range(len(y_pred)):
                y_pred[y][y_pred[y] >= highest_vals[y]] = 1
                y_pred[y][y_pred[y] < highest_vals[y]] = 0
            f1_array = []
            accuracy_array = []
            for y in range(len(y_pred)):
                accuracy_array.append(accuracy_score(y_test[y], y_pred[y]))
                f1_array.append(f1_score(y_test[y], y_pred[y], average="binary"))
                print(f1_array[y])
            y_pred = y_pred.transpose()
            y_test = np.asarray(y_test).transpose()

            micro_average = f1_score(y_test, y_pred, average="micro")

            cv_f1_fn = loc + data_type + "/nnet/scores/F1 " + file_name + ".txt"
            cv_acc_fn = loc + data_type + "/nnet/scores/ACC " + file_name + ".txt"
            dt.write1dArray(f1_array, cv_f1_fn)
            dt.write1dArray(accuracy_array, cv_acc_fn)

            f1_scores.append(f1_array)
            accuracy_scores.append(accuracy_array)
            f1_average = np.average(f1_array)
            accuracy_average = np.average(accuracy_array)
            f1_averages.append(f1_average)
            accuracy_averages.append(accuracy_average)

            print("Average F1 Binary", f1_average, "Acc", accuracy_average)
            print("Micro Average F1", micro_average)

            f1_array.append(f1_average)
            f1_array.append(micro_average)
            accuracy_array.append(accuracy_average)
            accuracy_array.append(0.0)

            scores = [accuracy_array, f1_array]

            csv_fn = loc+data_type+"/nnet/csv/"+csv_fn+".csv"

            file_names = [file_name + "ACC", file_name + "F1"]
            label_names = dt.import1dArray(label_names_fn)
            if dt.fileExists(csv_fn):
                print("File exists, writing to csv")
                try:
                    dt.write_to_csv(csv_fn, file_names, scores)
                except PermissionError:
                    print("CSV FILE WAS OPEN, WRITING TO ANOTHER FILE")
                    dt.write_to_csv(csv_fn[:len(csv_fn) - 4] + str(random.random()) + "FAIL.csv", [file_name],
                                    scores)
            else:
                print("File does not exist, recreating csv")
                key = []
                for l in label_names:
                    key.append(l)
                key.append("AVERAGE")
                key.append("MICRO AVERAGE")
                dt.write_csv(csv_fn, file_names, scores, key)

            if save_outputs:
                if limit_entities is False:
                    self.output_clusters = model.predict(nnet_vectors)
                else:
                    self.output_clusters = model.predict(entity_vectors)
                self.output_clusters = self.output_clusters.transpose()
                dt.write2dArray(self.output_clusters, rank_fn)


            for l in range(0, len(model.layers) - 1):
                if dropout_noise is not None and dropout_noise > 0.0:
                    if l % 2 == 1:
                        continue
                print("Writing", l, "layer")
                truncated_model = Sequential()
                for a in range(l+1):
                    truncated_model.add(model.layers[a])
                truncated_model.compile(loss=self.loss, optimizer="sgd")
                if get_nnet_vectors_path is not None:
                    self.end_space = truncated_model.predict(nnet_vectors)
                else:
                    self.end_space = truncated_model.predict(entity_vectors)
                total_file_name = loc + data_type + "/nnet/spaces/" + file_name
                dt.write2dArray(self.end_space, total_file_name + "L" + str(l) + ".txt")

            for l in range(len(model.layers)):
                try:
                    dt.write2dArray(model.layers[l].get_weights()[0],
                                    loc + data_type + "/nnet/weights/" + file_name + "L" + str(l) + ".txt")
                    dt.write1dArray(model.layers[l].get_weights()[1],
                                    loc + data_type + "/nnet/bias/" + file_name + "L" +  str(l) + ".txt")
                except IndexError:
                    print("Layer ", str(l), "Failed")


    def getEndSpace(self):
        return self.end_space

    def getEncoder(self):
        return self.model.layers[1]



    def classifierNetwork(self):
        print("CLASSIFIER")
        model = Sequential()
        print(self.input_size, self.hidden_layer_size, self.finetune_size, self.output_size)
        if deep_size is not None and len(deep_size) != 0:
            print(0, "Deep layer", self.input_size, self.deep_size[0], self.hidden_activation)
            model.add(Dense(output_dim=self.deep_size[0], input_dim=self.input_size, init=self.layer_init,
                            activation=self.hidden_activation, W_regularizer=l2(self.reg)))

        if self.dropout_noise is not None and self.dropout_noise != 0.0:
            print("Dropout layer")
            model.add(Dropout(self.dropout_noise))

        for a in range(1, len(self.deep_size)):
            print(a, "Deep layer", self.deep_size[a - 1], self.deep_size[a], self.hidden_activation)
            model.add(Dense(output_dim=self.deep_size[a], input_dim=self.deep_size[a - 1], init=self.layer_init,
                            activation=self.hidden_activation, W_regularizer=l2(self.reg)))
            if self.dropout_noise is not None and self.dropout_noise != 0.0:
                print("Dropout layer")
                model.add(Dropout(self.dropout_noise))

        print("Class outputs", self.output_size,
              self.output_activation)
        if deep_size is not None and len(deep_size) != 0:
            model.add(
                Dense(output_dim=self.output_size, input_dim=self.deep_size[len(self.deep_size) - 1],
                      activation=self.output_activation,
                      init=self.layer_init))
        else:
            model.add(
                Dense(output_dim=self.output_size, input_dim=self.input_size,
                      activation=self.output_activation,
                      init=self.layer_init))
        print("Compiling")
        model.compile(loss=self.loss, optimizer=self.optimizer)
        print(self.loss, self.optimizer)
        return model

    def fineTuneNetwork(self):
        print("FINETUNER")
        """
        tensorboard = TensorBoard(log_dir='/home/tom/Desktop/Logs/' + str(self.data_type) + "/" + file_name + '/',
                                  histogram_freq=0,
                                  write_graph=True, write_images=True)
        """   
        model = Sequential()
        print(self.input_size, self.hidden_layer_size, self.finetune_size, self.output_size)

        finetune_size = len(self.fine_tune_weights[0][0])
        #self.model.add(GaussianNoise(0.0, input_shape=(input_size,)))
        """
        # If we want to swap the identity layer to before the hidden layer
        if self.identity_swap:
            print("Identity swapped layer", self.input_size, self.hidden_layer_size, self.identity_activation)
            for a in range(len(self.amount_of_finetune)):
                model.add(Dense(output_dim=self.amount_of_finetune[a], activation= self.identity_activation, init = self.layer_init))

        if self.from_ae:
            for p in range(len(self.past_weights)):
                print(p, "Past AE layer", self.input_size, self.hidden_layer_size,  self.identity_activation)

                if self.dropout_noise is not None and self.dropout_noise != 0.0:
                    print("Dropout layer", self.dropout_noise)
                    model.add(Dropout(self.dropout_noise))

                model.add(Dense(output_dim=self.hidden_layer_size, input_dim=self.input_size, activation= self.identity_activation,
                                     weights=self.past_weights[p], W_regularizer=l2(self.reg)))
        """
        """
        # Add an identity layer that has equal values to the input space to find some more nonlinear relationships
        if self.is_identity:
            for a in range(len(self.amount_of_finetune)):
                if a == 0:
                    print("Identity layer", self.amount_of_finetune[a], self.input_size, self.identity_activation)
                    model.add(Dense(output_dim=self.amount_of_finetune[a], activation=self.identity_activation,
                          init=self.layer_init, input_dim=self.input_size))
                    if self.dropout_noise is not None and self.dropout_noise != 0.0:
                        print("Dropout layer")
                        model.add(Dropout(self.dropout_noise))
                else:
                    print("Identity layer", self.amount_of_finetune[a],  self.identity_activation)
                    model.add(Dense(output_dim=self.amount_of_finetune[a], activation=self.identity_activation,
                          init=self.layer_init))
                    if self.dropout_noise is not None and self.dropout_noise != 0.0:
                        print("Dropout layer")
                        model.add(Dropout(self.dropout_noise))
        """

        """
        if self.randomize_finetune_weights:
            print("Randomize finetune weights", self.hidden_layer_size, finetune_size, "linear")
            model.add(Dense(output_dim=finetune_size, input_dim=self.hidden_layer_size, activation="linear",
                                 init=self.layer_init))
        elif self.corrupt_finetune_weights:
            print("Corrupt finetune weights", self.hidden_layer_size, finetune_size, "linear")
            model.add(Dense(output_dim=finetune_size, input_dim=self.hidden_layer_size, activation="linear",
                                 weights=self.fine_tune_weights))
        else:
            
        """
        print("Fine tune weights", self.hidden_layer_size, finetune_size, self.identity_activation)
        model.add(Dense(output_dim=self.hidden_layer_size, input_dim=self.hidden_layer_size, activation=self.identity_activation))

        #model.add(Dropout(self.dropout_noise))
        """
        if self.get_scores:
            if self.randomize_finetune_weights or self.corrupt_finetune_weights or len(self.fine_tune_weights_fn) > 0:
                print("Class outputs", finetune_size, self.output_size, self.output_activation)
        """
        print("Output", self.output_size, self.finetune_size, self.output_activation, self.layer_init)
        model.add(Dense(output_dim=self.output_size, input_dim=self.finetune_size, activation=self.output_activation,
                  init=self.layer_init,
                             weights=self.fine_tune_weights))
        """
        if self.lock_weights_and_redo:
            model.layers[len(model.layers)-1].trainable = False
        """
        print("Compiling")
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model


import random


def main(loss, output_activation, optimizer_name, hidden_activation, ep, dropout_noise, cv_splits, deep_size,
             init_vector_path, data_type, rewrite_files, development, class_weight, classification_task, file_name, tune_vals):

    if isinstance(deep_size, str):
        ep = int(ep)
        dropout_noise = float(dropout_noise)
        cv_splits = int(cv_splits)
        deep_size = dt.stringToArray(deep_size)
        if class_weight != 'None':
            class_weight = float(class_weight)
        else:
            class_weight = None
        rewrite_files = bool(rewrite_files)
        development = bool(development)


    csv_fns = []
    split_fns = []
    for s in range(cv_splits):
        fn = file_name + " E" + str(ep) + " DS" + str(deep_size) + " DN" + str(dropout_noise) + " CT" + classification_task + \
                        " HA" + str(hidden_activation) + " CV" + str(cv_splits)  +  " S" + str(s) + " Dev" + str(development) + \
             " Opt" + optimizer_name
        split_fns.append(fn)

    for splits in range(cv_splits):
        file_name = split_fns[splits]
        print(deep_size, init_vector_path)
        loss = loss
        output_activation = output_activation
        optimizer_name = optimizer_name
        hidden_activation = hidden_activation
        classification_path = loc + data_type + "/classify/" + classification_task + "/class-All"
        label_names_fn = loc + data_type + "/classify/" + classification_task + "/names.txt"
        lr = 0.01
        fine_tune_weights_fn = None
        ep = ep
        amount_of_finetune = 0
        batch_size = 200
        save_outputs = True
        dropout_noise = dropout_noise
        is_identity = False
        identity_swap = False
        from_ae = False
        past_model_weights_fn = None
        past_model_bias_fn = None
        randomize_finetune_weights = False
        hidden_layer_size = 100
        output_size = 10
        randomize_finetune_weights = False
        corrupt_finetune_weights = False
        get_scores = True
        cs = 0.2
        deep_fns = []
        for s in range(len(deep_size)):
            deep_fns.append(split_fns[splits] + " SFT" + str(s))
        for d in range(len(deep_size)):
            file_name = deep_fns[d]
            SDA = NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                get_scores=get_scores, past_model_bias_fn=past_model_bias_fn, deep_size=deep_size,  cutoff_start=cs,
                randomize_finetune_weights=randomize_finetune_weights, amount_of_finetune=amount_of_finetune,
                vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=classification_path,
                identity_swap=identity_swap, dropout_noise=dropout_noise, save_outputs=save_outputs,
                hidden_activation=hidden_activation, output_activation=output_activation, epochs=ep,
                learn_rate=lr, is_identity=is_identity, output_size=output_size, split_to_use=splits,
                                label_names_fn=label_names_fn,
                batch_size=batch_size, past_model_weights_fn=past_model_weights_fn, loss=loss, cv_splits=cv_splits,
                                csv_fn = file_name, tune_vals=tune_vals,
                file_name=file_name, from_ae=from_ae, data_type=data_type, rewrite_files=rewrite_files, development=development,
                                class_weight=class_weight)

            csv_fns.append(loc+data_type+"/nnet/csv/"+file_name+".csv")
            new_file_names = []


    dt.averageCSVs(csv_fns)

loc = "../data/"
"""
data_type = "wines"
classification_task = "types"
file_name = "wines mds"
lowest_amt = 50
highest_amt = 10
init_vector_path = loc+data_type+"/nnet/spaces/wines100trimmed.txt"
"""
"""
data_type = "movies"
classification_task = "genres"
file_name = "movies mds"
lowest_amt = 100
highest_amt = 10
init_vector_path = loc+data_type+"/nnet/spaces/films200-"+classification_task+".txt"#
init_vector_path = loc+data_type+"/nnet/spaces/films200-genres100ndcg0.81200AllTerms3000FTL0.txt"
"""
data_type = "placetypes"
classification_task = "geonames"
lowest_amt = 50
highest_amt = 10
#init_vector_path = loc+data_type+"/bow/ppmi/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-"+classification_task
#file_name = "placetypes bow"
init_vector_path = loc+data_type+"/nnet/spaces/places100-"+classification_task+".txt"
file_name = "placetypes mds"
"""
hidden_activation = "relu"
dropout_noise = 0.5
output_activation = "sigmoid"
cutoff_start = 0.2
ep=200
"""
hidden_activation = "tanh"
dropout_noise = 0.6
output_activation = "softmax"
cutoff_start = 0.2
deep_size = [100]
#init_vector_path = loc+data_type+"/bow/ppmi/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-"+classification_task
#file_name = "movies ppmi"
ep =8000
tune_vals = True
class_weight = None
optimizer_name = "adadelta"
loss="categorical_crossentropy"
development = True
cv_splits = 5
rewrite_files = True
threads=3

variables = [loss, output_activation, optimizer_name, hidden_activation, ep, dropout_noise, cv_splits, deep_size,
             init_vector_path, data_type, rewrite_files, development, class_weight, classification_task, file_name]
""""
print("python nnet.py", end=' ')
for v in range(len(variables)):
    if v == len(variables)-1:
        print(variables[v])
    else:
        print(variables[v], end=' ')


args = sys.argv[1:]
print(args)

if len(args) > 0:
    loss = args[0]
    output_activation = args[1]
    optimizer_name = args[2]
    hidden_activation = args[3]
    ep = args[4]
    dropout_noise = args[5]
    cv_splits = args[6]
    deep_size = args[7]
    init_vector_path = args[8]
    data_type = args[9]
    rewrite_files = args[10]
    development = args[11]
    class_weight = args[12]
    classification_task = args[13]
    file_name = args[14]
    """
if  __name__ =='__main__':main(loss, output_activation, optimizer_name, hidden_activation, ep, dropout_noise, cv_splits, deep_size,
             init_vector_path, data_type, rewrite_files, development, class_weight, classification_task, file_name, tune_vals)

