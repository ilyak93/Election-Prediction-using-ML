from itertools import count

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU, ReLU, Activation
from keras.layers import Dropout
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
import keras.optimizers
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from collections import Counter

from pandas import DataFrame
from sklearn.metrics import accuracy_score
import os
import pandas as pd

from data_infrastructure import divide_data

label = 'Vote'

class MLP_ensemble():
    def __init__(self, path, parties_dict):
        self.num_of_classes = 13
        self.party_dict = parties_dict
        self.models = []

        for filename in os.listdir(path):
            model = load_model(path + '\\' + filename)
            self.models.append(model)

    def score(self, x_test, y_test):
        y_test = np.argmax(y_test, axis=1)
        y_pred = self.predict(x_test)
        acc = accuracy_score(y_test, y_pred)*100
        print(str(acc)+'%')

    def predict(self, x_test):
        num_samples = x_test.shape[0]
        pred = np.zeros((len(self.models), num_samples))

        for idx, model in enumerate(self.models):
            pred[idx:] = np.argmax(model.predict(x_test), axis=1)

        final_pred = []
        for idx in range(num_samples):
            ens_pred = pred[:,idx]
            final_pred.append(Counter(ens_pred).most_common(1)[0][0])
        return np.array(final_pred)

    def save_votes_to_csv(self, x_test):
        predictions = self.predict(x_test)
        pred_pd = pd.DataFrame({'Vote': predictions})
        pred_pd.to_csv('predicted_labels.csv')

    def predict_winner(self, x_test):
        predictions = self.predict(x_test)
        winner = max(set(predictions), key=predictions.tolist().count)
        winner_name = self.party_dict[winner]
        print("Winner prediction - ", winner_name, " party will win the elections.")
        return winner_name

    def predict_vote_division(self, x_test):
        predictions = self.predict(x_test)
        pred_hist = [0] * self.num_of_classes
        for pred in predictions:
            pred_hist[int(pred)] += 1
        total = sum(pred_hist)
        print(" prediction - Vote division:")
        for idx, num in enumerate(pred_hist):
            print("Party ", self.party_dict[idx], ":", np.round((num / total) * 100, 3), "%")

    def write_pred_to_csv(self, x_test, id_test):
        pred = self.predict(x_test)
        pred_names = []
        for p in pred:
            pred_names.append(self.party_dict[p])

        sorted_pred_names = [x for _, x in sorted(zip(id_test, pred_names))]
        sorted_ids = sorted(id_test)

        data = pd.DataFrame({'IdentityCard_Num': sorted_ids, 'PredictVote': sorted_pred_names})
        data.to_csv('vote_predictions.csv', sep=',', index=False)

        unlabeled_set = pd.read_csv('.\\'+'unlabeled_set.csv')

        df = unlabeled_set.sort_values('IdentityCard_Num')
        data = data.sort_values('IdentityCard_Num')

        #data = data['Vote'] = pred_names
        idx = len(list(data.columns))
        df.insert(idx, "Vote", data['PredictVote'], True)
        df = df.drop('IdentityCard_Num', axis=1)
        df.to_csv('vote_predictions2.csv', sep=',', index=False)





class Keras_MLP():
    def __init__(self, n_hidden_list, num_features=9, lrelu_alpha=0.1,
                 drop_p=0.2, max_epochs=1000, activation='lrelu', scheduling=False, b_norm=False):
        # model params:
        self.scheduling = scheduling
        self.opt_patience = 5
        self.val_split = 0.05 # it used only for monitoring
        self.max_epochs = max_epochs
        # Create model
        self.model = Sequential()

        for idx, n in enumerate(n_hidden_list):
            if idx == 0:
                self.model.add(Dense(n, activation='linear', input_shape=(num_features,)))

                if b_norm:
                    self.model.add(BatchNormalization())

                if activation == 'lrelu':
                    self.model.add(LeakyReLU(alpha=lrelu_alpha))
                elif activation == 'tanh':
                    self.model.add(Activation('tanh'))
                else:
                    self.model.add(ReLU())
                self.model.add(Dropout(drop_p))
            else:
                self.model.add(Dense(n, activation='linear'))

                if b_norm:
                    self.model.add(BatchNormalization())

                if activation == 'lrelu':
                    self.model.add(LeakyReLU(alpha=lrelu_alpha))
                elif activation == 'tanh':
                    self.model.add(Activation('tanh'))
                else:
                    self.model.add(ReLU())
                self.model.add(Dropout(drop_p))

        self.model.add(Dense(13, activation='softmax'))

        opt = keras.optimizers.Adam(beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=None,
                                    decay=0.0,
                                    amsgrad=False)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

        if self.scheduling:
            self.scheduler = Scheduler()

    def save(self, path):
        self.model.save(path)

    def fit(self, x_train, y_train, graphic=False):
        # set early stopping monitor so the model stops training when it won't improve anymore
        early_stopping_monitor = EarlyStopping(patience=self.opt_patience)

        if self.scheduling:
            lrate = LearningRateScheduler(Scheduler().schedule, verbose=0)
            # train model
            hist = self.model.fit(x_train, y_train, validation_split=self.val_split,
                              epochs=self.max_epochs, callbacks=[early_stopping_monitor, lrate], verbose=0)
        else:
            # train model
            hist = self.model.fit(x_train, y_train, validation_split=self.val_split,
                                  epochs=self.max_epochs, callbacks=[early_stopping_monitor], verbose=0)

        if graphic:
            for key in hist.history:
                data = hist.history[key]
                epochs = range(len(hist.history[key]))
                target = [0.95]*len(hist.history[key])
                plt.plot(epochs, data, label=key)
            plt.plot(epochs, target, label='target = 0.95')
            plt.legend()
            plt.xlabel('epochs')
            plt.show()

    def predict(self, x_test):
        return self.model.predict(x_test)


class Scheduler:
    def __init__(self, lr=0.01, epochs_list=[10,20,30], decay=0.1):
        self.epochs_list = epochs_list
        self.decay = decay
        self.lr = lr
        self.current_milestone_idx = 0

    def schedule(self, epoch):
        if epoch == self.epochs_list[self.current_milestone_idx]:
            if self.current_milestone_idx + 1 < len(self.epochs_list):
                self.current_milestone_idx += 1
            self.lr = self.lr * self.decay
        return self.lr
