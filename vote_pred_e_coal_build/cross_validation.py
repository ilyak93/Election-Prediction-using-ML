import numpy as np
import random
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from model import Keras_MLP
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

PLOTS_PATH = 'Cross_valid_plots'
ACC_THRESHOLD = 0.93
F1_THRESHOLD = 0.93
MODELS_PATH = 'saved_models'

class crossValidator():
    '''
    This class perfroms k-Fold Cross-Validation on training set to choose
    the proper hyperparameters for each model. It selects the models
    with best performance.
    '''
    def __init__(self, train_x, train_y, num_of_folds, max_epochs=1000):
        self.set_x = train_x
        self.set_y = train_y
        self.k = num_of_folds
        self.kf = KFold(n_splits=num_of_folds)
        self.best_model = None
        self.num_of_classes = 13
        self.max_epochs = max_epochs

    def rand_tune(self, iter=3, graphic=True):
        res_list = []

        for i in range(iter):
            avg_acc = 0
            avg_f1 = 0

            # Random params
            p = random.randrange(1, 60) / 100
            a = random.randrange(1, 50) / 100
            h_list = []

            for i in range(random.randint(1, 4)):
                h_list.append(random.randint(5, 200))

            rand = random.randint(0, 2)
            if rand == 0:
                act = 'lrelu'
            if rand == 1:
                act = 'tanh'
            if rand == 2:
                act = 'relu'

            rand = random.randint(0, 1)
            if rand == 0:
                scheduling = False
            if rand == 1:
                scheduling = True

            rand = random.randint(0, 1)
            if rand == 0:
                b_norm = False
            if rand == 1:
                b_norm = True

            mlp1 = Keras_MLP(n_hidden_list=h_list,
                             num_features=9,
                             lrelu_alpha=a,
                             drop_p=p,
                             max_epochs=self.max_epochs,
                             activation=act,
                             b_norm=b_norm)

            for train_index, test_index in self.kf.split(self.set_x):
                x_train, x_test = self.set_x[train_index], self.set_x[test_index]
                y_train, y_test = self.set_y[train_index], self.set_y[test_index]
                mlp1.fit(x_train, y_train)

                y_pred = mlp1.predict(x_test)
                f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted')
                acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

                avg_acc += acc
                avg_f1 += f1

            avg_acc = avg_acc / self.k
            avg_f1 = avg_f1 / self.k

            res = {'avg_acc': avg_acc, 'avg_f1': avg_f1,
                             'layers': h_list, 'drop_p': p, 'relu_slope': a,
                   'activation': act, 'scheduling': scheduling, 'b_norm': b_norm}
            print(res)
            res_list.append(res)
            with open('random_models.txt', 'a') as f:
                f.write("%s\n" % res)

            if avg_acc > ACC_THRESHOLD and avg_f1 > F1_THRESHOLD:
                print("saving...")
                mlp1.fit(self.set_x, self.set_y)
                rand_id = random.randint(1,1000000)
                path = MODELS_PATH + '\\' + "model_" + str(np.round(avg_acc, 4)) + '_' + str(rand_id) + ".h5"
                mlp1.save(path)

    def custom_tune(self, iter=3, graphic=True):
        res_list = []

        for i in range(iter):
            avg_acc = 0
            avg_f1 = 0

            # Random params
            p = 0.06
            a = 0.08
            h_list = [170, 174]
            act = 'tanh'
            scheduling = False
            b_norm = False

            mlp1 = Keras_MLP(n_hidden_list=h_list,
                             num_features=9,
                             lrelu_alpha=a,
                             drop_p=p,
                             max_epochs=self.max_epochs,
                             activation=act,
                             b_norm=b_norm)

            for train_index, test_index in self.kf.split(self.set_x):
                x_train, x_test = self.set_x[train_index], self.set_x[test_index]
                y_train, y_test = self.set_y[train_index], self.set_y[test_index]
                mlp1.fit(x_train, y_train)

                y_pred = mlp1.predict(x_test)
                f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted')
                acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

                avg_acc += acc
                avg_f1 += f1

            avg_acc = avg_acc / self.k
            avg_f1 = avg_f1 / self.k

            res = {'avg_acc': avg_acc, 'avg_f1': avg_f1,
                             'layers': h_list, 'drop_p': p, 'relu_slope': a,
                   'activation': act, 'scheduling': scheduling, 'b_norm': b_norm}
            print(res)
            res_list.append(res)
            with open('random_models.txt', 'a') as f:
                f.write("%s\n" % res)

            if avg_acc > ACC_THRESHOLD and avg_f1 > F1_THRESHOLD:
                print("saving...")
                mlp1.fit(self.set_x, self.set_y)
                rand_id = random.randint(1,1000000)
                path = MODELS_PATH + '\\' + "model_" + str(np.round(avg_acc, 4)) + '_' + str(rand_id) + ".h5"
                mlp1.save(path)




    def tune_dropout(self, graphic=True):
        acc_list = []
        f1_list = []

        p_list = np.arange(0.05, 0.9, 0.05)
        for p in p_list:
            print("training for p=", p)
            avg_acc = 0
            avg_f1 = 0

            mlp1 = Keras_MLP(n_hidden_list=[150, 100, 50],
                             num_features=9,
                             lrelu_alpha=0.1,
                             drop_p=p,
                             max_epochs=self.max_epochs)

            for train_index, test_index in self.kf.split(self.set_x):
                x_train, x_test = self.set_x[train_index], self.set_x[test_index]
                y_train, y_test = self.set_y[train_index], self.set_y[test_index]
                mlp1.fit(x_train, y_train)

                y_pred = mlp1.predict(x_test)
                f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted')
                acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

                avg_acc += acc
                avg_f1 += f1

            acc_list.append(avg_acc / self.k)
            f1_list.append(avg_f1 / self.k)

        if graphic:
            self.plot_valid(p_list, acc_list, f1_list,
                            title='Dropout p tuning for MLP',
                            ylabel='Accuracy, f1 score',
                            xlabel='Dropout probability',
                            filename='mlp_p_fig.png',
                            show=False)

    def tune_leaky_slope(self, graphic=True):
        acc_list = []
        f1_list = []

        a_list = np.arange(0.005, 0.4, 0.05)
        for a in a_list:
            print("training for a=", a)
            avg_acc = 0
            avg_f1 = 0

            mlp1 = Keras_MLP(n_hidden_list=[150, 100, 50],
                             num_features=9,
                             lrelu_alpha=a,
                             drop_p=0.2,
                             max_epochs=self.max_epochs)

            for train_index, test_index in self.kf.split(self.set_x):
                x_train, x_test = self.set_x[train_index], self.set_x[test_index]
                y_train, y_test = self.set_y[train_index], self.set_y[test_index]
                mlp1.fit(x_train, y_train)

                y_pred = mlp1.predict(x_test)
                f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted')
                acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

                avg_acc += acc
                avg_f1 += f1

            acc_list.append(avg_acc / self.k)
            f1_list.append(avg_f1 / self.k)

        if graphic:
            self.plot_valid(a_list, acc_list, f1_list,
                            title='Leaky ReLU slope tuning for MLP',
                            ylabel='Accuracy, f1 score',
                            xlabel='Slope',
                            filename='mlp_a_fig.png',
                            show=False)

    def tune_hidden_layers(self, graphic=True):
        acc_list = []
        f1_list = []

        h_list = [[150, 100, 50],
                  [150, 120, 80, 50],
                  [150, 130, 100, 70, 50],
                  [150, 140, 120, 100, 70, 50],
                  [150, 140, 120, 100, 80, 60, 50],
                  [150, 140, 130, 110, 90, 70, 60, 50],
                  [150, 140, 130, 120, 110, 100, 80, 70, 50]]
        h_list_idx = [3, 4, 5, 6, 7, 8, 9]

        for h in h_list:
            print("training for h=", h)
            avg_acc = 0
            avg_f1 = 0

            mlp1 = Keras_MLP(n_hidden_list=h,
                             num_features=9,
                             lrelu_alpha=0.1,
                             drop_p=0.2,
                             max_epochs=self.max_epochs)

            for train_index, test_index in self.kf.split(self.set_x):
                x_train, x_test = self.set_x[train_index], self.set_x[test_index]
                y_train, y_test = self.set_y[train_index], self.set_y[test_index]
                mlp1.fit(x_train, y_train)

                y_pred = mlp1.predict(x_test)
                f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted')
                acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

                avg_acc += acc
                avg_f1 += f1

            acc_list.append(avg_acc / self.k)
            f1_list.append(avg_f1 / self.k)

        if graphic:
            self.plot_valid(h_list_idx, acc_list, f1_list,
                            title='Number of hidden layers tuning for MLP',
                            ylabel='Accuracy, f1 score',
                            xlabel='Number of hidden layers',
                            filename='mlp_h_fig.png',
                            show=False)

    def plot_valid(self, param_list, acc_list, f1_list, title, ylabel, xlabel, filename, show=False):
        plt.plot(param_list, acc_list, label="Accuracy")
        plt.plot(param_list, f1_list, label="f1 score")

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        if show:
            fig = plt.gcf()
            path = PLOTS_PATH + '\\' + filename
            fig.savefig(path, bbox_inches='tight')
            plt.show()
        else:
            path = PLOTS_PATH + '\\' + filename
            plt.savefig(path, bbox_inches='tight')
            plt.clf()

