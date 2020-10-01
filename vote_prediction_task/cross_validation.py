import numpy as np
from sklearn.model_selection import KFold

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

PLOTS_PATH = 'Cross_valid_plots'

class crossValidator():
    '''
    This class perfroms k-Fold Cross-Validation on training set to choose
    the proper hyperparameters for each model. It selects the models
    with best performance.
    '''
    def __init__(self, train_x, train_y, num_of_folds):
        self.set_x = train_x
        self.set_y = train_y
        self.k = num_of_folds
        self.kf = KFold(n_splits=num_of_folds)
        self.best_svm = None
        self.best_knn = None
        self.best_random_forest = None
        self.best_mlp = None
        self.num_of_classes = 13

    def tuneSVM(self, c_steps ,type='coarse', graphic=True):
        scores = []
        c_list = c_steps
        for C in c_list:
            print("training for C=", C)
            avg_score = 0
            model = svm.SVC(kernel='rbf', C=C)
            for train_index, test_index in self.kf.split(self.set_x):
                x_train, x_test = self.set_x[train_index], self.set_x[test_index]
                y_train, y_test = self.set_y[train_index], self.set_y[test_index]
                model.fit(x_train, y_train)
                score = model.score(x_test, y_test)
                print("score ", score)
                avg_score += score
            scores.append(avg_score/self.k)
            print("avg score = ", avg_score/self.k)

        if graphic:
            plt.plot(c_list, scores)
            plt.title('C hyperparameter tuning for SVM')
            plt.ylabel('Accuracy')
            plt.xlabel('C value')
            if type == 'coarse':
                plt.xscale('log')
                name = 'SVM_C_hyper_fig_coarse.png'
            else:
                name = 'SVM_C_hyper_fig_fine.png'
            fig = plt.gcf()
            path = PLOTS_PATH + '\\' + name
            fig.savefig(path, bbox_inches='tight')
            plt.show()

    def tuneKNN(self, k_stop,graphic=True):
        scores_uniform = []
        scores_distance = []
        k_list = np.arange(1, k_stop, 2)
        for k in k_list:
            print("training for k=", k)
            avg_score_uniform = 0
            avg_score_distance = 0
            knn_uniform = KNeighborsClassifier(n_neighbors=k, weights='uniform')
            knn_distance = KNeighborsClassifier(n_neighbors=k, weights='distance')
            for train_index, test_index in self.kf.split(self.set_x):
                x_train, x_test = self.set_x[train_index], self.set_x[test_index]
                y_train, y_test = self.set_y[train_index], self.set_y[test_index]
                knn_uniform.fit(x_train, y_train)
                knn_distance.fit(x_train, y_train)
                score_uniform = knn_uniform.score(x_test, y_test)
                score_distance = knn_distance.score(x_test,y_test)
                avg_score_uniform += score_uniform
                avg_score_distance += score_distance
            scores_uniform.append(avg_score_uniform/self.k)
            scores_distance.append(avg_score_distance/self.k)
            print("avg scores = ", avg_score_uniform/self.k, "  ", avg_score_distance/self.k)

        if graphic:
            plt.plot(k_list, scores_uniform, label="uniform")
            plt.plot(k_list, scores_distance, label='distance')
            plt.title('k hyper parameter tuning for knn')
            plt.ylabel('Accuracy')
            plt.xlabel('number of neighbours')
            plt.legend()
            fig = plt.gcf()
            path = PLOTS_PATH + '\\' + 'k_hyper_fig.png'
            fig.savefig(path, bbox_inches='tight')
            plt.show()

    def tuneNForest(self, n_stop, graphic=True):
        scores1 = []
        scores2 = []
        scores3 = []
        n_list = np.arange(1, n_stop, 1)
        for n in n_list:
            print("training for n=", n)
            avg_score1 = 0
            avg_score2 = 0
            avg_score3 = 0

            forest1 = RandomForestClassifier(n_estimators=n, max_depth=5, min_samples_split=0.1)
            forest2 = RandomForestClassifier(n_estimators=n, max_depth=15, min_samples_split=0.1)
            forest3 = RandomForestClassifier(n_estimators=n, max_depth=35, min_samples_split=0.1)
            for train_index, test_index in self.kf.split(self.set_x):
                x_train, x_test = self.set_x[train_index], self.set_x[test_index]
                y_train, y_test = self.set_y[train_index], self.set_y[test_index]
                forest1.fit(x_train, y_train)
                forest2.fit(x_train, y_train)
                forest3.fit(x_train, y_train)
                score1 = forest1.score(x_test, y_test)
                score2 = forest2.score(x_test, y_test)
                score3 = forest3.score(x_test, y_test)
                avg_score1 += score1
                avg_score2 += score2
                avg_score3 += score3
            scores1.append(avg_score1/self.k)
            scores2.append(avg_score2 / self.k)
            scores3.append(avg_score3 / self.k)

        if graphic:
            plt.plot(n_list, scores1, label="depth=5")
            plt.plot(n_list, scores2, label="depth=10")
            plt.plot(n_list, scores3, label="depth=15")
            plt.title('num of estimators hyper parameter tuning for random forest')
            plt.ylabel('Accuracy')
            plt.xlabel('number of estimators')
            plt.legend()
            fig = plt.gcf()
            path = PLOTS_PATH + '\\' + 'n_hyper_forest_fig.png'
            fig.savefig(path, bbox_inches='tight')
            plt.show()

    def tuneDepthForest(self, d_stop,graphic=True):
        scores1 = []
        scores2 = []
        scores3 = []
        d_list = np.arange(1, d_stop, 1)
        for d in d_list:
            print("training for d=", d)
            avg_score1 = 0
            avg_score2 = 0
            avg_score3 = 0
            forest1 = RandomForestClassifier(n_estimators=25, max_depth=d, min_samples_split=0.1)
            forest2 = RandomForestClassifier(n_estimators=50, max_depth=d, min_samples_split=0.1)
            forest3 = RandomForestClassifier(n_estimators=100, max_depth=d, min_samples_split=0.1)
            for train_index, test_index in self.kf.split(self.set_x):
                x_train, x_test = self.set_x[train_index], self.set_x[test_index]
                y_train, y_test = self.set_y[train_index], self.set_y[test_index]
                forest1.fit(x_train, y_train)
                forest2.fit(x_train, y_train)
                forest3.fit(x_train, y_train)
                score1 = forest1.score(x_test, y_test)
                score2 = forest2.score(x_test, y_test)
                score3 = forest3.score(x_test, y_test)
                avg_score1 += score1
                avg_score2 += score2
                avg_score3 += score3
            scores1.append(avg_score1/self.k)
            scores2.append(avg_score2 / self.k)
            scores3.append(avg_score3 / self.k)

        if graphic:
            plt.plot(d_list, scores1, label="n_estimators=25")
            plt.plot(d_list, scores2, label="n_estimators=50")
            plt.plot(d_list, scores3, label="n_estimators=100")
            plt.title('max depth hyper parameter tuning for random forest')
            plt.ylabel('Accuracy')
            plt.xlabel('max depth')
            plt.legend()
            fig = plt.gcf()
            path = PLOTS_PATH + '\\' + 'd_hyper_forest_fig.png'
            fig.savefig(path, bbox_inches='tight')
            plt.show()

    def tuneSplitForest(self, graphic=True):
        scores1 = []
        scores2 = []
        scores3 = []
        s_list = np.arange(0.01, 1, 0.05)
        for s in s_list:
            print("training for s=", s)
            avg_score1 = 0
            avg_score2 = 0
            avg_score3 = 0
            forest1 = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=s)
            forest2 = RandomForestClassifier(n_estimators=50, max_depth=15, min_samples_split=s)
            forest3 = RandomForestClassifier(n_estimators=50, max_depth=50, min_samples_split=s)
            for train_index, test_index in self.kf.split(self.set_x):
                x_train, x_test = self.set_x[train_index], self.set_x[test_index]
                y_train, y_test = self.set_y[train_index], self.set_y[test_index]
                forest1.fit(x_train, y_train)
                forest2.fit(x_train, y_train)
                forest3.fit(x_train, y_train)
                score1 = forest1.score(x_test, y_test)
                score2 = forest2.score(x_test, y_test)
                score3 = forest3.score(x_test, y_test)
                avg_score1 += score1
                avg_score2 += score2
                avg_score3 += score3
            scores1.append(avg_score1/self.k)
            scores2.append(avg_score2 / self.k)
            scores3.append(avg_score3 / self.k)

        if graphic:
            plt.plot(s_list, scores1, label="max_depth=5")
            plt.plot(s_list, scores2, label="max_depth=15")
            plt.plot(s_list, scores3, label="max_depth=50")
            plt.title('min_samples_split parameter tuning for random forest')
            plt.ylabel('Accuracy')
            plt.xlabel('Min samples allowed to split')
            plt.legend()
            fig = plt.gcf()
            path = PLOTS_PATH + '\\' + 's_hyper_forest_fig.png'
            fig.savefig(path, bbox_inches='tight')
            plt.show()

    def tuneMLP(self, max_iter = 500, graphic=True):
        scores1 = []
        scores2 = []
        scores3 = []
        scores4 = []
        scores5 = []
        scores6 = []
        h_list = np.arange(15, 170, 20)
        for h in h_list:
            print("training for h=", h)
            avg_score1 = 0
            avg_score2 = 0
            avg_score3 = 0
            avg_score4 = 0
            avg_score5 = 0
            avg_score6 = 0
            mlp1 = MLPClassifier([h], activation='relu', max_iter=max_iter)
            mlp2 = MLPClassifier([h, h], activation='relu', max_iter=max_iter)
            mlp3 = MLPClassifier([h, h, h], activation='relu', max_iter=max_iter)
            mlp4 = MLPClassifier([h], activation='tanh', max_iter=max_iter)
            mlp5 = MLPClassifier([h, h], activation='tanh', max_iter=max_iter)
            mlp6 = MLPClassifier([h, h, h], activation='tanh', max_iter=max_iter)
            for train_index, test_index in self.kf.split(self.set_x):
                x_train, x_test = self.set_x[train_index], self.set_x[test_index]
                y_train, y_test = self.set_y[train_index], self.set_y[test_index]
                mlp1.fit(x_train, y_train)
                mlp2.fit(x_train, y_train)
                mlp3.fit(x_train, y_train)
                mlp4.fit(x_train, y_train)
                mlp5.fit(x_train, y_train)
                mlp6.fit(x_train, y_train)
                score1 = mlp1.score(x_test, y_test)
                score2 = mlp2.score(x_test, y_test)
                score3 = mlp3.score(x_test, y_test)
                score4 = mlp4.score(x_test, y_test)
                score5 = mlp5.score(x_test, y_test)
                score6 = mlp6.score(x_test, y_test)
                avg_score1 += score1
                avg_score2 += score2
                avg_score3 += score3
                avg_score4 += score4
                avg_score5 += score5
                avg_score6 += score6
            scores1.append(avg_score1 / self.k)
            scores2.append(avg_score2 / self.k)
            scores3.append(avg_score3 / self.k)
            scores4.append(avg_score4 / self.k)
            scores5.append(avg_score5 / self.k)
            scores6.append(avg_score6 / self.k)

        if graphic:
            plt.plot(h_list, scores1, label="1 hidden, relu")
            plt.plot(h_list, scores2, label="2 hidden, relu")
            plt.plot(h_list, scores3, label="3 hidden, relu")
            plt.plot(h_list, scores4, label="1 hidden, tanh")
            plt.plot(h_list, scores5, label="2 hidden, tanh")
            plt.plot(h_list, scores6, label="3 hidden, tanh")
            plt.title('hidden layers size tuning for MLP')
            plt.ylabel('Accuracy')
            plt.xlabel('hidden layers size')
            plt.legend()
            fig = plt.gcf()
            path = PLOTS_PATH + '\\' + 'mlp_h_fig.png'
            fig.savefig(path, bbox_inches='tight')
            plt.show()

