import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.svm import SVC

plt.rcParams.update({'font.size': 5})
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import os
from impute import *
from transform import *
import re

from data_infrastructure import *

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings('ignore')


def sfs_algo(x_train: DataFrame, y_train: DataFrame, clf, subset_size: int = None):
    """
    :param x_train: DataFrame
    :param y_train: labels
    :param clf: classifier to examine
    :param subset_size: user required subset size
    :return: selected feature subset
    """
    subset_selected_features = []
    best_total_score = float('-inf')

    if subset_size:
        subset_size = min(len(features_without_label), subset_size)
    else:
        subset_size = len(features_without_label)

    for _ in range(subset_size):
        best_score = float('-inf')
        best_feature = None
        unselect_features = [f for f in features_without_label if f not in subset_selected_features]
        for f in unselect_features:
            current_features = subset_selected_features + [f]
            current_score = score(x_train[current_features], y_train, clf)
            if current_score > best_score:
                best_score = current_score
                best_feature = f
        if best_score > best_total_score:
            best_total_score = best_score
            subset_selected_features.append(best_feature)
        else:
            break
    return subset_selected_features


def run_sfs_base_clfs(x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame, x_val: DataFrame, y_val: DataFrame):
    # examine sfs algorithm with SVM
    dtc = SGDClassifier(random_state=92, max_iter=1000, tol=1e-3)
    score_before_sfs = score(x_train, y_train, dtc)
    print("SVM Classifier accuracy score before SFS is: {}".format(score_before_sfs))

    selected_features_dtc = sfs_algo(x_train, y_train, dtc)
    print("SVM Classifier selected features are: {}".format(selected_features_dtc))

    score_after_sfs = score(x_train[selected_features_dtc], y_train, dtc)
    print("SVM Classifier score after SFS is: {}".format(score_after_sfs))

    # examine sfs algorithm with K Neighbors Classifier

    knn = KNeighborsClassifier(n_neighbors=5)
    score_before_sfs = score(x_train, y_train, knn)
    print("K Neighbors Classifier score before SFS is: {}".format(score_before_sfs))

    selected_features_knn = sfs_algo(x_train, y_train, knn)
    print("K Neighbors Classifier selected features are: {}".format(selected_features_knn))

    score_after_sfs = score(x_train[selected_features_knn], y_train, knn)
    print("K Neighbors Classifier score after SFS is: {}".format(score_after_sfs))

    return selected_features_dtc, selected_features_knn

map_selected_features = {
    0: "selected_features_my_sfs_knn",
    1: "selected_features_my_sfs_sgd",
    2: "my_sfs_knn_sgd",
    3: "my_sfs_MI",
    4: "my_all_selected_together",
    5: "selected_features_lib_sfs_knn",
    6: "selected_features_lib_sfs_sgd",
    7: "selected_features_MI",
    8: "lib_sfs_knn_sgd",
    9: "lib_sfs_MI",
    10: "lib_all_selected_together",
    11: "my_sfs_knn_lib_sfs_knn",
    12: "my_sfs_knn_lib_sfs_knn"
}


def main():
    train_set, val_set, test_set = load_and_split('ElectionsData.csv', '../../Aviv20/ML/hw/hw2')
    train_set_copy, val_set_copy, test_set_copy = train_set.copy(), val_set.copy(), test_set.copy()


    train_set, val_set, test_set = set_clean(train_set, val_set, test_set,
                                             verbose=True, graphic=True)
    train_set, val_set, test_set = remove_inconsistency(train_set, val_set, test_set)

    train_set, val_set, test_set = data_transformation(train_set, val_set, test_set, False)
    assert sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)]) == 0

    selected_features_my_sfs, selected_features_my_sgd, selected_features_sfs, selected_features_sgd, selected_features_MI = set_reduction(train_set, val_set, test_set)

    my_sfs_sgd = list((set(selected_features_my_sfs).union(selected_features_my_sgd)))
    sfs_sgd = list((set(selected_features_sfs).union(selected_features_sgd)))

    my_sfs_MI = list((set(selected_features_my_sfs).union(selected_features_MI)))
    sfs_MI = list((set(selected_features_sfs).union(selected_features_MI)))

    my_all = list((set(selected_features_my_sfs).union(selected_features_my_sgd)).union(selected_features_MI))
    all = list((set(selected_features_sfs).union(selected_features_sgd)).union(selected_features_MI))

    my_sfs_e_sfs = list((set(selected_features_my_sfs).union(selected_features_sfs)))
    my_sfs_e_sfs_MI = list((set(my_sfs_e_sfs).union(selected_features_MI)))
    selected_features_list = [selected_features_my_sfs,
                              selected_features_my_sgd,
                              my_sfs_sgd,
                              my_sfs_MI,
                              my_all,
                              selected_features_sfs, selected_features_sgd,
                              selected_features_MI,
                              sfs_sgd, sfs_MI, all, my_sfs_e_sfs,
                              my_sfs_e_sfs_MI]
    all_sets_orig =  [train_set.copy(), val_set.copy(), test_set.copy()]
    for i, selected_features in enumerate(selected_features_list):
        print(map_selected_features[i])
        print(selected_features)
        selected_features.append('Vote')
        all_sets = [train_set, val_set, test_set]
        for index, data_set in enumerate(all_sets):
            all_sets[index] = all_sets[index][selected_features]
            [train_set, val_set, test_set] = all_sets


        svc = SVC()
        X2 = train_set.drop('Vote', axis=1)
        y2 = train_set['Vote']
        svc.fit(X2, y2)
        Xp2 = val_set.drop('Vote', axis=1)
        yp2 = val_set['Vote']
        v2 = svc.score(Xp2, yp2)

        print("SVC accuracy = " + str(v2))


        knn2 = KNeighborsClassifier(n_neighbors=5)
        X2 = train_set.drop('Vote', axis=1)
        y2 = train_set['Vote']
        knn2.fit(X2, y2)
        Xp2 = val_set.drop('Vote', axis=1)
        yp2 = val_set['Vote']
        v2 = knn2.score(Xp2, yp2)

        print("KNN accuracy = " + str(v2))

        clf = ExtraTreesClassifier(n_estimators=100)
        clf = clf.fit(X2, y2)
        Xp2 = val_set.drop('Vote', axis=1)
        yp2 = val_set['Vote']
        v2 = clf.score(Xp2, yp2)

        print("Extra Trees accuracy = " + str(v2))

        from sklearn import tree

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X2, y2)
        Xp2 = val_set.drop('Vote', axis=1)
        yp2 = val_set['Vote']
        v2 = clf.score(Xp2, yp2)

        print("Decision Tree accuracy = " + str(v2))

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
        clf.fit(X2, y2)
        Xp2 = val_set.drop('Vote', axis=1)
        yp2 = val_set['Vote']
        v2 = clf.score(Xp2, yp2)

        clf = RandomForestClassifier()
        clf.fit(X2, y2)
        v2 = clf.score(Xp2, yp2)

        print("Random Forest accuracy = " + str(v2))

        [train_set, val_set, test_set] = all_sets_orig

    selected_features = sfs_MI
    all_sets = [train_set, val_set, test_set]
    for index, data_set in enumerate(all_sets):
        all_sets[index] = all_sets[index][selected_features]
        [train_set, val_set, test_set] = all_sets


    save_datasets(train_set, val_set, test_set)
    export_features_to_csv(list(train_set))

    '''
    train_set_copy, val_set_copy, test_set_copy = set_clean(train_set_copy, val_set_copy, test_set_copy,
                                             verbose=True, graphic=True)
    train_set_copy, val_set_copy, test_set_copy = remove_inconsistency(train_set_copy, val_set_copy, test_set_copy)

    train_set_copy, val_set_copy, test_set_copy = data_transformation_without_one_hot(train_set_copy, val_set_copy, test_set_copy, False)

    selected_features = set_reduction(train_set_copy, val_set_copy, test_set_copy)

    selected_features.append('Vote')
    all_sets = [train_set_copy, val_set_copy, test_set_copy]
    for index, data_set in enumerate(all_sets):
        all_sets[index] = all_sets[index][selected_features]
        [train_set_copy, val_set_copy, test_set_copy] = all_sets

    svc = SVC()
    X2 = train_set_copy.drop('Vote', axis=1)
    y2 = train_set_copy['Vote']
    svc.fit(X2, y2)
    Xp2 = train_set_copy.drop('Vote', axis=1)
    yp2 = train_set_copy['Vote']
    v2 = svc.score(Xp2, yp2)

    svc = SVC()
    X2 = train_set_copy.drop('Vote', axis=1)
    y2 = train_set_copy['Vote']
    svc.fit(X2, y2)
    Xp2 = val_set_copy.drop('Vote', axis=1)
    yp2 = val_set_copy['Vote']
    v2 = svc.score(Xp2, yp2)

    print("SVC accuracy = " + v2)

    knn2 = KNeighborsClassifier(n_neighbors=5)
    X2 = train_set.drop('Vote', axis=1)
    y2 = train_set['Vote']
    knn2.fit(X2, y2)
    Xp2 = test_set.drop('Vote', axis=1)
    yp2 = test_set['Vote']
    v2 = knn2.score(Xp2, yp2)

    knn2 = KNeighborsClassifier(n_neighbors=5)
    X2 = train_set.drop('Vote', axis=1)
    y2 = train_set['Vote']
    knn2.fit(X2, y2)
    Xp2 = val_set.drop('Vote', axis=1)
    yp2 = val_set['Vote']
    v2 = knn2.score(Xp2, yp2)

    print("KNN accuracy = " + v2)

    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(X2, y2)
    Xp2 = test_set.drop('Vote', axis=1)
    yp2 = test_set['Vote']
    v2 = clf.score(Xp2, yp2)

    print("Extra Trees accuracy = " + v2)

    from sklearn import tree

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X2, y2)
    Xp2 = test_set.drop('Vote', axis=1)
    yp2 = test_set['Vote']
    v2 = clf.score(Xp2, yp2)

    print("Decision Tree accuracy = " + v2)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X2, y2)
    Xp2 = test_set.drop('Vote', axis=1)
    yp2 = test_set['Vote']
    v2 = clf.score(Xp2, yp2)

    clf = RandomForestClassifier()
    clf.fit(X2, y2)
    v2 = clf.score(Xp2, yp2)

    print("Random Forest accuracy = " + v2)


    print("main finished")
    '''


def load_and_split(input_path, backup_dir, train=0.7, validation=0.15, test=0.15):
    '''
    This function load .csv file, the original is not modified.
    Split the data to – train (50-75%), validation, (25-15%), test (25-10%)
    For each set – Keep a copy of the raw-data in backup path
    :param input_path: path to data file .csv
    :param backup dir: dir for backup of 3 sets.
    :param train: train ratio of dataset
    :param validation: validation ration of dataset
    :param test: test ratio of dataset
    :return: 3 pandas arrays (datasets): train, validation, test
    '''
    all_data = pd.read_csv(input_path)

    all_data_length = all_data.shape[0]

    print(all_data_length)

    train_and_val, test_set = train_test_split(all_data, test_size=test, stratify=all_data[['Vote']])
    train_set, val_set = \
        train_test_split(train_and_val, test_size=validation/(validation+train),
                         stratify=train_and_val[['Vote']])

    train_size = train_set.shape[0]
    val_size = val_set.shape[0]
    test_size = test_set.shape[0]
    assert all_data_length == train_size + val_size + test_size
    assert train_size / all_data_length == train
    assert val_size / all_data_length == validation
    assert test_size / all_data_length == test

    for tag in all_data['Vote'].unique():
        total_tag_num = len(all_data[(all_data.Vote == tag)])
        train_tag_num = len(train_set[(train_set.Vote == tag)])
        assert abs(total_tag_num * 0.7 - train_tag_num) < 1
        val_tag_num = len(val_set[(val_set.Vote == tag)])
        assert abs(total_tag_num * 0.15 - val_tag_num) < 1
        test_tag_num = len(test_set[(test_set.Vote == tag)])
        assert abs(total_tag_num * 0.15 - test_tag_num) < 1

    train_set.to_csv(os.path.join(backup_dir, 'train_backup.csv'))
    val_set.to_csv(os.path.join(backup_dir, 'val_backup.csv'))
    test_set.to_csv(os.path.join(backup_dir, 'test_backup.csv'))
    return train_set, val_set, test_set


def set_clean(train_set, val_set, test_set, verbose=True, graphic=False):
    """
    - Fill missing values
    - Smooth noisy data
    - identify\remove outliers.
    - remove inconsistency
    :param train: pandas dataframe train set
    :param val: pandas dataframe val set
    :param test: pandas dataframe test set
    :return: cleaned train, val, test
    """
    init_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])

    delete_vals_out_of_range(train_set, val_set, test_set, verbose=True)
    clipped_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert clipped_num_nans >= init_num_nans #equals if all values in ranges

    train_set, val_set, test_set, redundant_numeric_features, useful_numeric_features = \
        fill_nans_by_lin_regress(train_set, val_set, test_set)
    first_fill_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert clipped_num_nans > first_fill_num_nans


    delete_outliers(train_set, val_set, test_set, useful_numeric_features)
    no_outliers_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert no_outliers_num_nans > first_fill_num_nans

    train_set, val_set, test_set, redundant_numeric_features, useful_numeric_features = \
        fill_nans_by_lin_regress(train_set, val_set, test_set)
    sec_lin_reg_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert sec_lin_reg_num_nans < no_outliers_num_nans

    delete_vals_out_of_range(train_set, val_set, test_set, verbose=True)
    reclipped_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert reclipped_num_nans > sec_lin_reg_num_nans

    #added !!!!!!!!!!!!!!!!!!

    train_set, val_set, test_set, redundant_numeric_features, useful_numeric_features = \
        fill_nans_by_lin_regress(train_set, val_set, test_set)
    sec_lin_reg_num_nans2 = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert sec_lin_reg_num_nans2 < reclipped_num_nans

    delete_vals_out_of_range(train_set, val_set, test_set, verbose=True)
    reclipped_num_nans3 = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert reclipped_num_nans3 > sec_lin_reg_num_nans2

   #!!!!!!!!!!!!!!!!!!!!!!!!
    '''
    pre_drop_num_cols = len(train_set.columns)
    all_sets = [train_set, val_set, test_set]
    for index, data_set in enumerate(all_sets):
        all_sets[index] = data_set.drop(redundant_numeric_features, axis=1)
    [train_set, val_set, test_set] = all_sets
    post_drop_num_cols = len(train_set.columns)
    assert post_drop_num_cols < pre_drop_num_cols
    '''

    fill_missing_vals_by_mean(train_set, val_set, test_set, useful_numeric_features)
    num_features_full_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert num_features_full_num_nans < sec_lin_reg_num_nans
    assert sum([s[useful_numeric_features].isna().sum().sum() for s in (train_set, val_set, test_set)]) == 0

    #delete_rare_categorical_vals(train_set, val_set, test_set)  # Does nothing on our data set
    #fill_categorical_missing_vals(train_set, val_set, test_set)
    #assert sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)]) == 0

    return train_set, val_set, test_set


def smooth_noisy_data(train_set, val_set, test_set, verbose=True, graphic=False):
    '''
    This function handles negative values of
    parameters which can't be negative.
    '''
    if graphic:
        show_set_hist(train_set, title='train_set noisy data')
        show_set_hist(val_set, title='val_set noisy data')
        show_set_hist(test_set, title='test_set noisy data')

    init_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    delete_vals_out_of_range(train_set, val_set, test_set)
    clip_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert clip_num_nans > init_num_nans
    delete_outliers(train_set, val_set, test_set)
    outlier_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert outlier_num_nans >= clip_num_nans

    if graphic:
        show_set_hist(train_set, title='train_set noisy data')
        show_set_hist(val_set, title='val_set noisy data')
        show_set_hist(test_set, title='test_set noisy data')

    return train_set, val_set, test_set


def remove_inconsistency(train_set, val_set, test_set):
    '''
    Removes columns which have exact same features besides Vote, yet differ on the Vote
    '''
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
    columns = list(train_set)
    columns.remove('Vote')
    for data_set in (train_set, val_set, test_set):
        data_set.drop_duplicates(columns)
    return train_set, val_set, test_set


def data_transformation(train_set, val_set, test_set, graphic=False):
    """
    - Scaling
    - Normalization (Z-score or min-max)
    - Conversion
    :param train_set:
    :param val_set:
    :param test_set:
    :param how:
    :return:
    """

    if graphic:
        show_set_hist(train_set, title='train_set histogram before scaling')
    train_set, val_set, test_set = scale_sets(train_set, val_set, test_set)

    fill_categorical_missing_vals(train_set, val_set, test_set)
    assert sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)]) == 0



    if graphic:
        show_set_hist(train_set, title='train_set histogram after scaling')

    transform_categoric(train_set, val_set, test_set)

    return train_set, val_set, test_set

def data_transformation_without_one_hot(train_set, val_set, test_set, graphic=False):
    """
    - Scaling
    - Normalization (Z-score or min-max)
    - Conversion
    :param train_set:
    :param val_set:
    :param test_set:
    :param how:
    :return:
    """

    if graphic:
        show_set_hist(train_set, title='train_set histogram before scaling')
    train_set, val_set, test_set = scale_sets(train_set, val_set, test_set)

    fill_categorical_missing_vals(train_set, val_set, test_set)
    assert sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)]) == 0



    if graphic:
        show_set_hist(train_set, title='train_set histogram after scaling')

    #transform_categoric(train_set, val_set, test_set)
    for data_set in [train_set, val_set, test_set]:
        transform_label(data_set, "Vote")

    return train_set, val_set, test_set


def select_features(train_set, val_set, test_set):

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import GenericUnivariateSelect
    from sklearn.feature_selection import f_regression
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.feature_selection import f_classif
    from sklearn.feature_selection import mutual_info_classif
    # filter
    # ***********************************************************

    #features = set()
    features_ch = set()

    X_train = train_set.drop('Vote', axis=1)
    y_train = train_set['Vote']

    map = dict()
    for idx,axe in list(enumerate(X_train.columns)):
        map[idx] = axe
    '''
    X_new = SelectKBest(f_regression).fit(X_train, y_train)

    vals = X_new.pvalues_
    vals = list(enumerate(vals))
    vals = [tp for tp in vals if tp[1] <= 0.05]
    vals.sort(key=lambda tup: tup[1])
    indexes = [tp[0] for tp in vals][:10]

    selected_features1 = []
    for f in indexes:
        selected_features1.append(map[f])

    features_ch = features_ch.union(indexes)
    '''
    X_new = SelectKBest(mutual_info_regression).fit(X_train, y_train)

    indexes = X_new.get_support(indices=True)

    selected_features2 = []
    for f in indexes:
        selected_features2.append(map[f])

    features_ch = features_ch.union(indexes)

    selected_features2 = []
    for f in features_ch:
        selected_features2.append(map[f])
    '''
    X_new = SelectKBest(f_classif).fit(X_train, y_train)

    vals = X_new.pvalues_
    vals = list(enumerate(vals))
    vals = [tp for tp in vals if tp[1] <= 0.05]
    vals.sort(key=lambda tup: tup[1])
    indexes = [tp[0] for tp in vals]

    features = features.union(indexes[:10])
    '''
    X_new = SelectKBest(mutual_info_classif).fit(X_train, y_train)

    indexes = X_new.get_support(indices=True)

    selected_features2 = []
    for f in indexes:
        selected_features2.append(map[f])

    features_ch = features_ch.union(indexes)

    selected_features2 = []
    for f in features_ch:
        selected_features2.append(map[f])
    '''
    X_new = GenericUnivariateSelect(f_regression).fit(X_train, y_train)

    vals = X_new.pvalues_
    vals = list(enumerate(vals))
    vals = [tp for tp in vals if tp[1] <= 0.05]
    vals.sort(key=lambda tup: tup[1])
    indexes = [tp[0] for tp in vals]

    features = features.union(indexes[:5])

    X_new = GenericUnivariateSelect(mutual_info_regression).fit(X_train, y_train)

    vals = X_new.scores_
    vals = list(enumerate(vals))
    vals.sort(key=lambda tup: tup[1], reverse=True)
    indexes = [tp[0] for tp in vals]

    #features_ch = features_ch.union(indexes[:5])

    X_new = GenericUnivariateSelect(f_classif).fit(X_train, y_train)

    vals = X_new.pvalues_
    vals = list(enumerate(vals))
    vals = [tp for tp in vals if tp[1] <= 0.05]
    vals.sort(key=lambda tup: tup[1])
    indexes = [tp[0] for tp in vals]

    features = features.union(indexes[:5])

    X_new = GenericUnivariateSelect(mutual_info_classif).fit(X_train, y_train)

    vals = X_new.scores_
    vals = list(enumerate(vals))
    vals.sort(key=lambda tup: tup[1], reverse=True)
    indexes = [tp[0] for tp in vals]

    features = features.union(indexes[:5])
    
    lsvc = LinearSVC(C=0.002, penalty="l1", dual=False).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X_train)
    '''
    '''
    clf = RandomForestClassifier()
    clf = clf.fit(X_train, y_train)
    clf.feature_importances_

    vals = clf.feature_importances_
    vals = list(enumerate(vals))
    vals.sort(key=lambda tup: tup[1], reverse=True)
    indexes = [tp[0] for tp in vals]

    #features_ch = features_ch.union(indexes[:10])
    '''
    #relief wrapper
    '''
    X_val = val_set.drop('Vote', axis=1)
    y_val = val_set['Vote']

    selected_features_relief = relief(pd.concat([X_train, X_val]).copy(), pd.concat([y_train, y_val]).copy(),
                                      nominal_features,
                                      numerical_features, 1000, 1)

    print("Relief algorithm selected features are: {}".format(selected_features_relief))
    '''

    #wrapper
    #***********************************************************
    X_train = train_set.drop('Vote', axis=1)
    y_train = train_set['Vote']

    knn = KNeighborsClassifier(n_neighbors=5)

    sfs1 = SFS(estimator=knn,
               k_features=(5, 15),
               forward=True,
               floating=False,
               scoring='accuracy',
               cv=5)

    pipe = make_pipeline(StandardScaler(), sfs1)

    pipe.fit(X_train, y_train)

    fn5 = [int(x) for x in sfs1.k_feature_names_]

    selected_features_sfs_knn = []
    for f in fn5:
        selected_features_sfs_knn.append(map[f])


    dtc = SGDClassifier(random_state=92, max_iter=1000, tol=1e-3)

    sfs1 = SFS(estimator=dtc,
               k_features=(5, 15),
               forward=True,
               floating=False,
               scoring='accuracy',
               cv=5)

    pipe = make_pipeline(StandardScaler(), sfs1)

    pipe.fit(X_train, y_train)

    fn5 = [int(x) for x in sfs1.k_feature_names_]

    selected_features_sfs_svm = []
    for f in fn5:
        selected_features_sfs_svm.append(map[f])


    X_test = test_set.drop('Vote', axis=1)
    y_test = test_set['Vote']
    X_val = val_set.drop('Vote', axis=1)
    y_val = val_set['Vote']
    selected_features_svm, selected_features_knn = run_sfs_base_clfs(X_train, y_train, X_val, y_val, X_test, y_test)
    print("for SVM SFS selected features are: {}".format(selected_features_svm))
    print("for KNN SFS selected features are: {}".format(selected_features_knn))

    selected_features = []
    for f in features_ch:
        selected_features.append(map[f])

    return selected_features_knn, selected_features_svm,\
           selected_features_sfs_knn, selected_features_sfs_svm,\
           selected_features2


def select_features(train_set, val_set, test_set):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import GenericUnivariateSelect
    from sklearn.feature_selection import f_regression
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.feature_selection import f_classif
    from sklearn.feature_selection import mutual_info_classif
    # filter
    # ***********************************************************

    # features = set()
    features_ch = set()

    X_train = train_set.drop('Vote', axis=1)
    y_train = train_set['Vote']

    map = dict()
    for idx, axe in list(enumerate(X_train.columns)):
        map[idx] = axe
    '''
    X_new = SelectKBest(f_regression).fit(X_train, y_train)

    vals = X_new.pvalues_
    vals = list(enumerate(vals))
    vals = [tp for tp in vals if tp[1] <= 0.05]
    vals.sort(key=lambda tup: tup[1])
    indexes = [tp[0] for tp in vals][:10]

    selected_features1 = []
    for f in indexes:
        selected_features1.append(map[f])

    features_ch = features_ch.union(indexes)
    '''
    X_new = SelectKBest(mutual_info_regression).fit(X_train, y_train)

    indexes = X_new.get_support(indices=True)

    selected_features2 = []
    for f in indexes:
        selected_features2.append(map[f])

    features_ch = features_ch.union(indexes)

    selected_features2 = []
    for f in features_ch:
        selected_features2.append(map[f])
    '''
    X_new = SelectKBest(f_classif).fit(X_train, y_train)

    vals = X_new.pvalues_
    vals = list(enumerate(vals))
    vals = [tp for tp in vals if tp[1] <= 0.05]
    vals.sort(key=lambda tup: tup[1])
    indexes = [tp[0] for tp in vals]

    features = features.union(indexes[:10])
    '''
    X_new = SelectKBest(mutual_info_classif).fit(X_train, y_train)

    indexes = X_new.get_support(indices=True)

    selected_features2 = []
    for f in indexes:
        selected_features2.append(map[f])

    features_ch = features_ch.union(indexes)

    selected_features2 = []
    for f in features_ch:
        selected_features2.append(map[f])
    '''
    X_new = GenericUnivariateSelect(f_regression).fit(X_train, y_train)

    vals = X_new.pvalues_
    vals = list(enumerate(vals))
    vals = [tp for tp in vals if tp[1] <= 0.05]
    vals.sort(key=lambda tup: tup[1])
    indexes = [tp[0] for tp in vals]

    features = features.union(indexes[:5])

    X_new = GenericUnivariateSelect(mutual_info_regression).fit(X_train, y_train)

    vals = X_new.scores_
    vals = list(enumerate(vals))
    vals.sort(key=lambda tup: tup[1], reverse=True)
    indexes = [tp[0] for tp in vals]

    #features_ch = features_ch.union(indexes[:5])

    X_new = GenericUnivariateSelect(f_classif).fit(X_train, y_train)

    vals = X_new.pvalues_
    vals = list(enumerate(vals))
    vals = [tp for tp in vals if tp[1] <= 0.05]
    vals.sort(key=lambda tup: tup[1])
    indexes = [tp[0] for tp in vals]

    features = features.union(indexes[:5])

    X_new = GenericUnivariateSelect(mutual_info_classif).fit(X_train, y_train)

    vals = X_new.scores_
    vals = list(enumerate(vals))
    vals.sort(key=lambda tup: tup[1], reverse=True)
    indexes = [tp[0] for tp in vals]

    features = features.union(indexes[:5])

    lsvc = LinearSVC(C=0.002, penalty="l1", dual=False).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X_train)
    '''
    '''
    clf = RandomForestClassifier()
    clf = clf.fit(X_train, y_train)
    clf.feature_importances_

    vals = clf.feature_importances_
    vals = list(enumerate(vals))
    vals.sort(key=lambda tup: tup[1], reverse=True)
    indexes = [tp[0] for tp in vals]

    #features_ch = features_ch.union(indexes[:10])
    '''
    # relief wrapper
    '''
    X_val = val_set.drop('Vote', axis=1)
    y_val = val_set['Vote']

    selected_features_relief = relief(pd.concat([X_train, X_val]).copy(), pd.concat([y_train, y_val]).copy(),
                                      nominal_features,
                                      numerical_features, 1000, 1)

    print("Relief algorithm selected features are: {}".format(selected_features_relief))
    '''

    # wrapper
    # ***********************************************************
    X_train = train_set.drop('Vote', axis=1)
    y_train = train_set['Vote']

    knn = KNeighborsClassifier(n_neighbors=5)

    sfs1 = SFS(estimator=knn,
               k_features=(5, 15),
               forward=True,
               floating=False,
               scoring='accuracy',
               cv=5)

    pipe = make_pipeline(StandardScaler(), sfs1)

    pipe.fit(X_train, y_train)

    fn5 = [int(x) for x in sfs1.k_feature_names_]

    selected_features_sfs_knn = []
    for f in fn5:
        selected_features_sfs_knn.append(map[f])

    dtc = SGDClassifier(random_state=92, max_iter=1000, tol=1e-3)

    sfs1 = SFS(estimator=dtc,
               k_features=(5, 15),
               forward=True,
               floating=False,
               scoring='accuracy',
               cv=5)

    pipe = make_pipeline(StandardScaler(), sfs1)

    pipe.fit(X_train, y_train)

    fn5 = [int(x) for x in sfs1.k_feature_names_]

    selected_features_sfs_svm = []
    for f in fn5:
        selected_features_sfs_svm.append(map[f])

    X_test = test_set.drop('Vote', axis=1)
    y_test = test_set['Vote']
    X_val = val_set.drop('Vote', axis=1)
    y_val = val_set['Vote']
    selected_features_svm, selected_features_knn = run_sfs_base_clfs(X_train, y_train, X_val, y_val, X_test, y_test)
    print("for SVM SFS selected features are: {}".format(selected_features_svm))
    print("for KNN SFS selected features are: {}".format(selected_features_knn))

    selected_features = []
    for f in features_ch:
        selected_features.append(map[f])

    return selected_features_knn, selected_features_svm, \
           selected_features_sfs_knn, selected_features_sfs_svm, \
           selected_features2


def set_reduction(train_set, val_set, test_set):
    '''
    - implement Feature selection:
    - One filter method
    - One wrapper method
    :param train_set:
    :param val_set:
    :param test_set:
    :param how:
    :return: reduced set
    '''
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
    selected_features_my_sfs, selected_features_my_sgd,  selected_features_sfs, selected_features_sgd, selected_features_MI = select_features(train_set, val_set, test_set)
    return selected_features_my_sfs,selected_features_my_sgd,selected_features_sfs,  selected_features_sgd, selected_features_MI


def save_datasets(train_set, val_set, test_set):
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
    train_set.to_csv('train_transformed.csv')
    val_set.to_csv('validation_transformed.csv')
    test_set.to_csv('test_transformed.csv')


def export_features_to_csv(features):
    used_features = set()
    for f in features:
        if f == 'Vote':
            continue
        match = re.search('^Is_(.+)__.+$', f)
        if match is not None:
            used_features.add(match.group(1))
        else:
            used_features.add(f)
    used_features = list(used_features)
    used_features.sort()
    txt = ','.join(used_features)
    with open('SelectedFeatures.csv', 'w') as f:
        f.write(txt)


if __name__ == "__main__":
    from enum import Enum


    class AtrrType(Enum):
        BINARY = 1
        DISCRETE = 2
        CONTINUOUS = 3
        CATEGORICAL = 4


    elections_data = pd.read_csv('./' + 'ElectionsData.csv')
    df_types = elections_data.dtypes
    # print(df_types)
    # print(type(df_types))
    attrs = df_types.keys()

    attrs_type = {
        attrs[0]: AtrrType.CATEGORICAL,
        attrs[1]: AtrrType.BINARY,
        attrs[2]: AtrrType.CONTINUOUS,
        attrs[3]: AtrrType.CONTINUOUS,
        attrs[4]: AtrrType.CATEGORICAL,
        attrs[5]: AtrrType.CONTINUOUS,
        attrs[6]: AtrrType.BINARY,
        attrs[7]: AtrrType.CONTINUOUS,
        attrs[8]: AtrrType.BINARY,
        attrs[9]: AtrrType.BINARY,
        attrs[10]: AtrrType.BINARY,
        attrs[11]: AtrrType.CONTINUOUS,
        attrs[12]: AtrrType.CONTINUOUS,
        attrs[13]: AtrrType.CONTINUOUS,
        attrs[14]: AtrrType.CONTINUOUS,
        attrs[15]: AtrrType.CONTINUOUS,
        attrs[16]: AtrrType.CONTINUOUS,
        attrs[17]: AtrrType.CONTINUOUS,
        attrs[18]: AtrrType.CONTINUOUS,
        attrs[19]: AtrrType.CONTINUOUS,
        attrs[20]: AtrrType.CONTINUOUS,
        attrs[21]: AtrrType.CONTINUOUS,
        attrs[22]: AtrrType.CATEGORICAL,
        attrs[23]: AtrrType.CONTINUOUS,
        attrs[24]: AtrrType.CONTINUOUS,
        attrs[25]: AtrrType.CONTINUOUS,
        attrs[26]: AtrrType.CONTINUOUS,
        attrs[27]: AtrrType.CONTINUOUS,
        attrs[28]: AtrrType.DISCRETE,
        attrs[29]: AtrrType.CATEGORICAL,
        attrs[30]: AtrrType.DISCRETE,
        attrs[31]: AtrrType.CONTINUOUS,
        attrs[32]: AtrrType.DISCRETE,
        attrs[33]: AtrrType.CATEGORICAL,
        attrs[34]: AtrrType.CATEGORICAL,
        attrs[35]: AtrrType.BINARY,
        attrs[36]: AtrrType.DISCRETE,
        attrs[37]: AtrrType.CONTINUOUS,
    }
    print("features analysis:")
    print(attrs_type)  # TODO: to check correctnes
    print()

    main()