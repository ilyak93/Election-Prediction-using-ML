from os import path

import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
import operator
from collections import OrderedDict
import matplotlib.pyplot as plt

PATH = path.dirname(path.realpath(__file__)) + "/"
DATA_PATH = PATH + "ElectionsData.csv"
TRAIN_PATH = PATH + "train_transformed.csv"
VALIDATION_PATH = PATH + "validation_transformed.csv"
TEST_PATH = PATH + "test_transformed.csv"

global_transportation_threshold = 0.59
label = 'Vote'

# lists
selected_features = ["Yearly_IncomeK", "Number_of_differnt_parties_voted_for",
                   "Political_interest_Total_Score",
                   "Avg_Satisfaction_with_previous_vote",
                   "Avg_monthly_income_all_years", "Most_Important_Issue",
                   "Overall_happiness_score", "Avg_size_per_room", "Weighted_education_rank",
                   "Vote"]

selected_features_without_label = ["Yearly_IncomeK", "Number_of_differnt_parties_voted_for",
                   "Political_interest_Total_Score",
                   "Avg_Satisfaction_with_previous_vote",
                   "Avg_monthly_income_all_years", "Most_Important_Issue",
                   "Overall_happiness_score", "Avg_size_per_room", "Weighted_education_rank"]

selected_nominal_features = ['Most_Important_Issue']

selected_numerical_features = ["Yearly_IncomeK", "Number_of_differnt_parties_voted_for",
                           "Political_interest_Total_Score",
                           "Avg_Satisfaction_with_previous_vote",
                           "Avg_monthly_income_all_years",
                           "Overall_happiness_score", "Avg_size_per_room",
                           "Weighted_education_rank"]


def get_vote_dict():
    '''
    :return: dictionary which maps 'Vote' category to numbers.
    '''
    original_train = pd.read_csv(PATH+'train_backup.csv')
    party_names = original_train['Vote'].values
    train_set = pd.read_csv(PATH+'train_transformed.csv')
    party_nums = train_set.pop('Vote').values

    num_list = []
    name_list = []
    for num, name in zip(party_nums, party_names):
        if num not in num_list:
            num_list.append(num)
            name_list.append(name)
        else:
            idx = num_list.index(num)
            assert name_list[idx] == name
    vote_dict = dict(zip(num_list, name_list))
    return vote_dict

aux = get_vote_dict()
label2num = {aux[x]:x for x in aux}

num2label = aux


def load_data(filepath: str) -> DataFrame:
    df = read_csv(filepath, header=0)
    return df


def load_prepared_dataFrames():
    df_train, df_valid, df_test = load_data(TRAIN_PATH), load_data(VALIDATION_PATH), load_data(TEST_PATH)
    return df_train.drop('Unnamed: 0', axis=1), \
           df_valid.drop('Unnamed: 0', axis=1), \
           df_test.drop('Unnamed: 0', axis=1)


def filter_possible_coalitions(possible_coalitions: dict):
    """
    :param possible_coalitions: all possible coalition
    :return: possible coalition without duplication
    """
    # remove duplicates
    filtered_possible_coalitions = dict()
    for _coalition_name, _coalition_list in possible_coalitions.items():
        _coalition_list.sort()
        if _coalition_list not in filtered_possible_coalitions.values():
            filtered_possible_coalitions[_coalition_name] = _coalition_list
    return filtered_possible_coalitions


def to_binary_class(data, value):
    """
    :param data: regular data
    :param value: the value to be assigned as 1
    :return: binary classified data
    """
    binary_data = data.copy()
    bool_labels = binary_data[label] == value
    binary_data[label] = bool_labels
    return binary_data


def get_sorted_vote_division(y):
    vote_results = dict()
    for label_name, label_index in label2num.items():
        percent_of_voters = sum(list(y == label_index)) / len(y)
        vote_results[label_index] = percent_of_voters
    return OrderedDict(sorted(vote_results.items(), key=operator.itemgetter(1)))


def divide_data(df: DataFrame, data_class=label):
    x_df = df.loc[:, df.columns != data_class]
    y_df = df[data_class]
    return x_df, y_df


def export_to_csv(filespath: str, x_train: DataFrame, x_val: DataFrame,
                  x_test: DataFrame, y_train: DataFrame, y_val: DataFrame,
                  y_test: DataFrame, prefix: str):
    x_train = x_train.assign(Vote=y_train.values)
    x_val = x_val.assign(Vote=y_val.values)
    x_test = x_test.assign(Vote=y_test.values)
    x_train.to_csv(filespath + "{}_train.csv".format(prefix), index=False)
    x_val.to_csv(filespath + "{}_val.csv".format(prefix), index=False)
    x_test.to_csv(filespath + "{}_test.csv".format(prefix), index=False)


def score(x_train: DataFrame, y_train: DataFrame, clf, k: int):
    return cross_val_score(clf, x_train, y_train, cv=k, scoring='accuracy').mean()



def export_to_csv(filepath: str, df: DataFrame):
    df.to_csv(filepath, index=False)

def winner_color(clf, x_test: DataFrame):
    #y_test_proba: np.ndarray = np.average(clf.predict_proba(x_test), axis=0)
    predictions = clf.predict(x_test)
    pred_hist = [0] * 13
    for pred in predictions:
        pred_hist[pred] += 1
    pred_winner = np.argmax(pred_hist)
    print(f"The predicted party to win the elections is {num2label[pred_winner]}")
    plt.plot(pred_hist)  # arguments are passed to np.histogram
    plt.title("Test Vote Probabilities")
    plt.show()