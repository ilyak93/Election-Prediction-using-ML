import os
import pandas as pd
from sklearn.model_selection import train_test_split
import inspect
import re

from tensorflow.keras.utils import to_categorical

from model import MLP_ensemble


class dataProvider():
    def __init__(self, input_path=''):
        self.train_size = None
        self.val_size = None
        self.test_size = None

        if input_path == '':
            delimiter = ''
        else:
            delimiter = '\\'

        train_path = input_path+delimiter+'train_transformed.csv'
        val_path = input_path + delimiter + 'validation_transformed.csv'
        test_path = input_path + delimiter + 'test_transformed.csv'
        self.train_set = pd.concat([pd.read_csv(train_path),pd.read_csv(val_path)])

        self.val_set = pd.read_csv(test_path)

        self.test_id, self.test_set = self.transform_test_set(input_path + delimiter + 'unlabeled_set.csv')

        unlabeled_test_path = input_path + delimiter + 'unlabeled_set.csv'

        self.unlabeled_test = pd.read_csv(unlabeled_test_path)

        # prepare dict:
        party_names = self.train_set['Party'].values
        party_nums = self.train_set['Vote'].values

        num_list = []
        name_list = []
        for num, name in zip(party_nums, party_names):
            if num not in num_list:
                num_list.append(num)
                name_list.append(name)
            else:
                idx = num_list.index(num)
                assert name_list[idx] == name

        self.vote_dictionary = dict(zip(num_list, name_list))

        # get rid of parties names:
        self.train_set.pop('Party')
        self.val_set.pop('Party')

        # preparing dataset to model:
        self.y_train = self.train_set.pop('Vote').values
        self.y_val = self.val_set.pop('Vote').values

        # sort columns lexicorgaphically
        self.train_set = self.train_set.reindex(sorted(self.train_set.columns), axis=1)
        self.val_set = self.val_set.reindex(sorted(self.val_set.columns), axis=1)

        # Ensure that everything is sorted:
        for a, b, c in zip(self.train_set.columns,
                           self.val_set.columns,
                           self.test_set.columns):
            assert a == b
            assert b == c
            assert c == a

        self.x_train = self.train_set.values
        self.x_val = self.val_set.values
        self.x_test = self.test_set.values

        #self.test_set_indices = self.test_set.index.values
        self.feature_names = self.train_set.columns

    def get_test_id(self):
        return self.test_id

    def transform_test_set(self, path):
        test_set = pd.read_csv(path)
        #print(test_set)
        test_id = test_set.pop('IdentityCard_Num')
        test_set = test_set.reindex(sorted(test_set.columns), axis=1)
        return test_id, test_set

    def test_for_nans(self):
        assert sum([s.isna().sum().sum() for s in (self.train_set, self.val_set, self.test_set)]) == 0

    def get_vote_dict(self):
        '''
        :return: dictionary which maps 'Vote' category to numbers.
        '''
        return self.vote_dictionary

    def get_sets_as_pd(self):
        return self.train_set, self.val_set, self.test_set

    def get_train_xy(self, onehot_y=False):
        if not onehot_y:
            return self.x_train, self.y_train
        else:
            dummy_y = to_categorical(self.y_train)
            return self.x_train, dummy_y

    def get_val_xy(self, onehot_y=False):
        if not onehot_y:
            return self.x_val, self.y_val
        else:
            dummy_y = to_categorical(self.y_val)
            return self.x_val, dummy_y

    def get_test_xy(self):
        return self.x_test


    def get_feature_names(self):
        return self.test_set.columns[1:]
