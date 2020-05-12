import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random

from sklearn.ensemble import RandomForestClassifier

from transform import *
from sklearn import linear_model
from graphic_utils import *
import math


LINEAR_FILL_CORR_THRESHOLD = 0.8
CAT_RARITY_THRESHOLD = 0.01
STD_DIFF = 5

plt.rcParams.update({'font.size': 5})


def get_num_feature_list(train_set):
    '''
    :param train_set: A Pandas dataframe
    :return: list containing names of all numeric features. Every non-numeric feature is categorical
    '''
    assert isinstance(train_set, pd.DataFrame)
    ret = train_set.select_dtypes(exclude=['object']).columns
    assert 'Vote' not in ret
    return ret


def analyze_NaNs(train_set, val_set, test_set, verbose=True):
    if verbose:
        train_miss = train_set.isnull().sum().sum()
        val_miss = val_set.isnull().sum().sum()
        test_miss = test_set.isnull().sum().sum()
        train_bad_samples = (train_set.shape[0] - train_set.dropna().shape[0])*100/train_set.shape[0]
        val_bad_samples = (val_set.shape[0] - val_set.dropna().shape[0])*100/val_set.shape[0]
        test_bad_samples = (test_set.shape[0] - test_set.dropna().shape[0])*100/test_set.shape[0]

        print("========== Dataset Analysis ==========")
        print("dataset sizes:", train_set.shape[0], val_set.shape[0], test_set.shape[0])
        print("train_set has in total", train_miss,
              "missing values, in ~", np.round(train_bad_samples), "% of samples")
        print("val_set has in total", val_miss,
              "missing values, in ~", np.round(val_bad_samples), "% of samples")
        print("test_set has in total", test_miss,
              "missing values, in ~", np.round(test_bad_samples), "% of samples")


def delete_missing_values(train_set, val_set, test_set, verbose=True):
    # This option will lead to loss of 45% of samples!!!
    for data_set in (train_set, val_set, test_set):
        data_set.dropna()
    return train_set, val_set, test_set


def fill_missing_most_common(train_set, val_set, test_set):
    """
    This function fill missing values with
    most common value in a CATEGORICAL column
    """
    for col in train_set.columns:
        # Check if categorical
        if not np.issubdtype(train_set[col].dtype, np.number):
            train_set[col].fillna(train_set[col].mode().iloc[0], inplace=True)

    for col in val_set.columns:
        # Check if categorical
        if not np.issubdtype(val_set[col].dtype, np.number):
            val_set[col].fillna(train_set[col].mode().iloc[0], inplace=True)

    for col in test_set.columns:
        # Check if categorical
        if not np.issubdtype(test_set[col].dtype, np.number):
            test_set[col].fillna(test_set[col].mode().iloc[0], inplace=True)

    return train_set, val_set, test_set


def fill_missing_mean(train_set, val_set, test_set):
    """
    This function fill missing values with
    most common value in a NUMERICAL column
    """
    for col in train_set.columns:
        # Check if numerical
        if np.issubdtype(train_set[col].dtype, np.number):
            train_set[col].fillna(train_set[col].mean(), inplace=True)

    for col in val_set.columns:
        # Check if numerical
        if np.issubdtype(val_set[col].dtype, np.number):
            val_set[col].fillna(train_set[col].mean(), inplace=True)

    for col in test_set.columns:
        # Check if numerical
        if np.issubdtype(test_set[col].dtype, np.number):
            test_set[col].fillna(test_set[col].mean(), inplace=True)

    return train_set, val_set, test_set


def get_correlated_feature_groups(corr_matrix, threshold=0.93, verbose=False):
    '''
    Gets a correlation matrix and returns all groups of features that have higher
    absolute correlation than threshold
    :param corr_matrix:
    :param threshold:
    :param verbose:
    :return: A list of lists, each list represents a group of features that are correlated
    '''
    corr_matrix_cpy = corr_matrix.copy()
    corr_matrix_cpy.loc[:, :] = np.tril(corr_matrix_cpy, k=-1)

    already_in = set()
    corr_features_groups = []
    for col in corr_matrix_cpy:
        perfect_corr = corr_matrix_cpy[col][corr_matrix_cpy[col] > threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            corr_features_groups.append(perfect_corr)

    for corr_group in corr_features_groups:  # Check that the relation is transitive
        for x, y in list(itertools.combinations(corr_group, 2)):
            assert corr_matrix[x][y] > threshold
    if verbose:
        print("correlated groups are :")
        for corr_group in corr_features_groups:
            print(corr_group, "\n")
    return corr_features_groups


def impute_by_lin_model(model, data, X, Y):
    '''
    :param model: linear regression model of X and Y
    :param data: The DataFrame we fill missing values in
    :param X: The reference label
    :param Y: The label we fill missing values in
    :return: data, imputed
    '''
    assert isinstance(model, linear_model.LinearRegression)
    assert isinstance(data, pd.DataFrame)
    num_nans_start = data[Y].isna().sum()
    for index, row in data[data[Y].isnull()].iterrows():
        if not np.isnan(data[X][index]):
            data.at[index, Y] = model.predict([[data[X][index]]])[0][0]
    print("Filled ", num_nans_start - data[Y].isna().sum(), "nans of feature", Y, "from ", X)
    return data


def __fill_missing_linear_regression(train, validation, test, features, corr_mat):
    '''
    Fill missing values in all 3 sets by a linear regression to the most correlated other feature,
    in absolute value. Will never fill missing values if the correlation between features is lower
    than LINEAR_FILL_CORR_THRESHOLD
    :param train: train set
    :param validation: validation set
    :param test: test set
    :param features: list of feature names to fill, must be numeric features
    :param corr_mat: correlation matrix between all features
    :return:
    '''
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert isinstance(validation, pd.DataFrame)
    assert isinstance(features, list)

    sum_nas_before = sum([s[features].isna().sum().sum() for s in (train, validation, test)])

    all_sets = [train, validation, test]
    for feature in features:
        corr_tuples = [(corr_mat[feature][col], col) for col in corr_mat.columns if col != feature]
        corr_tuples.sort(key=lambda x: x[0], reverse=True)
        while corr_tuples[0][0] >= LINEAR_FILL_CORR_THRESHOLD and \
                sum([s[feature].isna().sum() for s in (train, validation, test)]) > 0:
            reference_feature = corr_tuples[0][1]

            feature_duo_train = train[[reference_feature, feature]].copy()
            feature_duo_val = validation[[reference_feature, feature]].copy()
            feature_duo = pd.concat([feature_duo_train, feature_duo_val])
            feature_duo = feature_duo.dropna(how='any')

            lin_model = linear_model.LinearRegression()
            model = lin_model.fit(feature_duo[reference_feature].values.reshape(-1, 1),
                                  feature_duo[feature].values.reshape(-1, 1))

            for index, data_set in enumerate(all_sets):
                all_sets[index] = impute_by_lin_model(model, data_set, reference_feature, feature)

            corr_tuples.pop(0)

    [train, validation, test] = all_sets
    sum_nas_after = sum([s[features].isna().sum().sum() for s in (train, validation, test)])
    assert sum_nas_after < sum_nas_before

    print('we filled', (1-float(sum_nas_after)/sum_nas_before)*100, '% of nas in numerical features',
          'of all three sets')

    return train, validation, test


def fill_missing_vals_by_mean(train, val, test, features):
    '''
    Fills three data sets' all missing values in given features by the mean value of that feature
    Used as a last resort
    :param train: train data set
    :param val: validation data set
    :param test: test data set
    :param features: A list or set of feature names to fill. Features MUST be numeric
    :return:
    '''
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert isinstance(features, (list, set))

    for f in features:
        # compute mean
        train_and_val = pd.concat([train, val])
        mean = train_and_val[f].mean()

        for data_set in (train, val, test):
            data_set[f].fillna(mean, inplace=True)
            assert data_set[f].isna().sum() == 0
    return train, val, test


def fill_missing_vals_exp1(train_set, val_set, test_set, verbose=True, graphic=False):
    '''
    Show results of very basic experiments, filling missing values with the mean or most common value
    :param train_set:
    :param val_set:
    :param test_set:
    :param verbose:
    :return:
    '''
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)

    exp_train = train_set.copy()
    exp_val = val_set.copy()
    exp_test = test_set.copy()
    # Show % of corrupted samples.
    analyze_NaNs(exp_train, exp_val, exp_test, verbose)

    # Show histograms
    if graphic:
        show_set_hist(exp_train, title='train_set before removing NaNs')

    # Fill categorical data with most common
    exp_train, exp_val, exp_test = fill_missing_most_common(exp_train, exp_val, exp_test)

    # Show % of corrupted samples.
    analyze_NaNs(exp_train, exp_val, exp_test, verbose)

    # Fill numerical data with mean
    exp_train, exp_val, test_set = fill_missing_mean(exp_train, exp_val, exp_test)

    # Show % of corrupted samples.
    analyze_NaNs(exp_train, exp_val, exp_test, verbose)

    if graphic:
        show_set_hist(exp_train, title='train_set after removing NaNs')

    assert exp_train.isnull().sum().sum()
    assert exp_val.isnull().sum().sum() == 0
    assert exp_test.isnull().sum().sum() == 0


def fill_nans_by_lin_regress(train_set, val_set, test_set, verbose=True, graphic=False, all_history=False):
    '''
    Fills all numeric missing values in all three sets, first by correlated features then the rest
    are just filled by the median value
    :param graphic: Whether to show graphs
    :param all_history: Running entire history of experimentations
    :return:
    '''
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)

    train_and_val = pd.concat([train_set, val_set])

    numeric_features = train_and_val.select_dtypes(exclude=['object']).columns
    corr_matrix = train_and_val[numeric_features].corr()  # ignores string columns

    if all_history:
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(numeric_features), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(numeric_features)
        ax.set_yticklabels(numeric_features)
        plt.show()
    corr_matrix = abs(corr_matrix)

    if verbose:
        corr_indices = np.where(corr_matrix > 0.95)
        corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*corr_indices)
                      if x != y and x < y]

        print("correlated feature pairs are:\n")
        for pair in corr_pairs:
            print(pair, "\n")

    # failed experiment
    # if graphic:
    #     from pandas.plotting import scatter_matrix
    #     scatter_matrix(train_set)
    #     plt.show()

    corr_feature_groups = get_correlated_feature_groups(corr_matrix)
    if verbose:
        print("Correlated feature groups are: ", corr_feature_groups)
    redundant_features = []
    for corr_group in corr_feature_groups:
        for feature in corr_group[1:]:
            redundant_features.append(feature)
    if verbose:
        print("redundant features are: ", redundant_features)
    #useful_features = set(numeric_features).difference(set(redundant_features)) #TODO: last try
    useful_features = set(numeric_features)
    useful_features = list(useful_features)
    useful_features.sort()
    train_set, val_set, test_set = \
        __fill_missing_linear_regression(train_set, val_set, test_set, useful_features, corr_matrix)

    return train_set, val_set, test_set, redundant_features, useful_features


def __delete_vals_out_of_range(data_set, feature, min_val=-math.inf, max_val=math.inf):
    '''
    Marks all values of a given feature in a give data frame that are below min_val or
    above max_val as nans
    :param data_set: Pandas DataFrame, includes column feature
    '''
    assert isinstance(data_set, pd.DataFrame)
    assert max_val >= min_val
    count = -data_set[feature].isna().sum()
    for index, row in data_set[~data_set[feature].between(min_val, max_val)].iterrows():
        data_set.ix[index, feature] = np.nan
        count += 1
    if count > 0:
        print("removed", count, "vals out of range for feature", feature)


def delete_vals_out_of_range(train_set, val_set, test_set, verbose=True):
    '''
    Delete all values in all data sets that are out of range
    Deleted values will be marked as nans
    No need to give features as an argument - they are hard coded and classified here
    :return:
    '''
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)

    if verbose:
        start_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])

    non_negative_features = ['Avg_lottary_expanses',
                             'Avg_Residancy_Altitude',
                             'Avg_Satisfaction_with_previous_vote',
                             'Avg_education_importance',
                             'Avg_monthly_expense_on_pets_or_plants',
                             'Avg_monthly_household_cost',
                             'Avg_monthly_income_all_years',
                             'Avg_monthly_expense_when_under_age_21',
                             'Avg_environmental_importance',
                             'Political_interest_Total_Score',
                             'Avg_size_per_room',
                             'Garden_sqr_meter_per_person_in_residancy_area',
                             'Num_of_kids_born_last_10_years',
                             'Number_of_differnt_parties_voted_for',
                             'Weighted_education_rank',
                             'Yearly_ExpensesK',
                             'Yearly_IncomeK',
                             ]
    percentage_features = ['%Time_invested_in_work',
                           '%_satisfaction_financial_policy',
                           'Last_school_grades']
    zero_to_ten_scale_features = ['Occupation_Satisfaction']
    zero_to_one_scale_features = ['%Of_Household_Income',
                                  'Financial_balance_score_(0-1)']
    zero_to_120_scale_features = ['Number_of_valued_Kneset_members']
    for data_set in (train_set, test_set, val_set):
        for f in non_negative_features:
            __delete_vals_out_of_range(data_set, f, min_val=0)
        for f in zero_to_120_scale_features:
            __delete_vals_out_of_range(data_set, f, min_val=0, max_val=120)
        for f in percentage_features:
            __delete_vals_out_of_range(data_set, f, min_val=0, max_val=100)
        for f in zero_to_ten_scale_features:
            __delete_vals_out_of_range(data_set, f, min_val=0, max_val=10)
        for f in zero_to_one_scale_features:
            __delete_vals_out_of_range(data_set, f, min_val=0, max_val=1)

    if verbose:
        num_nans_after_clip = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
        num_vals_in_frame = 10000 * len(train_set.columns)
        percentage_dropped_by_clipping = \
            (float(num_nans_after_clip - start_num_nans) * 100) / num_vals_in_frame
        print("Clipping dropped:", str(percentage_dropped_by_clipping) + '%',
              'from all data sets combined')


def delete_outliers(train_set, val_set, test_set, features, verbose=True):
    '''
    Deletes all values of features that are STD_THRESHOLD number of standard deviations above mean
    '''
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)

    if verbose:
        start_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])

    train_and_val = pd.concat([train_set, val_set])

    for f in features:
        # find the mean and the std

        std = train_and_val[f].std()
        mean = train_and_val[f].mean()

        for data_set in (train_set, val_set, test_set):
            delta = STD_DIFF * std
            for index, row in data_set[~data_set[f].between(mean - delta, mean + delta)].iterrows():
                data_set.ix[index, f] = np.nan

    if verbose:
        final_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
        percentage_dropped_by_std_dropping = \
            (float(final_num_nans - start_num_nans) * 100) / (10000 * len(train_set.columns))
        print("Outliers dropped:", str(percentage_dropped_by_std_dropping) + '%', 'of all data sets combined')




def delete_rare_categorical_vals(train_set, val_set, test_set):
    '''
    Looks for categorical features that have categories appearing less often than CAT_RARITY_THRESHOLD
    as a percentage across all label of that feature across all three sets.
    NOTE: we did not find such labels. This code will not delete the labels if it finds any,
    just prints them
    '''
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)

    categoric_features = train_set.select_dtypes(include=['object']).columns

    for f in categoric_features:
        value_counts = train_set[f].value_counts()
        value_counts.add(val_set[f].value_counts(), fill_value=0)
        value_counts.add(test_set[f].value_counts(), fill_value=0)
        total_value_count = value_counts.sum()
        for label, count in value_counts.iteritems():
            if float(count)/total_value_count <= CAT_RARITY_THRESHOLD:
                print('Found a categorical rarity at feature', f, 'label', label)



'''
def fill_categorical_missing_vals(train, val, test):

    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    categoric_features = train.select_dtypes(include=['object']).columns
    for f in categoric_features:
        value_counts = train[f].value_counts()
        total_value_count = value_counts.sum()

        for data_set in (train, val, test):
            for index, row in data_set[data_set[f].isnull()].iterrows():
                sample_index = random.randint(1, total_value_count)
                for label, count in value_counts.iteritems():
                    if sample_index <= count:
                        data_set.ix[index, f] = label
                        break
                    sample_index -= count
                assert data_set[f][index] != np.nan

    for data_set in (train, test, val):
        assert data_set[[f for f in categoric_features]].isna().sum().sum() == 0
'''

def fill_categorical_missing_vals(train, val, test):
    
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


    categoric_features = train.drop('Vote', axis=1).select_dtypes(include=['object']).columns

    for f in categoric_features:
        all_sets = [train, val, test]
        for i, data_set in enumerate(all_sets):
            transform_label(all_sets[i], f)
    [train, val, test] = all_sets

    all_sets = [train, val, test]

    for f in categoric_features:
        train_and_val = pd.concat([train, val])
        train_cpy = []
        for ff in categoric_features:
            train_cpy = train_and_val[(~train_and_val[ff] < 0 )]

        tree_data = train_cpy[(~train_cpy[f] < 0 )]
        tree_data = tree_data.drop(columns='Vote')
        #transform_label(tree_data, f)
        X = tree_data.drop(columns=f)
        Y = tree_data[f]

        from sklearn import neighbors
        clf = RandomForestClassifier()
        clf.fit(X, Y)


        for i, data_set in enumerate(all_sets):
            #transform_label(all_sets[i], f)
            for_prediction = data_set[data_set[f] < 0].drop(columns=f)
            for_prediction = for_prediction.drop(columns='Vote')
            for index, row in data_set[data_set[f] < 0].iterrows():
                predict = clf.predict(for_prediction.loc[index].to_numpy().reshape(1, -1))
                all_sets[i].ix[index, f] = predict

        train = all_sets[0]
        val = all_sets[1]
        test = all_sets[2]
        all_sets = [train, val, test]

        assert len(train[(train[f] < 0)]) == 0
        assert len(val[(val[f] < 0)]) == 0
        assert len(test[(test[f] < 0)]) == 0


    train["Occupation"] = all_sets[0]["Occupation"].astype('category')
    val["Occupation"] = all_sets[1]["Occupation"].astype('category')
    test["Occupation"] = all_sets[2]["Occupation"].astype('category')

    train["Main_transportation"] = all_sets[0]["Main_transportation"].astype('category')
    val["Main_transportation"] = all_sets[1]["Main_transportation"].astype('category')
    test["Main_transportation"] = all_sets[2]["Main_transportation"].astype('category')

    train["Most_Important_Issue"] = all_sets[0]["Most_Important_Issue"].astype('category')
    val["Most_Important_Issue"] = all_sets[1]["Most_Important_Issue"].astype('category')
    test["Most_Important_Issue"] = all_sets[2]["Most_Important_Issue"].astype('category')

    all_sets = [train, val, test]
    for i, data_set in enumerate(all_sets):
        for f1 in ['Looking_at_poles_results', 'Gender', 'Married', 'Voting_Time', 'Financial_agenda_matters']:
            for index, row in data_set[data_set[f1] == 0].iterrows():
                all_sets[i].ix[index, f1] = -1
        for f in ['Age_group' , 'Will_vote_only_large_party']:
            all_sets[i][f] = all_sets[i][f]-1
    [train, val, test] = all_sets

    return train, val, test