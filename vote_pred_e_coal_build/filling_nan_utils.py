import random
from sklearn import linear_model
from graphic_utils import *
import math
from transform_util import transform_label

LINEAR_FILL_CORR_THRESHOLD = 0.90
CAT_RARITY_THRESHOLD = 0.01
STD_DIFF = 5

plt.rcParams.update({'font.size': 5})


def num_nas(train, val, test, unlabeled, features):
    '''
    Gets three data sets and returns the total number of nas of all three of them in given
    features list
    '''
    sum_nas = 0
    for s in train, val, test, unlabeled:
        for f in set(features) - {'Vote'}:
            sum_nas += s[f].isna().sum().sum()
    return sum_nas


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


def __fill_missing_linear_regression(train, validation, test, unlabeled, features,
                                     corr_mat):
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
    assert isinstance(validation, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert isinstance(unlabeled, pd.DataFrame)
    assert isinstance(features, list)

    sum_nas_before = num_nas(train, validation, test, unlabeled, features)

    all_sets = [train, validation, test, unlabeled]
    for feature in features:
        corr_tuples = [(corr_mat[feature][col], col) for col in corr_mat.columns if col != feature]
        corr_tuples.sort(key=lambda x: x[0], reverse=True)
        while corr_tuples[0][0] >= LINEAR_FILL_CORR_THRESHOLD and \
                sum([s[feature].isna().sum() for s in (train, validation, test)]) > 0:
            reference_feature = corr_tuples[0][1]
            print("useful feature: ", reference_feature)
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

    [train, validation, test, unlabeled] = all_sets
    sum_nas_after = num_nas(train, validation, test, unlabeled, features)
    assert sum_nas_after <= sum_nas_before

    print('we filled', (1-float(sum_nas_after)/sum_nas_before)*100, '% of nas in numerical features',
          'of all three sets')

    return train, validation, test, unlabeled


def fill_missing_vals_by_mean(train, val, test, unlabeled, features):
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
    assert isinstance(unlabeled, pd.DataFrame)
    assert isinstance(features, (list, set))

    for f in features:
        # compute mean
        train_and_val = pd.concat([train, val])
        mean = train_and_val[f].mean()

        for data_set in (train, val, test, unlabeled):
            data_set[f].fillna(mean, inplace=True)
            assert data_set[f].isna().sum() == 0
    return train, val, test, unlabeled


def fill_nans_by_lin_regress(train_set, val_set, test_set, unlabeled_set, corr_features,
                             target_features, verbose=True, graphic=False,
                             all_history=False):
    '''
    Fills all numeric missing values in all three sets, first by correlated features then the rest
    are just filled by the median value
    :param graphic: Whether to show graphs
    :param corr_features: list of relevant numeric features
    :param target_features: all features we fill NaNs in
    :param all_history: Running entire history of experimentations
    :return:
    '''
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
    assert isinstance(unlabeled_set, pd.DataFrame)

    train_and_val = pd.concat([train_set, val_set])
    corr_matrix = train_and_val[corr_features].corr()
    corr_matrix = abs(corr_matrix)

    train_set, val_set, test_set, unlabeled_set = \
        __fill_missing_linear_regression(train_set, val_set, test_set, unlabeled_set,
                                         target_features, corr_matrix)

    return train_set, val_set, test_set, unlabeled_set


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


def delete_vals_out_of_range(train_set, val_set, test_set, unlabeled_set, features,
                             verbose=True):
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
        start_num_nans = num_nas(train_set, val_set, test_set, unlabeled_set, features)

    non_negative_features = ['Avg_Satisfaction_with_previous_vote',
                             'Avg_monthly_income_all_years',
                             'Political_interest_Total_Score',
                             'Avg_size_per_room',
                             'Number_of_differnt_parties_voted_for',
                             'Weighted_education_rank',
                             'Yearly_IncomeK',
                             'Overall_happiness_score',
                             'Avg_monthly_expense_when_under_age_21',
                             'Avg_monthly_household_cost',
                             'Phone_minutes_10_years',
                             'Avg_education_importance',
                             'Avg_environmental_importance',
                             'Avg_Residancy_Altitude'
                             ]

    zero_to_1000_scale_features = ['Avg_education_importance', 'Avg_environmental_importance']
    zero_to_12000_scale_features = ['Avg_Satisfaction_with_previous_vote']

    '''
    Make sure we didn't forget a feature
    '''
    scaled_features = set()
    for f_list in (non_negative_features, zero_to_1000_scale_features,
                   zero_to_12000_scale_features):
        scaled_features = scaled_features.union(set(f_list))
    for f in features:
        assert f in scaled_features

    for data_set in (train_set, test_set, val_set, unlabeled_set):
        for f in train_set.columns:
            if f in non_negative_features:
                __delete_vals_out_of_range(data_set, f, min_val=0)
            elif f in zero_to_1000_scale_features:
                __delete_vals_out_of_range(data_set, f, min_val=0, max_val=1000)
            elif f in zero_to_12000_scale_features:
                __delete_vals_out_of_range(data_set, f, min_val=0, max_val=12000)

    if verbose:
        num_nans_after_clip = num_nas(train_set, val_set, test_set, unlabeled_set,
                                      features)
        num_vals_in_frame = 10000 * len(train_set.columns)
        percentage_dropped_by_clipping = \
            (float(num_nans_after_clip - start_num_nans) * 100) / num_vals_in_frame
        print("Clipping dropped:", str(percentage_dropped_by_clipping) + '%',
              'from all data sets combined')


def delete_outliers(train_set, val_set, test_set, unlabeled_set, features, verbose=True):
    '''
    Deletes all values of features that are STD_THRESHOLD number of standard deviations above mean
    '''
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
    assert isinstance(unlabeled_set, pd.DataFrame)

    if verbose:
        start_num_nans = num_nas(train_set, val_set, test_set, unlabeled_set, features)

    train_and_val = pd.concat([train_set, val_set])

    for f in features:
        # find the mean and the std

        std = train_and_val[f].std()
        mean = train_and_val[f].mean()

        for data_set in (train_set, val_set, test_set, unlabeled_set):
            delta = STD_DIFF * std
            for index, row in data_set[~data_set[f].between(mean - delta, mean + delta)].iterrows():
                data_set.ix[index, f] = np.nan

    if verbose:
        final_num_nans = num_nas(train_set, val_set, test_set, unlabeled_set, features)
        percentage_dropped_by_std_dropping = \
            (float(final_num_nans - start_num_nans) * 100) / (10000 * len(train_set.columns))
        print("Outliers dropped:", str(percentage_dropped_by_std_dropping) + '%', 'of all data sets combined')


def fill_categorical_missing_vals(train, val, test, unlabeled, features):
    '''
    Fills categorical features by the same distribution as present in train data
    :param train: training data set
    :param val: validation data set
    :param test: testing data set
    '''
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert isinstance(unlabeled, pd.DataFrame)

    train_and_val = pd.concat([train, val])
    tree_data = train_and_val[(~train_and_val['Most_Important_Issue'].isnull())]
    assert train_and_val[(~train_and_val['Most_Important_Issue'].isnull())].isna().sum().sum() == 0
    tree_data = tree_data.drop(columns='Vote')
    transform_label(tree_data, 'Most_Important_Issue')
    X = tree_data.drop(columns='Most_Important_Issue')
    Y = tree_data['Most_Important_Issue']

    from sklearn import neighbors
    clf = neighbors.KNeighborsClassifier(10)
    clf = clf.fit(X, Y)

    all_sets = [train, val, test, unlabeled]
    for i, data_set in enumerate(all_sets):
        assert train_and_val[(~train_and_val['Most_Important_Issue'].isnull())].isna().sum().sum() == 0

        transform_label(all_sets[i], 'Most_Important_Issue')

        for_prediction = data_set[data_set['Most_Important_Issue'] < 0].drop(columns='Most_Important_Issue')

        if 'Vote' in for_prediction.columns:
            for_prediction = for_prediction.drop(columns='Vote')
        elif 'IdentityCard_Num' in for_prediction.columns:
            for_prediction = for_prediction.drop(columns='IdentityCard_Num')

        for index, row in data_set[data_set['Most_Important_Issue'] < 0].iterrows():
            predict = clf.predict(for_prediction.loc[index].to_numpy().reshape(1, -1))
            data_set.ix[index, 'Most_Important_Issue'] = predict

    train = all_sets[0]
    val = all_sets[1]
    test = all_sets[2]
    unlabeled = all_sets[3]

    assert num_nas(train, val, test, unlabeled, ['Most_Important_Issue']) == 0

    return train, val, test, unlabeled
