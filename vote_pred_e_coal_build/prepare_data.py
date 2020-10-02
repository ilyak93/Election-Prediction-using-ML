import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams.update({'font.size': 5})
from sklearn.model_selection import train_test_split
import os
from filling_nan_utils import *
from transform_util import *
import re

pd.options.mode.chained_assignment = None

TARGET_FEATURES = ["Yearly_IncomeK", "Number_of_differnt_parties_voted_for",
                   "Political_interest_Total_Score",
                   "Avg_Satisfaction_with_previous_vote",
                   "Avg_monthly_income_all_years", "Most_Important_Issue",
                   "Overall_happiness_score", "Avg_size_per_room", "Weighted_education_rank",
                   "Vote"]
NUMERIC_TARGET_FEATURES = ["Yearly_IncomeK", "Number_of_differnt_parties_voted_for",
                           "Political_interest_Total_Score",
                           "Avg_Satisfaction_with_previous_vote",
                           "Avg_monthly_income_all_years",
                           "Overall_happiness_score", "Avg_size_per_room",
                           "Weighted_education_rank"]

CATEGORIC_TARGET_FEATURES = ['Most_Important_Issue']
CORRELATED_NUMERIC_FEATURES = ['Avg_monthly_expense_when_under_age_21', 'Avg_monthly_household_cost',
                               'Phone_minutes_10_years',
                               'Avg_education_importance', 'Avg_environmental_importance',
                               'Avg_Residancy_Altitude']
NUMERIC_USEFUL_FEATURES = NUMERIC_TARGET_FEATURES + CORRELATED_NUMERIC_FEATURES

GAUSSIAN_TARGET_FEATURES = ['Avg_Satisfaction_with_previous_vote', 'Avg_size_per_room', 'Weighted_education_rank']
GAUSSIAN_CORRELATED_FEATURES = ['Avg_Residancy_Altitude', 'Avg_monthly_household_cost',
                                'Avg_size_per_room', 'Weighted_education_rank',
                                'Avg_education_importance',
                                'Avg_environmental_importance']
ALL_GAUSSIAN_FEATURES = GAUSSIAN_TARGET_FEATURES + GAUSSIAN_CORRELATED_FEATURES
NON_GAUSSIAN_TARGET_FEATURES = ['Yearly_IncomeK', "Avg_monthly_income_all_years"]

ID_FEATURE = "IdentityCard_Num"


def main():
    train_set, val_set, test_set, unlabeled_set = load_and_split('ElectionsData.csv', '.')
    train_set, val_set, test_set, unlabeled_set = set_clean(train_set, val_set, test_set,
                                                            unlabeled_set, verbose=True,
                                                            graphic=True)

    train_set, val_set, test_set, unlabeled_set = \
        remove_inconsistency(train_set, val_set, test_set, unlabeled_set)
    train_set, val_set, test_set, unlabeled_set = \
        data_transformation(train_set, val_set, test_set, unlabeled_set, False)
    assert sum([s.isna().sum().sum() for s in
                (train_set, val_set, test_set, unlabeled_set)]) == 0

    save_datasets(train_set, val_set, test_set, unlabeled_set)
    #export_features_to_csv(list(train_set))

    print("main finished")


def load_and_split(input_path, backup_dir, train=0.7, validation=0.15, test=0.15):
    '''
    This function load .csv file, the original is not modified.
    Split the data to â€“ train (50-75%), validation, (25-15%), test (25-10%)
    For each set â€“ Keep a copy of the raw-data in backup path
    :param input_path: path to data file .csv
    :param backup dir: dir for backup of 3 sets.
    :param train: train ratio of dataset
    :param validation: validation ration of dataset
    :param test: test ratio of dataset
    :return: 3 pandas arrays (datasets): train, validation, test
    '''
    all_data = pd.read_csv(input_path)

    all_data_length = all_data.shape[0]

    train_and_val, test_set = train_test_split(all_data, test_size=test, stratify=all_data[['Vote']])
    train_set, val_set = \
        train_test_split(train_and_val, test_size=validation / (validation + train),
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
    unlabeled_set = pd.read_csv('ElectionsData_Pred_Features.csv')

    return train_set, val_set, test_set, unlabeled_set


def set_clean(train_set, val_set, test_set, unlabeled_set, verbose=True, graphic=False):
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

    init_features = NUMERIC_USEFUL_FEATURES + CATEGORIC_TARGET_FEATURES + ['Vote']
    all_sets = [train_set, val_set, test_set]
    for index, data_set in enumerate(all_sets):
        all_sets[index] = all_sets[index][init_features]
    [train_set, val_set, test_set] = all_sets

    unlabeled_features = NUMERIC_USEFUL_FEATURES + CATEGORIC_TARGET_FEATURES + [ID_FEATURE]
    unlabeled_set = unlabeled_set[unlabeled_features]

    init_num_nans = num_nas(train_set, val_set, test_set, unlabeled_set, TARGET_FEATURES)

    delete_vals_out_of_range(train_set, val_set, test_set, unlabeled_set,
                         NUMERIC_USEFUL_FEATURES, verbose=True)
    clipped_num_nans = num_nas(train_set, val_set, test_set, unlabeled_set,
                           TARGET_FEATURES)
    assert clipped_num_nans > init_num_nans

    train_set, val_set, test_set, unlabeled_set = \
    fill_nans_by_lin_regress(train_set, val_set, test_set, unlabeled_set,
                             NUMERIC_USEFUL_FEATURES, NUMERIC_TARGET_FEATURES)
    first_fill_num_nans = num_nas(train_set, val_set, test_set, unlabeled_set,
                              TARGET_FEATURES)
    assert clipped_num_nans >= first_fill_num_nans

    delete_outliers(train_set, val_set, test_set, unlabeled_set, ALL_GAUSSIAN_FEATURES)
    no_outliers_num_nans = num_nas(train_set, val_set, test_set, unlabeled_set,
                               TARGET_FEATURES)
    assert no_outliers_num_nans >= first_fill_num_nans

    train_set, val_set, test_set, unlabeled_set = \
    fill_nans_by_lin_regress(train_set, val_set, test_set, unlabeled_set,
                             NUMERIC_USEFUL_FEATURES, NUMERIC_TARGET_FEATURES)
    sec_lin_reg_num_nans = num_nas(train_set, val_set, test_set, unlabeled_set,
                               TARGET_FEATURES)
    assert sec_lin_reg_num_nans <= no_outliers_num_nans

    delete_vals_out_of_range(train_set, val_set, test_set, unlabeled_set,
                         NUMERIC_TARGET_FEATURES, verbose=True)
    reclipped_num_nans = num_nas(train_set, val_set, test_set, unlabeled_set,
                             TARGET_FEATURES)
    assert reclipped_num_nans >= sec_lin_reg_num_nans

    # ++++++++++++++++++++++++ added
    '''
    train_set, val_set, test_set = \
    fill_nans_by_lin_regress(train_set, val_set, test_set, NUMERIC_USEFUL_FEATURES, NUMERIC_TARGET_FEATURES)
    sec_lin_reg_num_nans = num_nas(train_set, val_set, test_set, TARGET_FEATURES)
    assert sec_lin_reg_num_nans <= no_outliers_num_nans

    delete_vals_out_of_range(train_set, val_set, test_set, NUMERIC_TARGET_FEATURES, verbose=True)
    reclipped_num_nans = num_nas(train_set, val_set, test_set, TARGET_FEATURES)
    assert reclipped_num_nans >= sec_lin_reg_num_nans
    '''
    # ++++++++++++++++++++++++ added

    fill_missing_vals_by_mean(train_set, val_set, test_set, unlabeled_set, NUMERIC_TARGET_FEATURES)
    num_features_full_num_nans = num_nas(train_set, val_set, test_set, unlabeled_set, TARGET_FEATURES)
    assert num_features_full_num_nans <= sec_lin_reg_num_nans
    assert num_nas(train_set, val_set, test_set, unlabeled_set, NUMERIC_TARGET_FEATURES) == 0

    all_sets = [train_set, val_set, test_set]
    for index, data_set in enumerate(all_sets):
        all_sets[index] = all_sets[index][TARGET_FEATURES]
    [train_set, val_set, test_set] = all_sets

    unlabeled_features = set(TARGET_FEATURES) - {'Vote'}
    unlabeled_features.add(ID_FEATURE)
    unlabeled_set = unlabeled_set[unlabeled_features]

    return train_set, val_set, test_set, unlabeled_set


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


def remove_inconsistency(train_set, val_set, test_set, unlabeled_set):
    '''
    Removes columns which have exact same features besides Vote, yet differ on the Vote
    '''
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
    columns = list(train_set)
    columns.remove('Vote')
    for data_set in (train_set, val_set, test_set, unlabeled_set):
        data_set.drop_duplicates(columns)
    return train_set, val_set, test_set, unlabeled_set


def data_transformation_without_one_hot(train_set, val_set, test_set, unlabeled_set, graphic=False):
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
    train_set, val_set, test_set, unlabeled_set = scale_sets(train_set, val_set, test_set, unlabeled_set)

    fill_categorical_missing_vals(train_set, val_set, test_set, unlabeled_set, CATEGORIC_TARGET_FEATURES)
    assert sum([s.isna().sum().sum() for s in (train_set, val_set, test_set, unlabeled_set)]) == 0

    if graphic:
        show_set_hist(train_set, title='train_set histogram after scaling')

    # transform_categoric(train_set, val_set, test_set)
    for data_set in [train_set, val_set, test_set]:
        transform_label(data_set, "Vote")

    return train_set, val_set, test_set, unlabeled_set


def data_transformation(train_set, val_set, test_set, unlabeled_set, graphic=False):
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
    train_set, val_set, test_set, unlabeled_set = scale_sets(train_set, val_set, test_set, unlabeled_set)

    train_set, val_set, test_set, unlabeled_set = \
        fill_categorical_missing_vals(train_set, val_set, test_set, unlabeled_set, CATEGORIC_TARGET_FEATURES)

    assert num_nas(train_set, val_set, test_set, unlabeled_set, TARGET_FEATURES) == 0

    if graphic:
        show_set_hist(train_set, title='train_set histogram after scaling')

    transform_categoric(train_set, val_set, test_set)

    return train_set, val_set, test_set, unlabeled_set


def save_datasets(train_set, val_set, test_set, unlabeled_set):
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
    train_set.to_csv('train_transformed.csv', index=False)
    val_set.to_csv('validation_transformed.csv', index=False)
    test_set.to_csv('test_transformed.csv', index=False)
    unlabeled_set.to_csv('unlabeled_set.csv', index=False)


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
    with open('out_SelectedFeatures.csv', 'w') as f:
        f.write(txt)


if __name__ == "__main__":
    main()
