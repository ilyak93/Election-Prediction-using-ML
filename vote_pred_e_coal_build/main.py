import matplotlib.pyplot as plt
import numpy
from keras.utils import np_utils
from sklearn.neural_network import MLPClassifier

from coalition_advanced import print_cluster_distrebutions, print_per_party_distrebution, get_clsuters_to_blocks, \
    get_block_center_of_mass, get_closet_blocks_dict, test_gmm, get_new_coalition_variance
from data_provider import dataProvider
from cross_validation import crossValidator
import numpy as np
import pandas as pd

from model import Keras_MLP
from model import MLP_ensemble

from data_infrastructure import *
from coalition import *



def main():

    dp = dataProvider()
    dp.test_for_nans()
    party_dict = dp.get_vote_dict()
    x_train, y_train = dp.get_train_xy(onehot_y=True)
    x_val, y_val = dp.get_val_xy(onehot_y=True)
    x_test = dp.get_test_xy()
    id_test = dp.get_test_id()

    # Cross validation Random search
   # print("Cross validation Random search with train set of size: ", len(y_train), " val set size:", len(y_val))
    #cv = crossValidator(train_x=x_train, train_y=y_train, num_of_folds=4, max_epochs=500)
    #cv.rand_tune(iter=10000)

    #sanity check
    '''
    classifier = MLPClassifier([95, 95, 95], activation='tanh', max_iter=1000)
    classifier.fit(x_train, y_train)
    y_pred_test = classifier.predict(x_val)
    from sklearn.metrics import f1_score
    print(f"classifier accuracy score: {f1_score(y_val, y_pred_test, average='weighted')}")
    winner = max(y_pred_test.tolist(), key=y_val.tolist().count)
    winner = max(y_val.tolist(), key=y_val.tolist().count)
    winner = max(y_train.tolist(), key=y_train.tolist().count)
    '''


    ensamble = MLP_ensemble('saved_models', party_dict)

    dict = dp.get_vote_dict()

    all_v = numpy.concatenate([y_train, y_val])

    old_winner = (max(all_v.tolist(), key=all_v.tolist().count)).index(1.0)
    print()
    print("The real winner in first Votes Data is: " + dict[old_winner])
    print()

    predicted_winner = ensamble.predict(x_val)
    predicted_winner = max(predicted_winner.tolist(), key=predicted_winner.tolist().count)
    print("The predicted winner of our untouchble test data is: " + dict[predicted_winner])
    print()
    print("Accuracy on our untouchble test data:")
    ensamble.score(x_val, y_val)
    print()

    print("The winner, vote division on the new data:")
    print()
    ensamble.predict_winner(x_test)
    ensamble.predict_vote_division(x_test)
    ensamble.write_pred_to_csv(x_test, id_test)

    print()
    print("the coalition:")
    print()

    df_train, df_val, df_test = load_prepared_dataFrames()

    plot_feature_variance(selected_numerical_features, df_train.var(axis=0)[selected_numerical_features],
                          "Feature Variance")

    new_data = pd.read_csv('.\\'+'unlabeled_set.csv')
    new_data = new_data.drop('IdentityCard_Num', axis=1)
    new_data = new_data.reindex(sorted(new_data.columns), axis=1)
    y_pred_test = ensamble.predict(new_data.values)

    frames = [new_data, pd.Series(y_pred_test, name='Vote'), ]

    df_new_test = pd.concat(frames, axis=1)

    #get_coalition_by_clustering(df_train, df_val, df_test, new_data.values, y_pred_test)

    #get_coalition_by_generative(df_train, df_val, df_test, y_pred_test)

    #train_set, test_set = train_test_split(df_new_test, test_size=0.15, stratify=df_new_test[['Vote']])

    get_coalition_by_clustering2(df_new_test, dp.vote_dictionary)

    get_coalition_by_generative2(df_new_test, dp.vote_dictionary)

    # train_df = concat([import_from_csv(TRAIN_PATH), import_from_csv(VALIDATION_PATH), import_from_csv(TEST_PATH)])
    #test_unlabeled_df = import_from_csv(".\\" + "unlabeled_set.csv")

     #new_data, y_train = divide_data(train_df)  # labels are for generating scores
    #new_data = new_data.drop('Most_Important_Issue', axis=1)
    y_pred_test = pd.Series(y_pred_test)
    #test_gmm(new_data, pd.Series(y_pred_test))

    clf = GaussianMixture(n_components=5, covariance_type='full', init_params='random', random_state=0)

    x_unlabeled_test = new_data
    y_unlabeled_test = y_pred_test

    y_unlabeled_pred = clf.fit_predict(x_unlabeled_test)
    print_cluster_distrebutions(5, y_unlabeled_pred, y_unlabeled_test)
    print_per_party_distrebution(5, y_unlabeled_pred, y_unlabeled_test)

    blocks_dict = get_clsuters_to_blocks(x_unlabeled_test, y_unlabeled_test)
    blocks_cms = get_block_center_of_mass(blocks_dict, x_unlabeled_test, y_unlabeled_test)
    blocks_dist = get_closet_blocks_dict(blocks_cms)

    print(blocks_dist)

    blocks_dict[0].extend(blocks_dict[1])

    vars = get_new_coalition_variance(blocks_dict[0], x_unlabeled_test, y_unlabeled_test)

    plot_feature_variance(selected_numerical_features, vars[0])



if __name__ == "__main__":
    main()