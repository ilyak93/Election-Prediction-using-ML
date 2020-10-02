import numpy as np
import matplotlib.pyplot as plt
from hist_plotter import plot_hist
from hist_plotter import plot_3hist

PATH_DRAMATIC_FEATURE = 'dramatic_feature'
PRECISION = 3
C_GROW_RANGE = np.arange(1, 20, 0.1)
C_DEC_RANGE = np.arange(1, 0, -0.01)

class featureManipulator():
    def __init__(self, model, x_test, y_test, feature_names, party_dict):
        self.model = model[0]
        self.model_name = model[1]
        self.x_test = x_test
        self.y_test = y_test
        self.party_dict = party_dict
        self.true_winner = 9
        self.feature_names = feature_names
        self.continuous_data = ["Yearly_IncomeK", "Number_of_differnt_parties_voted_for",
                   "Political_interest_Total_Score",
                   "Avg_Satisfaction_with_previous_vote",
                   "Avg_monthly_income_all_years", "Most_Important_Issue",
                   "Overall_happiness_score", "Avg_size_per_room", "Weighted_education_rank",
                   ]
        self.one_hot_data = None

    def find_binary_dramatic_feature(self):
        # continuous data
        true_predictions = self.model.predict(self.x_test)
        for col in range(self.x_test.shape[1]):
            if self.feature_names[col] in self.one_hot_data:
                alterated_x = self.set_one_hot(self.x_test, col)
                predictions = self.model.predict(alterated_x)
                winner = max(set(predictions), key=predictions.tolist().count)
                if winner != self.true_winner:
                    suptitle = 'Feature ' + self.feature_names[col] + ' set to 1 results'
                    winner_name = self.party_dict[winner]
                    title = 'New winner: ' + winner_name
                    path = PATH_DRAMATIC_FEATURE + '\\' + self.feature_names[col] + '_set.png'
                    plot_3hist(path, title, predictions, true_predictions, self.y_test,
                              'manipulated', 'original', 'true', self.party_dict, suptitle=suptitle)


                    print("If ", self.feature_names[col], " will be important to everyone, that will cause ",
                          winner_name,
                          " to win")


    def set_one_hot(self, x_data, col):
        assert self.feature_names[col] in self.one_hot_data
        alterated = np.array(x_data, copy=True)
        for column in range(self.x_test.shape[1]):
            if self.feature_names[column] in self.one_hot_data:
                alterated[:, column] = 0
        alterated[:, col] = 1
        return alterated

    def find_continuous_dramatic_feature(self):
        # continuous data
        true_predictions = self.model.predict(self.x_test)
        for col in range(self.x_test.shape[1]):
            if self.feature_names[col] in self.continuous_data:
                for c in C_GROW_RANGE:
                    alterated_x = self.alterate_column(self.x_test, col, c)
                    predictions = self.model.predict(alterated_x)
                    winner = max(set(predictions), key=predictions.tolist().count)
                    if winner != self.true_winner:
                        suptitle = 'Feature ' + self.feature_names[col] + ' grow by ' + str(np.round(c, PRECISION))
                        winner_name = self.party_dict[winner]
                        title = 'New winner: ' + winner_name
                        path = PATH_DRAMATIC_FEATURE + '\\' + self.feature_names[col] + '_increased.png'

                        plot_3hist(path, title, predictions, true_predictions, self.y_test,
                                   'manipulated', 'original', 'true', self.party_dict, suptitle=suptitle)

                        print("If ", self.feature_names[col], " will grow by ",
                              np.round(c, PRECISION), ", that will cause ",
                              winner_name, " to win")
                        break

                for c in C_DEC_RANGE:
                    alterated_x = self.alterate_column(self.x_test, col, c)
                    predictions = self.model.predict(alterated_x)
                    winner = max(set(predictions), key=predictions.tolist().count)
                    if winner != self.true_winner:
                        suptitle = 'Feature ' + self.feature_names[col] + ' decreased by ' + str(np.round(c, PRECISION))
                        winner_name = self.party_dict[winner]
                        title = 'New winner: ' + winner_name
                        path = PATH_DRAMATIC_FEATURE + '\\' + self.feature_names[col] + '_decreased.png'
                        plot_3hist(path, title, predictions, true_predictions, self.y_test,
                                   'manipulated', 'original', 'true', self.party_dict, suptitle=suptitle)

                        print("If ", self.feature_names[col], " will decrease by ",
                              np.round(c, PRECISION), ", that will cause ",
                              winner_name, " to win")
                        break


    def alterate_column(self, x_data, col, c):
        assert self.feature_names[col] in self.continuous_data
        alterated = np.array(x_data, copy=True)
        alterated[:, col] = c*alterated[:, col]
        return alterated

