import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 5})


def show_set_hist(dataset, title):
    dataset.hist(bins=30)
    plt.suptitle('%s' % title, fontsize="x-large")
    plt.tight_layout()
    plt.show()


def show_feature_hist(column):
    column.hist(bins=30)
    plt.title(title, ' numeric data histogram')
    plt.tight_layout()
    plt.show()
