from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_data_frame
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import r_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
from matplotlib import pyplot


politifact_more_than_1_relation = './resources/correlation_data/politifact_more_than_1_relation/'
politifact_at_least_1_relation = './resources/correlation_data/politifact_at_least_1_relation/'
politifact_all = './resources/correlation_data/politifact_all/'
gossipcop = './resources/correlation_data/gossipcop/'

# TODO: move to json file
features_files = [
    'eigenvector_to_label.csv', 'harmonic_closeness_to_label.csv',
    'hits_to_label.csv', 'betweenness_to_label.csv', 'closeness_to_label.csv',
    'page_rank_to_label.csv', 'article_rank_to_label.csv',
    'degree_to_label.csv'
]


def get_feature_full_path(data_set_path, feature_data):
    return f'{data_set_path}{feature_data}'


def get_features_paths(data_set_path, features_files):
    return list(
        map(
            lambda feature_file: get_feature_full_path(
                data_set_path, feature_file), features_files))


def tree_classifier(X_train, y_train, X_test):
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_train, y_train)
    print(clf.feature_importances_)


# feature selection
def select_features_chi2(X_train, y_train, X_test):
    print('chi2 test')
    # configure to select all features
    fs = SelectKBest(score_func=chi2, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    for i in range(len(fs.scores_)):
        print('Feature %s: score: %f p_value: %f' % (fs.feature_names_in_[i], fs.scores_[i], fs.pvalues_[i]))
        # plot p_value
    # pyplot.bar([i for i in range(len(fs.pvalues_))], fs.pvalues_)
    # pyplot.show()

    return X_train_fs, X_test_fs, fs

def select_features_f(X_train, y_train, X_test):
    print('f classif test')
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    for i in range(len(fs.scores_)):
        print('Feature %s: score: %f p_value: %f' % (fs.feature_names_in_[i], fs.scores_[i], fs.pvalues_[i]))
    #     # plot p_value

    return X_train_fs, X_test_fs, fs

def select_features_mutual_info(X_train, y_train, X_test):
    print('mutual_info_classif test')
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    for i in range(len(fs.scores_)):
        print('Feature %s: score: %f' % (fs.feature_names_in_[i], fs.scores_[i]))
    #     # plot p_value

    return X_train_fs, X_test_fs, fs

def select_features_r(X_train, y_train, X_test):
    print('r_regression test')
    # configure to select all features
    fs = SelectKBest(score_func=r_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    for i in range(len(fs.scores_)):
        print('Feature %s: score: %f' % (fs.feature_names_in_[i], fs.scores_[i]))
    #     # plot p_value

    return X_train_fs, X_test_fs, fs

df = get_data_frame(get_features_paths(gossipcop, features_files))

# with pd.option_context('display.max_rows', None, 'display.max_columns', None,
#                        'display.precision', 3):
#     pd.options.display.float_format = '{:.3f}'.format
#     print(df.corr(method='pearson'))

X = df[['eigenvector_score', 'harmonic_closeness_centrality', 'hits_hub',
    'hits_auth', 'betweenness_score', 'closeness_score', 'page_rank_score',
    'outDegree', 'inDegree', 'degree']]

y = df['user.label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_regression)

# select_features_mutual_info(X_train, y_train, X_test)
# select_features_f(X_train, y_train, X_test)
# select_features_chi2(X_train, y_train, X_test)
# select_features_r(X_train, y_train, X_test)
tree_classifier(X_train, y_train, X_test)

# what are scores for the features

# for i in range(len(fs.scores_)):KB
# 	print('Feature %s: score: %f p_value: %f' % (fs.feature_names_in_[i], fs.scores_[i], fs.pvalues_[i]))
# # plot the scores
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.show()


# closeness = df['closeness_score']
# user_label = df['user.label']

# #print(feature_selection.mutual_info_regression(closeness.to_numpy(), user_label.to_numpy()))
# fit = np.polyfit(x=user_label, y=closeness, deg=1)
# line_fit = np.poly1d(fit)
# plt.plot(user_label, line_fit(closeness))
# plt.scatter(x=user_label, y=closeness, color='red', alpha=0.01)
# plt.title("Pearson correlation")
# plt.show()