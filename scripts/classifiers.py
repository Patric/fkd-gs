from utils import get_data_frame

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

politifact_more_than_1_relation = '../resources/correlation_data/politifact_more_than_1_relation/'

# TODO: move to json file
features_files = ['eigenvector_correlation_data.csv',
    'harmonic_closeness_correlation_data.csv',
    'hits_correlation_data.csv',
    'betweenness_correlation_data.csv',
    'closeness_correlation_data.csv',
    'page_rank_correlation_data.csv',
    'article_rank_correlation_data.csv',
    'degree_correlation_data.csv']


def get_feature_full_path(data_set_path, feature_data):
    return f'{data_set_path}{feature_data}'


def get_features_paths(data_set_path, features_files):
    return list(map(lambda feature_file : get_feature_full_path(data_set_path, feature_file), features_files))

df = get_data_frame(get_features_paths(politifact_more_than_1_relation, features_files))

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=1500),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()
]

X = df[[
    'eigenvector_score', 'harmonic_closeness_centrality', 'hits_hub',
    'hits_auth', 'betweenness_score', 'closeness_score', 'page_rank_score',
    'outDegree', 'inDegree', 'degree'
]]

y = df['user.label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
log_cols = ["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__

    print("=" * 30)
    print(name)

    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))

    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))

    log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
    log = log.append(log_entry)

# clf=RandomForestClassifier(n_estimators=2500)
# clf.fit(X_train,y_train)
print("=" * 30)

# y_pred=clf.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))