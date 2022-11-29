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
from sklearn.model_selection import KFold




politifact_more_than_1_relation = './resources/correlation_data/politifact_more_than_1_relation/'
politifact_at_least_1_relation = './resources/correlation_data/politifact_at_least_1_relation/'
politifact_all = './resources/correlation_data/politifact_all/'
gossipcop = './resources/correlation_data/gossipcop/'

e13_followers = './resources/correlation_data/MIB/E13/'
fsf_followers = './resources/correlation_data/MIB/FSF/'
int_followers = './resources/correlation_data/MIB/INT/'
twt_followers = './resources/correlation_data/MIB/TWT/'
tfp_followers = './resources/correlation_data/MIB/TFP/'


# TODO: move to json file
features_files = ['eigenvector_to_label.csv',
    'harmonic_closeness_to_label.csv',
    'hits_to_label.csv',
    'betweenness_to_label.csv',
    'closeness_to_label.csv',
    'page_rank_to_label.csv',
    'article_rank_to_label.csv',
    'degree_to_label.csv'
]


def get_feature_full_path(data_set_path, feature_data):
        return f'{data_set_path}{feature_data}'


def get_features_paths(data_set_path, features_files):
        return list(map(lambda feature_file : get_feature_full_path(data_set_path, feature_file), features_files))


def test_classfiers(df):
        classifiers = [
        # KNeighborsClassifier(3),
        # SVC(kernel="rbf", C=0.025, probability=True),
        # NuSVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=1500),
        # AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        # LinearDiscriminantAnalysis(),
        # QuadraticDiscriminantAnalysis()
        ]

        X = df[[#'eigenvector_score', 
               # 'harmonic_closeness_centrality', 
            #    'hits_hub',
           #     'hits_auth',
                'betweenness_score',
                'closeness_score',
            #    'page_rank_score',
            #    'outDegree',
            #    'inDegree',
            #    'degree'
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


        print("=" * 30)

def test_classfiers_for_two_sets(df1, df2):
        classifiers = [
        # KNeighborsClassifier(3),
        # SVC(kernel="rbf", C=0.025, probability=True),
        # NuSVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=1500),
        # AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        # LinearDiscriminantAnalysis(),
        # QuadraticDiscriminantAnalysis()
        ]

        features = [#'eigenvector_score', 
               # 'harmonic_closeness_centrality', 
               #'hits_hub',
               # 'hits_auth',
                'betweenness_score',
                'closeness_score',
             #   'page_rank_score',
             #   'outDegree',
             #   'inDegree',
              #  'degree'
              ]

        X1 = df1[features]

        y1 = df1['user.label']

        
        X2 = df2[features]

        y2 = df2['user.label']

        X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3)
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.9)

        log_cols = ["Classifier", "Accuracy", "Log Loss"]
        log = pd.DataFrame(columns=log_cols)

        for clf in classifiers:
                clf.fit(X1_train, y1_train)
                name = clf.__class__.__name__

                print("=" * 30)
                print(name)

                print('****Results****')
                train_predictions = clf.predict(X2_test)
                acc = accuracy_score(y2_test, train_predictions)
                print("Accuracy: {:.4%}".format(acc))
                


                train_predictions = clf.predict_proba(X2_test)
                ll = log_loss(y2_test, train_predictions)
                print("Log Loss: {}".format(ll))

                log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
                log = log.append(log_entry)


        print("=" * 30)




# df_1 = get_data_frame(get_features_paths(politifact_more_than_1_relation, features_files))
df_2 = get_data_frame(get_features_paths(politifact_all, features_files))
# df_3 = get_data_frame(get_features_paths(politifact_at_least_1_relation, features_files))
df_gossipcop = get_data_frame(get_features_paths(gossipcop, features_files))
#test_classfiers(pd.concat([df_2, df_gossipcop], join='outer', axis=0))
#test_classfiers(df_2)
dfs = list(map(lambda dataset_path: get_data_frame(get_features_paths(dataset_path, features_files)),
[e13_followers, int_followers, twt_followers, fsf_followers, tfp_followers]))

# df_fsf = get_data_frame(get_features_paths(fsf_followers, features_files))
# df_tfp = get_data_frame(get_features_paths(tfp_followers, features_files))

dfs_followers = pd.concat(dfs)

# dfs_followers = dfs_followers.query('degree != 0')
test_classfiers(dfs_followers.query('degree != 0'))
# test_classfiers_for_two_sets(dfs_followers.query('degree != 0'), pd.concat([df_fsf, df_tfp]).query('degree != 0'))

# dfs = pd.concat([df_2, df_gossipcop])
# print(dfs)
# test_classfiers(dfs)

# print('More than 1 relation')
# print("=" * 30, '\n')
# test_classfiers(df_1)

# print('All')
# print("=" * 30, '\n')
# test_classfiers(df_2)

# print('At least 1 relation')
# print("=" * 30, '\n')
# test_classfiers(df_3)