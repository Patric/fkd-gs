import pandas as pd
import matplotlib.pyplot as plt
from utils import get_data_frame

politifact_more_than_1_relation = '../resources/correlation_data/politifact_more_than_1_relation/'
politifact_at_least_1_relation = '../resources/correlation_data/politifact_at_least_1_relation/'
politifact_all = '../resources/correlation_data/politifact_all/'
gossipcop = '../resources/correlation_data/gossipcop/'

# TODO: move to json file
features_files = ['eigenvector_to_label.csv',
    'harmonic_closeness_to_label.csv',
    'hits_to_label.csv',
    'betweenness_to_label.csv',
    'closeness_to_label.csv',
    'page_rank_to_label.csv',
    'article_rank_to_label.csv',
    'degree_to_label.csv']


def get_feature_full_path(data_set_path, feature_data):
        return f'{data_set_path}{feature_data}'

def get_features_paths(data_set_path, features_files):
        return list(map(lambda feature_file : get_feature_full_path(data_set_path, feature_file), features_files))

with pd.option_context('display.max_rows', None,'display.max_columns', None,'display.precision', 3):
        df = get_data_frame(get_features_paths(politifact_all, features_files))
        print(df.corr())


hist = df.hist()
plt.show()