from scripts.classifiers import get_features_paths
from scripts.utils import get_data_frame


politifact_more_than_1_relation = './resources/correlation_data/politifact_more_than_1_relation/'
politifact_at_least_1_relation = './resources/correlation_data/politifact_at_least_1_relation/'
politifact_all = './resources/correlation_data/politifact_all/'
gossipcop = './resources/correlation_data/gossipcop/'


features_files = ['eigenvector_to_label.csv',
    'harmonic_closeness_to_label.csv',
    'hits_to_label.csv',
    'betweenness_to_label.csv',
    'closeness_to_label.csv',
    'page_rank_to_label.csv',
    'article_rank_to_label.csv',
    'degree_to_label.csv'
]

e13_followers = './resources/correlation_data/MIB/E13/'
fsf_followers = './resources/correlation_data/MIB/FSF/'
int_followers = './resources/correlation_data/MIB/INT/'
twt_followers = './resources/correlation_data/MIB/TWT/'
tfp_followers = './resources/correlation_data/MIB/TFP/'

MIB_followers = [e13_followers, fsf_followers, int_followers, twt_followers, tfp_followers]


def get_data_frame(*paths):
  dfs = list(map(lambda dataset_path: get_data_frame(get_features_paths(dataset_path, features_files)), paths))

