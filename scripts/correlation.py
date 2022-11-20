import pandas as pd
import matplotlib.pyplot as plt
from utils import get_data_frame

eigenvector_correlation_data = '../resources/correlation_data/eigenvector_correlation_data.csv'
harmonic_closeness_correlation_data = '../resources/correlation_data/harmonic_closeness_correlation_data.csv'
hits_correlation_data = '../resources/correlation_data/hits_correlation_data.csv'
betweenness_correlation_data = '../resources/correlation_data/betweenness_correlation_data.csv'
closeness_correlation_data = '../resources/correlation_data/closeness_correlation_data.csv'
page_rank_correlation_data = '../resources/correlation_data/page_rank_correlation_data.csv'
article_rank_correlation_data = '../resources/correlation_data/article_rank_correlation_data.csv'
degree_correlation_data = '../resources/correlation_data/degree_correlation_data.csv'
df = get_data_frame(
        eigenvector_correlation_data,
        harmonic_closeness_correlation_data,
        hits_correlation_data,
        betweenness_correlation_data,
        closeness_correlation_data,
        page_rank_correlation_data,
        article_rank_correlation_data,
        degree_correlation_data)
print(df.corr())
hist = df.hist()
plt.show()