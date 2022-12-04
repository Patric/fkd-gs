import numpy as np
import scipy.sparse as sp
import csv
import os
import json
import pandas as pd

def get_data_frame(*paths):
        paths = list(sum(paths, [])) 
        dfs = list(map(lambda path: pd.read_csv(path), paths))
        df = pd.concat(dfs, join='outer', axis=1)
        df = df.loc[:,~df.columns.duplicated()].copy()
        return df

def merge_jsons_arrays_to_csv(json_path, writer, array_property_name, df_labels):
  for filename in os.listdir(json_path):
    file_path = os.path.join(json_path, filename)
    print(file_path)
    with open(file_path, 'r') as f:
        data = json.load(f)
        user_id = data['user_id']
        label = df_labels.loc[df_labels['user_id'] == user_id]['label'].values[0]
        for array_elem in data[array_property_name]:
          writer.writerow([user_id, array_elem, label])
    
def retrieve_user_ids_from_news_dir(dir, writer=None, label=None):
  user_ids = []
  print(dir)
  for file_or_dir_name in os.listdir(dir):
    file_or_dir_path = os.path.join(dir, file_or_dir_name)
    if os.path.isdir(file_or_dir_path) and file_or_dir_name == 'tweets':
      user_ids += retrieve_user_ids_from_tweets_dir(file_or_dir_path, writer, label)
  return user_ids

def retrieve_user_ids_from_tweets_dir(tweets_dir_path, writer=None, label=None):
  user_ids = []
  for filename in os.listdir(tweets_dir_path):
    file_path = os.path.join(tweets_dir_path, filename)
    user_id = extract_user_id(file_path, writer, label)
    user_ids.append(user_id)

  return user_ids

def extract_user_id(file_path, writer=None, label=None):
  with open(file_path, 'r') as f:
    data = json.load(f)
    user_id = data['user']['id']
    if writer != None:
      writer.writerow([user_id, label])
    return user_id

def get_user_from_news_dir(dir_path, writer=None, label=None):
  user_ids = []
  for news_dir in os.listdir(dir_path):
    dir = os.path.join(dir_path, news_dir)
    user_ids += retrieve_user_ids_from_news_dir(dir, writer, label)

def map_user_profiles_to_labels(output_file_path):
  with open(output_file_path, 'w+', newline='') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(['user_id', 'label'])
    users_fake = get_user_from_news_dir(politifact_fake_tweets_directory_path, writer, 0)
    users_true = get_user_from_news_dir(politifact_true_tweets_directory_path, writer, 1)
    print('Done')

def merge_jsons_to_csv(dir_path, output_file_path, array_property_name, profile_labales_path):
  with open(output_file_path, 'w+', newline='') as out_file:
    df_labels = get_data_frame([profile_labales_path])
    writer = csv.writer(out_file)
    writer.writerow(['user_id', array_property_name, 'label'])
    merge_jsons_arrays_to_csv(dir_path, writer, array_property_name, df_labels)
    print('Done')


user_followers_directory_path = 'D:\\Studia\\Magisterka inf\\datasets\\FakeNewsNet\\code\\fakenewsnet_dataset\\user_followers'
user_following_directory_path = 'D:\\Studia\\Magisterka inf\\datasets\\FakeNewsNet\\code\\fakenewsnet_dataset\\user_following'

gossipcop_fake_tweets_directory_path = 'D:\\Studia\\Magisterka inf\\datasets\\FakeNewsNet\\code\\fakenewsnet_dataset\\gossipcop\\fake'
gossipcop_true_tweets_directory_path = 'D:\\Studia\\Magisterka inf\\datasets\\FakeNewsNet\\code\\fakenewsnet_dataset\\gossipcop\\real'

politifact_fake_tweets_directory_path = 'D:\\Studia\\Magisterka inf\\datasets\\FakeNewsNet\\code\\fakenewsnet_dataset\\politifact\\fake'
politifact_true_tweets_directory_path = 'D:\\Studia\\Magisterka inf\\datasets\\FakeNewsNet\\code\\fakenewsnet_dataset\\politifact\\real'

# map_user_profiles_to_labels('./resources/fake_news_net_followers/profiles_to_label.csv')

merge_jsons_to_csv(user_following_directory_path, './resources/fake_news_net_followers/users_following.csv', 'following', './resources/fake_news_net_followers/profiles_to_label.csv')
merge_jsons_to_csv(user_followers_directory_path, './resources/fake_news_net_followers/users_followers.csv', 'followers', './resources/fake_news_net_followers/profiles_to_label.csv')

