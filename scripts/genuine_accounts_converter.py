import csv
import pandas as pd
import re

def generate_nodes_to_graph_id(graph_labels_npy_path, node_graph_id_npy_path, output_path):
    with open(output_path, 'w+', newline='') as out_file:
        graph_labels = np.load(graph_labels_npy_path)
        node_graph_id = np.load(node_graph_id_npy_path)
        writer = csv.writer(out_file)
        writer.writerow(['user_node_id', 'graph_id', 'label'])
      
        for node_id, graph_id in enumerate(node_graph_id):
            print([node_id, graph_id, int(graph_labels[graph_id])])
            writer.writerow([node_id, graph_id, graph_labels[graph_id]])

        print('Done')\


headerList = ['id','text','source','user_id','truncated','in_reply_to_status_id','in_reply_to_user_id','in_reply_to_screen_name','retweeted_status_id','geo','place','contributors','retweet_count','reply_count','favorite_count','favorited','retweeted','possibly_sensitive','num_hashtags','num_urls','num_mentions','created_at','timestamp','crawled_at','updated']


def extract_url(input):
    pattern = "((https?):((//))+([\w\d:#@%/;$()~_?\+-=\.&](#!)?)*)"
    match = re.search(pattern, input)
    return match.group(1) if match else ''

with open('./resources/genuine_accounts.csv/tweets.csv', 'r', encoding='utf-8') as in_file, open('./resources/genuine_accounts.csv/tweets_processed.csv', 'w+', encoding='utf-8', newline='') as out_file:

    stop = None
    writer = csv.writer(out_file)
    writer.writerow(headerList)
    idx = 0
    for i, line in enumerate(csv.reader(
        in_file,
        delimiter=',',
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        skipinitialspace=True,
        escapechar='\\')):

        if stop and idx == stop:
            break
        
        # skip broken text
        line_len = len(line)
        if line_len != 26:
            continue

        for index, cell in enumerate(line):
            if cell == '\\N':
                line[index] = ''

        # merge geolocation coords
        line[9] = line[9] + (':' if line[9] and line[10] else '') + line[10]
        del line[10]

        # extract url
        line[2] = extract_url(line[2])

        for pos, entry in enumerate(line):
            line[pos] = re.sub('"|\"', "'", entry)
            line[pos] = line[pos].replace(",", "")
            line[pos] = line[pos].replace("\\", "\\_")
        
        writer.writerow(line)
        idx += 1
        if i%10000 == 0:
            print(int(i/8470000*100), '%')
    print('Done')

# with open('./resources/genuine_accounts.csv/tweets.csv', 'rb') as in_file, open('./resources/genuine_accounts.csv/tweets_demo.csv', 'wb+') as out_file:
#     out_file.write(in_file.read(512*1000000))


# file = pd.read_csv("./resources/genuine_accounts.csv/tweets.csv")
# file.to_csv("./resources/genuine_accounts.csv/tweets_with_header.csv", header=headerList, index=False)