import numpy as np
import scipy.sparse as sp
import csv

# feature = 'profile'
# node_attributes = sp.load_npz(f'new_{feature}_feature.npz')
# with open("A.txt", "r") as edges_file:
#     for count, line in enumerate(edges_file):
#         pass

# print('lines:', count)

# node_graph_id = np.load('node_graph_id.npy')
# graph_node_ids_to_graph_ids = []
# for idx, row in enumerate(node_graph_id):
#     graph_node_ids_to_graph_ids.append([row, idx])
# graph_labels = np.load('graph_labels.npy')

# feature = 'spacy'
# spacy_attributes = sp.load_npz(f'new_{feature}_feature.npz')

# with np.printoptions(threshold=np.inf):
#     print('graphid: ', graph_node_ids_to_graph_ids)
# print('graphlabels: ', graph_labels.size)
# print('node_attributes: ', node_attributes)
# print('node_attributes size: ', node_attributes.size)
# print('spacy_attributes: ', spacy_attributes.size)

def generate_edges():
    with open("./resources/A.txt", "r") as in_file, open('./output/edges.csv', 'w+', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['from', 'to'])

        for line in in_file:
            nodes = line.replace("\n", "").split(", ")
            print([nodes[0], nodes[1]])
            writer.writerow([nodes[0], nodes[1]])

        print('Done')

def generate_nodes_to_graph_id():
    with open('./output/nodes_to_graph_id.csv', 'w+', newline='') as out_file:
        graph_labels = np.load('./resources/graph_labels.npy')
        node_graph_id = np.load('./resources/node_graph_id.npy')
        writer = csv.writer(out_file)
        writer.writerow(['user_node_id', 'graph_id', 'label'])
      
        for node_id, graph_id in enumerate(node_graph_id):
            print([node_id, graph_id, int(graph_labels[graph_id])])
            writer.writerow([node_id, graph_id, graph_labels[graph_id]])

        print('Done')

# graph_labels = np.load('graph_labels.npy')
# print(graph_labels)

generate_nodes_to_graph_id()