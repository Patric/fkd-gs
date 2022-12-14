import numpy as np
import scipy.sparse as sp
import csv

def generate_edges_csv(edges_input_path, output_path):
    with open(edges_input_path, "r") as in_file, open(output_path, 'w+', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['from', 'to'])

        for line in in_file:
            nodes = line.replace("\n", "").split(", ")
            # print([nodes[0], nodes[1]])
            writer.writerow([nodes[0], nodes[1]])

        print('Done')

def generate_nodes_to_graph_id(graph_labels_npy_path, node_graph_id_npy_path, output_path):
    with open(output_path, 'w+', newline='') as out_file:
        graph_labels = np.load(graph_labels_npy_path)
        node_graph_id = np.load(node_graph_id_npy_path)
        writer = csv.writer(out_file)
        writer.writerow(['user_node_id', 'graph_id', 'label'])
      
        for node_id, graph_id in enumerate(node_graph_id):
            # print([node_id, graph_id, int(graph_labels[graph_id])])
            writer.writerow([node_id, graph_id, graph_labels[graph_id]])

        print('Done')

generate_edges_csv('../resources/gossipcop/A.txt', '../output/gossipcop/edges.csv')
generate_nodes_to_graph_id('../resources/gossipcop/graph_labels.npy', '../resources/gossipcop/node_graph_id.npy', '../output/gossipcop/nodes_to_graph_id.csv')

generate_edges_csv('../resources/politifact/A.txt', '../output/politifact/edges.csv')
generate_nodes_to_graph_id('../resources/politifact/graph_labels.npy', '../resources/politifact/node_graph_id.npy', '../output/politifact/nodes_to_graph_id.csv')
