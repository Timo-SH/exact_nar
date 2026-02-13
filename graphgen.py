from random import shuffle
from torch_geometric.loader import DataLoader
from torch.utils.data import DistributedSampler
import torch 
import numpy as np
import torch_geometric
import networkx as nx 
import tqdm 
import time 
import os
import copy
import math
import random
BF_VAL = 1000



#creates a single data instance for Bellman-Ford algorithm on graph G
class BellmanFordStep(torch_geometric.data.Data):
    def __init__(self, x, edge_index, edge_attr, y, pos=None):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos

# Sampler class for training data, allowing for distributed sampling if multiple devices are used
class TrainingDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, device_count=1):
        if device_count > 1:
            self.loader = DataLoader(
                dataset,
                batch_size=batch_size // device_count,
                shuffle=False,
                sampler=DistributedSampler(dataset),
            )
            self.loader.sampler.set_epoch(0)
        else:
            self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.iterator = iter(self.loader)
        self.device_count = device_count
        self.epoch = 0

    def sample(self):
        minibatch = next(self.iterator, None)
        if minibatch is None:
            self.iterator = iter(self.loader)
            minibatch = next(self.iterator, None)
            if self.device_count > 1:
                self.epoch += 1
                self.loader.sampler.set_epoch(self.epoch)
        return minibatch

def nx_to_pyg():
    pass

#create a single Bellman-Ford instance from a given networkx graph
def create_bf_instance(G, steps, start=0, start_node=0, furthest_node=None, less_expressive_mpnn=False):
    """
    G: networkx graph with 'weight' attribute on edges and 'attr' attribute on nodes (initial bf values)
    steps: number of bf steps to perform
    start: step at which to record the initial bf attributes (0 means before any step)
    start_node: node from which bf starts (its initial attribute should be 0)
    furthest_node: if given, the target node to report the final value for (otherwise the last node with updated value is used)
    less_expressive_mpnn: if True, all initial node features are set to the same constant value (BF_VAL) (used in Q2)
    """
    edge_attr = []
    edge_index = [[],[]]
    num_nodes = G.number_of_nodes()
    init_node_features = [0.0] * num_nodes
    final_node_features = [0.0] * num_nodes

    final_bf_attr, start_bf_attr, last_node_val, last_node_pos = bf_instance_calculator(
        G=G, steps=steps, start=start, start_node=start_node, furthest_node=furthest_node
    )

    for e in G.edges:
        #edge weights are already given from the corresponding nx graphs during creation
        edge_index[0].append(e[0])
        edge_index[1].append(e[1])
        edge_attr.append(G[e[0]][e[1]]['weight'])
        edge_index[0].append(e[1])
        edge_index[1].append(e[0])
        edge_attr.append(G[e[1]][e[0]]['weight'])
    for node in G.nodes:
        edge_index[0].append(node)
        edge_index[1].append(node)
        edge_attr.append(0)

    for node in G.nodes:
        init_node_features[node] = start_bf_attr[node]['attr']
        final_node_features[node] = final_bf_attr[node]['attr']
    if less_expressive_mpnn: 
        init_node_features = [BF_VAL for val in init_node_features]
    # Return scalar target: last-node value, and store its position
    if last_node_val is None and furthest_node is None:
        last_node_val = 0.0  # fallback when no node met the criteria
        last_node_pos = None
    data = BellmanFordStep(
        x=torch.tensor(init_node_features, dtype=torch.float).unsqueeze(-1),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float).unsqueeze(-1),
        y=torch.tensor(final_node_features, dtype=torch.float).unsqueeze(-1),
        #pos=(torch.tensor(last_node_pos, dtype=torch.long) if last_node_pos is not None else None),
    )
    return data

def create_line_graph(K, weight_pos=0, start_node=0, weight=10):
    """creates a line graph with K+2 nodes and a single edge with non-zero weight at position weight_pos"""
    G = nx.path_graph(K+2)  # Create a line graph with K+2 nodes
    for i in range(len(G.nodes)-1):
        G[i][i+1]['weight'] = 0.0
    G[weight_pos][weight_pos+1]['weight'] = weight
    nx.set_node_attributes(G, BF_VAL, 'attr')
    G.nodes[start_node]['attr'] = 0.0

    return G


def create_multi_line_graph(K, weight_pos, start_node=0, weight=10):
    """creates a graph with K+2 nodes on the main path and K additional paths of decreasing length attached to it,
    with a single edge with non-zero weight at position weight_pos on the main path"""
    G = nx.path_graph(K+2)
    for i in range(len(G.nodes)-1):
        G[i][i+1]['weight'] = 0.0
    G[0][1]['weight'] = weight

    for i in range(K+2, 2, -1):
        F = nx.path_graph(i-2)

        for i in range(len(F.nodes)-1):
            F[i][i+1]['weight'] = 0.0

        current_max_node = len(G.nodes)-1
        G = nx.disjoint_union(G, F)
        G.add_edge(0, current_max_node + 1) #add starting edge
        G[0][current_max_node + 1]['weight'] = 0.0
        G.add_edge(K+2-1, max(G.nodes))
        G[max(G.nodes)][K+2-1]['weight'] = weight  #add ending edge with weight
    nx.set_node_attributes(G, BF_VAL, 'attr')
    G.nodes[start_node]['attr'] = 0.0

    return G


def create_er_graph_with_single_weight(n, p, weight_pos=0, start_node=0, weight=10):
    """"creates an Erdos-Renyi graph with n nodes and edge probability p, with a single edge with non-zero weight at position weight_pos"""
    G = nx.erdos_renyi_graph(n=n, p=p)
    for edge in G.edges:
        G[edge[0]][edge[1]]['weight'] = 0.0
    edges = list(G.edges)
    if weight_pos < len(edges):
        edge_to_modify = edges[weight_pos]
        G[edge_to_modify[0]][edge_to_modify[1]]['weight'] = weight
    nx.set_node_attributes(G, BF_VAL, 'attr')
    G.nodes[start_node]['attr'] = 0.0
    return G

def create_er_graph(n,p,start_node=0, weight_range=(1,10)):
    """"creates an Erdos-Renyi graph with n nodes and edge probability p, with uniform random weights in weight_range"""
    G = nx.erdos_renyi_graph(n=n, p=p)
    for edge in G.edges:
        G[edge[0]][edge[1]]['weight'] = np.random.uniform(weight_range[0], weight_range[1])
    nx.set_node_attributes(G, BF_VAL, 'attr')
    G.nodes[start_node]['attr'] = 0.0
    return G

def create_dual_ba_graph(n, start_node=0, weight_range=(1,5)):
    """creates a dual Barabasi-Albert graph with n nodes, with uniform random weights in weight_range"""
    G = nx.dual_barabasi_albert_graph(n=n, m1=5, m2=3, p=0.6)
    for edge in G.edges:
        G[edge[0]][edge[1]]['weight'] = np.random.uniform(weight_range[0], weight_range[1])
    nx.set_node_attributes(G, BF_VAL, 'attr')
    G.nodes[start_node]['attr'] = 0.0
    return G


def create_complete_graph(n, start_node=0, weight_range=(1,10)):
    """creates a complete graph with n nodes, with uniform random weights in weight_range"""
    G = nx.complete_graph(n=n)
    for edge in G.edges:
        G[edge[0]][edge[1]]['weight'] = np.random.uniform(weight_range[0], weight_range[1])
    nx.set_node_attributes(G, BF_VAL, 'attr')
    G.nodes[start_node]['attr'] = 0.0
    return G

def create_sbm_graph(n, start_node=0, weight_range=(1,10), probs=[[0.7, 0.05, 0.02],[0.05,0.6,0.03],[0.02,0.03,0.4]]):
    """creates a stochastic block model graph with n nodes divided into 3 equal communities, with uniform random weights in weight_range"""
    sizes = [n//3, n//3, n - 2*(n//3)]
    G = nx.stochastic_block_model(sizes, probs)
    for edge in G.edges:
        G[edge[0]][edge[1]]['weight'] = np.random.uniform(weight_range[0], weight_range[1])
    nx.set_node_attributes(G, BF_VAL, 'attr')
    G.nodes[start_node]['attr'] = 0.0
    return G

def create_star_graph(n, start_node=0, weight_range=(1,10)):
    """creates a star graph with n nodes, with uniform random weights in weight_range"""
    G = nx.star_graph(n-1)  
    for edge in G.edges:
        G[edge[0]][edge[1]]['weight'] = np.random.uniform(weight_range[0], weight_range[1])
    nx.set_node_attributes(G, BF_VAL, 'attr')
    G.nodes[start_node]['attr'] = 0.0
    return G

def bf_instance_calculator(G, steps, start=0, start_node=0, furthest_node=None):
    """"performs Bellman-Ford algorithm on graph G for a given number of steps, returns the node attributes at the start step and at the end,
    as well as the value and position of the last updated node"""
    G = copy.deepcopy(G)
    temp = {}
    for node in G.nodes:
        temp[node] = {"attr": G.nodes[node]['attr']}
    if start == 0:
        start_dict = copy.deepcopy(temp)
    for k in range(steps):
        print("step ", k)
        for node in G.nodes:
            if node == start_node:
                continue
            min_val = G.nodes[node]['attr']
            for neighbor in G.neighbors(node):
                val = G[node][neighbor]['weight'] + G.nodes[neighbor]['attr']
                if val < min_val:
                    min_val = val
            temp[node]['attr'] = min_val
        if k == start - 1:
            start_dict = copy.deepcopy(temp)
        nx.set_node_attributes(G, temp)

    # Find the last node (deterministic: highest node id) with value not BF_VAL (BF_VAL is set such that unvisited nodes have this value)
    last_node_pos = None
    last_node_val = None
    if furthest_node is None:
        for node in sorted(G.nodes):
            attr_val = temp[node]['attr']
            if attr_val != BF_VAL:
                last_node_pos = node
                last_node_val = attr_val

    else:
        last_node_pos = furthest_node
        last_node_val = temp[furthest_node]['attr']
    return temp, start_dict, last_node_val, last_node_pos


def _less_expr_suffix(less_expressive_mpnn):
    return "lessmpnn" if less_expressive_mpnn else "fullmpnn"

#create constructor for specific types of graphs
def construct_random_train_dataset(num_graphs, num_nodes, steps, weight=5, less_expressive_mpnn=False):
    """creates a dataset of random graphs for training"""
    dataset = []
    for _ in range(num_graphs):
        G = create_dual_ba_graph(num_nodes, start_node=0, weight_range=(1,5))#create_er_graph(n=num_nodes, p=0.1, weight_range=(1,30))
        data = create_bf_instance(G, steps, start=1, start_node=0, less_expressive_mpnn=less_expressive_mpnn)
        dataset.append(data)
    return dataset

def load_random_train_data(batch_size, weight, less_expressive_mpnn=False):
    """creates and loads a dataset of random graphs for training, saved to disk with given save path"""
    os.makedirs("data", exist_ok=True)
    suffix = _less_expr_suffix(less_expressive_mpnn)
    save_path = os.path.join("data", f"bf_random_train_{weight}_{suffix}.pt")
    if (not os.path.exists(save_path)):
        dataset = construct_random_train_dataset(num_graphs=5, num_nodes=10, steps=3, weight=weight, less_expressive_mpnn=less_expressive_mpnn)
        torch.save(dataset, save_path)
    else:
        dataset = torch.load(save_path, weights_only=False)
    return TrainingDataLoader(dataset, batch_size, shuffle=True)

def construct_random_train_dataset_paths(K, num_graphs, num_nodes, steps, weight=5, less_expressive_mpnn=False):
    """creates a dataset of random graphs and line graphs for training (combination of random graphs and path graphs from Q1)"""
    dataset = []
    for _ in range(num_graphs):
        G = create_er_graph(n=num_nodes, p=0.1, weight_range=(1,30))
        data = create_bf_instance(G, steps, start=1, start_node=0, less_expressive_mpnn=less_expressive_mpnn)
        dataset.append(data)
    for i in range(K+1):
        G = create_line_graph(K=K, weight_pos=i, weight=weight)
        data = create_bf_instance(G=G, steps=steps, start=1, start_node=0, less_expressive_mpnn=less_expressive_mpnn)
        dataset.append(data)
    return dataset

def load_random_train_data_paths(batch_size, weight, less_expressive_mpnn=False):
    """creates and loads a dataset of random graphs and path graphs for training, saved to disk with given save path"""
    os.makedirs("data", exist_ok=True)
    suffix = _less_expr_suffix(less_expressive_mpnn)
    save_path = os.path.join("data", f"bf_random_train_paths_{weight}_{suffix}.pt")
    if (not os.path.exists(save_path)):
        dataset = construct_random_train_dataset_paths(K=2, num_graphs=5, num_nodes=10, steps=3, weight=weight, less_expressive_mpnn=less_expressive_mpnn)
        torch.save(dataset, save_path)
    else:
        dataset = torch.load(save_path, weights_only=False)
    return TrainingDataLoader(dataset, batch_size, shuffle=True)

#create constructor for training dataset
def construct_train_dataset(K, steps=1, weight = 5.0, less_expressive_mpnn=True):
    """creates a dataset of line graphs for training"""
    dataset = []
    for i in range(K+1):
        G = create_line_graph(K=K, weight_pos=i, weight=weight)
        data = create_bf_instance(G=G, steps=steps, start=1, start_node=0, less_expressive_mpnn=less_expressive_mpnn)
        dataset.append(data)

        #optionally create more graphs if needed
    for i in range(K+1):
        G = create_line_graph(K=K, weight_pos=i, weight=weight*0.5)
        data = create_bf_instance(G=G, steps=steps, start=1, start_node=0, less_expressive_mpnn=less_expressive_mpnn)
        dataset.append(data)

    for i in range(K+1):
        G = create_line_graph(K=K, weight_pos=i, weight=weight*2)
        data = create_bf_instance(G=G, steps=steps, start=1, start_node=0, less_expressive_mpnn=less_expressive_mpnn)
        dataset.append(data)

    F = create_multi_line_graph(K=K, weight_pos=K, weight=weight)
    data = create_bf_instance(G=F, steps=steps, start=1, start_node=0, furthest_node=K+1, less_expressive_mpnn=less_expressive_mpnn)
    dataset.append(data)

    return dataset 


def construct_test_dataset(num_graphs_er, num_nodes, steps, less_expressive_mpnn=True):
    """creates a dataset of Erdos-Renyi graphs for testing (ER-constdeg, ER dataset)"""
    dataset = []
    for _ in range(num_graphs_er):
        G = create_er_graph(n=num_nodes, p=0.1, weight_range=(1,100))
        data = create_bf_instance(G, steps, start=0, start_node=0, less_expressive_mpnn=less_expressive_mpnn)
        dataset.append(data)
    return dataset


def construct_complete_test_dataset(num_graphs, num_nodes, steps, less_expressive_mpnn=True):
    """creates a dataset of various graph types for testing (General dataset)"""
    dataset = []
    for _ in range(num_graphs):
        G = create_er_graph(n=num_nodes, p=0.1, weight_range=(1,100))
        data = create_bf_instance(G, steps, start=0, start_node=0, less_expressive_mpnn=less_expressive_mpnn)
        dataset.append(data)
        H = create_complete_graph(n=num_nodes, weight_range=(1,100))
        data = create_bf_instance(H, steps, start=0, start_node=0, less_expressive_mpnn=less_expressive_mpnn)
        dataset.append(data)
        I = create_sbm_graph(n=num_nodes, weight_range=(1,100))
        data = create_bf_instance(I, steps, start=0, start_node=0, less_expressive_mpnn=less_expressive_mpnn)
        dataset.append(data)
        J = create_star_graph(n=num_nodes, weight_range=(1,100))
        data = create_bf_instance(J, steps, start=0, start_node=0, less_expressive_mpnn=less_expressive_mpnn)
        dataset.append(data)
        K = create_multi_line_graph(K=2, weight_pos=2, weight=random.uniform(1,100))
        data = create_bf_instance(K, steps, start=0, start_node=0, less_expressive_mpnn=less_expressive_mpnn)
        dataset.append(data)
    return dataset

def load_complete_test_data(num_graphs_er, num_nodes, steps, less_expressive_mpnn=True):
    """creates and loads a dataset of various graph types for testing, saved to disk with given save path"""
    os.makedirs("data", exist_ok=True)
    suffix = _less_expr_suffix(less_expressive_mpnn)
    save_dataset = os.path.join("data", f"bf_test_complete_{num_graphs_er}_{num_nodes}_{steps}_{suffix}.pt")
    if not os.path.exists(save_dataset):
        dataset = construct_complete_test_dataset(num_graphs_er, num_nodes, steps, less_expressive_mpnn=less_expressive_mpnn)
        torch.save(dataset, save_dataset)
    else:
        dataset = torch.load(save_dataset, weights_only=False)

    return DataLoader(dataset, 1, shuffle=False)

def load_train_data(batch_size,weight, less_expressive_mpnn=True):
    """creates and loads a dataset of line graphs for training, saved to disk with given save path"""
    os.makedirs("data", exist_ok=True)
    suffix = _less_expr_suffix(less_expressive_mpnn)
    save_path = os.path.join("data", f"bf_train_{weight}_{suffix}.pt")
    if (not os.path.exists(save_path)):
        dataset = construct_train_dataset(K=2, steps=3, weight=weight, less_expressive_mpnn=less_expressive_mpnn)
        torch.save(dataset, save_path)
    else:
        dataset = torch.load(save_path, weights_only=False)
    return TrainingDataLoader(dataset, batch_size, shuffle=True)

def load_test_data(num_graphs_er, num_nodes, steps, less_expressive_mpnn=True):
    """creates and loads a dataset of Erdos-Renyi graphs for testing, saved to disk with given save path"""
    os.makedirs("data", exist_ok=True)
    suffix = _less_expr_suffix(less_expressive_mpnn)
    save_dataset = os.path.join("data", f"bf_test_{num_graphs_er}_{num_nodes}_{steps}_{suffix}_normaler.pt")
    if not os.path.exists(save_dataset):
        dataset = construct_test_dataset(num_graphs_er, num_nodes, steps, less_expressive_mpnn=less_expressive_mpnn)
        torch.save(dataset, save_dataset)
    else:
        dataset = torch.load(save_dataset, weights_only=False)

    return DataLoader(dataset, 1, shuffle=False)