import networkx as nx
from tqdm import tqdm
import os
import numpy as np
from collections import defaultdict


def weighted_degree(graph, node, mode='out'):
    """
    Calculates the in or out degree of a nod in a diGraph.

    Args:
        graph (nx.classes.graph): networkx graph.
        node (int/str): node label.
        mode (str): in-degree or out-degree calculations ('in','out')

    Returns:
        dict: A dictionary containing mean and standard deviation of similarities for each node.
        tuple: Overall mean and standard deviation across all nodes.
    """
    if mode == 'out':
        # Outgoing weighted degree
        return sum(data['weight'] for _, _, data in graph.out_edges(node, data=True))
    elif mode == 'in':
        # Incoming weighted degree
        return sum(data['weight'] for _, _, data in graph.in_edges(node, data=True))
    else:
        raise ValueError("Mode should be 'in' or 'out'.")
    


def combine_graphs(graph_list): #The more efficient Version
    """
    Combines the graphs based on their node attribute 'name', instead of their node label.
    Works with MultiDiGraphs by collapsing parallel edges into weighted single edges.

    Args:
        graph_list (list of nx.MultiDiGraph): List of networkx MultiDiGraphs to be combined.

    Returns:
        nx.DiGraph: A weighted directed graph where edges are combined based on node 'name' attributes.
    """
    composition_graph = nx.DiGraph()
    edge_weights = defaultdict(int)

    for g in tqdm(graph_list, desc="Combining graphs", leave=False):
        for u, v in g.edges():
            u_id, v_id = g.nodes[u]['name'], g.nodes[v]['name']
            edge_weights[(u_id, v_id)] += 1

    # Add weighted edges efficiently
    composition_graph.add_edges_from((u, v, {'weight': w}) for (u, v), w in edge_weights.items())

    return composition_graph


def graph_to_adjacency_matrix(graph, nodelist, mode='out'):
    """
    base on the model, (in degree, out degree or both), converts the graph to adj matrix
    """
    nodelist = sorted(nodelist)
    if mode=='in' or mode=='both':
        reversed = graph.reverse()
    elif mode!='out':
        raise ValueError("Mode can only take one of the 'in', 'out' or 'both' values.")
    
    # Generate adjacency matrices
    if mode=='in' or mode=='both':
        reversed_adjacency_matrix = nx.adjacency_matrix(reversed, nodelist=nodelist).toarray()
            
    
    if mode=='out' or mode=='both':
        adjacency_matrix = nx.adjacency_matrix(graph, nodelist=nodelist).toarray()
  
        
    if mode=='both':
        final_adjacency_matrix = adjacency_matrix + reversed_adjacency_matrix 
    elif mode=='in':
        final_adjacency_matrix = reversed_adjacency_matrix
    elif mode=='out':
        final_adjacency_matrix = adjacency_matrix
    else:
        raise ValueError('did not enter the if cases')
    
    return final_adjacency_matrix



def equalize_multiple_graphs(graph_list, mode = 'out'):
    """
    Adds isolated nodes to graphs such that the adjecency matrices of them has the same dimensions.

    Args:
        graph_list (list nx.classes.graph): networkx graph to be equalized.

    Returns:
        list: of np.ndarray 
    """
    # Find the union of all nodes in all graphs
    all_nodes = set()
    for graph in graph_list:
        all_nodes.update(graph.nodes)
    all_nodes = sorted(all_nodes)  # Ensure a consistent order of nodes

    # Add missing nodes to each graph
    for graph in graph_list:
        missing_nodes = set(all_nodes)- set(graph.nodes)
        graph.add_nodes_from(missing_nodes)

    #making adjacency matrrices
    adjacency_matrices = [graph_to_adjacency_matrix(graph, all_nodes,mode=mode) for graph in graph_list]
    
    return adjacency_matrices


def shrink_graphs_to_intersection(graph_list, mode = 'out', adj = True):
    """
    Given a list of MultiDiGraphs, this function returns a list of DiGraphs.
    Each DiGraph contains:
      1. Only nodes from the intersection of all MultiDiGraphs' node sets.
      2. Weighted edges where the weight is the number of parallel edges between two nodes in the corresponding MultiDiGraph.

    Parameters:
        multidigraphs (list of nx.MultiDiGraph): List of MultiDiGraphs to process.

    Returns:
        list of nx.DiGraph: List of shrunk weighted DiGraphs.
    """
    # Find the intersection of all node sets
    node_sets = [set(graph.nodes) for graph in graph_list]
    common_nodes = set.intersection(*node_sets)

    # Create weighted DiGraphs for each MultiDiGraph
    adjacency_matrices = []
    for graph in graph_list:

        #for digraphs of aggregation
        if isinstance(graph, nx.DiGraph):
            subgraph = graph.subgraph(common_nodes).copy()

        #for multidigraphs
        elif isinstance(graph, nx.MultiDiGraph): 
            subgraph = nx.DiGraph()

            # Add only common nodes to the DiGraph
            subgraph.add_nodes_from(common_nodes)

            # Add edges with weights based on the number of parallel edges in the MultiDiGraph
            for u, v, data in graph.edges(data=True):
                if u in common_nodes and v in common_nodes:
                    if subgraph.has_edge(u, v):
                        subgraph[u][v]['weight'] += 1
                    else:
                        subgraph.add_edge(u, v, weight=1)
        
        else:
            raise ValueError('The graph input type must either be nx.DiGraph or nx.MultiDiGraph!')

        if adj:
            adjacency_matrix = graph_to_adjacency_matrix(subgraph, common_nodes, mode=mode)
            adjacency_matrices.append(np.array(adjacency_matrix))
        else:
            adjacency_matrices.append(subgraph)
    
    return adjacency_matrices



def find_mutual_nodes(graphs):
    """
    Finds mutual nodes existing in a the given graphs.

    Args:
        graphs (list of nx.classes.graph): networkx graphs to be searched for mutual nodes.

    Returns:
        list: of mutual node labels of the given graphs.

    """
    # Start with the nodes of the first graph
    mutual_nodes = set(graphs[0].nodes)
    for graph in graphs[1:]:
        # Take the intersection with the nodes of each subsequent graph
        mutual_nodes &= set(graph.nodes)
    return list(mutual_nodes)


def names_of(graph):
    """
    Finds the 'name' atttribute of nodes of a graph. (specifically desgined for RT project)

    Args:
        graph (nx.classes.graph): networkx graph to be searched.

    Returns:
        set: of names of nodes

    """
    extracted_names = {item[1]['name'] for item in list(graph.nodes(data=True))}
    return  extracted_names



def compose_temporal_union(main_directory, selected_hashtags, selected_dates, save=False):
    """
    Uses combine_graphs function to combine graphs of different time windows for each given hashtag

    Args:
        main_directory (str): The path to the main directory containing subdirectories for each hashtag.
        selected_hashtags (list of str): A list of hashtags to process.
        selected_dates (list of str): A list of date strings used to match files within the hashtag directories.

    Returns:
        list :A list of composed graphs, where each graph corresponds to a hashtag.
    """
    composed_graphs = []  # List to store the composed graphs for each hashtag
    
    for hashtag in tqdm(selected_hashtags, desc="Temporal Window Union", leave=False):
        raw_data = []  # List to store raw graphs for the current hashtag
        
        for date in selected_dates:
            # Construct the directory and locate matching files
            folder_dir = os.path.join(main_directory, hashtag)
            files = os.listdir(folder_dir)
            matching_files = [file for file in files if hashtag in file and date in file]
            
            if not matching_files:
                raise FileNotFoundError(f"No matching file found for hashtag '{hashtag}' and date '{date}' in {folder_dir}")
            
            # Construct the file path and load the graph
            file_dir = os.path.join(folder_dir, matching_files[0])
            raw_data.append(nx.read_graphml(file_dir))
        
    # Combine the raw graphs for the current hashtag
    combined = combine_graphs(raw_data)
    composed_graphs.append(combined)
    
    #saving the data
    if save:
        output_path = os.path.join(main_directory, hashtag, hashtag+"_agg.graphml")
        nx.write_graphml(combined, output_path)

    else:  
        return composed_graphs
