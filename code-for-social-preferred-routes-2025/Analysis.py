import os
import warnings
import numpy as np
import networkx as nx
from tqdm import tqdm 
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
"""
Note: In this code, the name "overlap" has been used instead of the 'Modified Weighted Jaccard Index'
for convinience.
"""

def sturges(n):
    """
    Sturges rule for number of histogram bins for a data
    """
    return int(np.log(n)/np.log(2)) + 1

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# For Real Data (working with networkx)
# Modified Weighted Jaccard Index and Cosine Similarity Analysis

def calculate_preference_measures(graph_list, mode='both', weighted_overlap=True, overlap=True, similarity=True):
    """
    - Shrinks MultiDiGraphs to only contain nodes from the intersection of all input graphs.
    - Converts MultiDiGraphs to weighted DiGraphs.
    - Computes edge overlap ratio and similarity between two graphs.

    Args:
        graph_list (list): List of two MultiDiGraphs or DiGraphs.
        mode (str): 'out' to only include outward links of a node in constructing an undirected graph,  
                    'in' to only include outward links of a node,
                    'both' for both directions.
        weighted_overlap (bool): if True, will calculate the modified weighted Jaccard index,
                    else will calcualte the [unweighted] Jaccard index.
        overlap (bool): if True, will calulate the Jaccard Index, else will not.
        similarity (bool): if True, will calulate the average similarity of nodes, else will not.

    Returns:
        dict: with keys 'overlap_ratio' and 'similarity_score'
    """
    if len(graph_list) != 2:
        raise ValueError("Function only supports comparison between two graphs.")

    G1, G2 = graph_list

    # Step 1: Find common nodes across graphs
    common_nodes = set(G1.nodes) & set(G2.nodes)
    if not common_nodes:
        warnings.warn("No common nodes found between graphs.")
        return {
        "overlap_ratio": -1,
        "similarity_score": -1
    }

    # Step 2: Convert MultiDiGraphs to weighted DiGraphs
    def shrink_to_common_nodes(graph):
        """Converts a MultiDiGraph to a weighted DiGraph."""
        if isinstance(graph, nx.DiGraph):
            return graph.subgraph(common_nodes).copy()

        elif isinstance(graph, nx.MultiDiGraph):
            weighted_graph = nx.DiGraph()
            weighted_graph.add_nodes_from(common_nodes)

            edge_weights = defaultdict(int)
            for u, v in graph.edges():
                if u in common_nodes and v in common_nodes:
                    edge_weights[(u, v)] += 1

            weighted_graph.add_edges_from((u, v, {'weight': w}) for (u, v), w in edge_weights.items())
            return weighted_graph
        
        else:
            raise ValueError("Graph must be either nx.DiGraph or nx.MultiDiGraph!")

    G1_shrunk, G2_shrunk = shrink_to_common_nodes(G1), shrink_to_common_nodes(G2)

    isolated_nodes1 = list(nx.isolates(G1_shrunk))
    G1_shrunk.remove_nodes_from(isolated_nodes1)
    isolated_nodes2 = list(nx.isolates(G2_shrunk))
    G2_shrunk.remove_nodes_from(isolated_nodes2)

    if len(list(G1_shrunk.edges()))==0 or len(list(G2_shrunk.edges()))==0:
        warnings.warn('G1 or G2 has no edges after common nodes and isolation.')

    if overlap:
        # Step 4: Compute edge overlap
        if weighted_overlap:
            def get_edge_set(graph, mode):
                """Returns a dictionary of edges with summed weights."""
                edges = {}
                total_weights = 0
                
                for u, v, data in graph.edges(data=True):
                    edge = tuple(sorted((u, v))) if mode == 'both' else (u, v)
                    weight = data.get('weight', 0)
                    total_weights += weight
                    edges[edge] = edges.get(edge, 0) + weight
                
                return edges, total_weights
            
            def weighted_intersection_union(edge_set1, edge_set2, weight1, weight2):
                if weight1==0 or weight2==0:
                    warnings.warn(f'One of the weights are zero: W1={weight1}, W2={weight2}')
                    return -1 , -1
                """Computes the weighted intersection and union of two edge sets."""
                #modified weighted version of Jaccard index
                intersection_weight = sum(edge_set1[edge]/weight1 + edge_set2[edge]/weight2 for edge in edge_set1.keys() & edge_set2.keys())
                union_weight = sum(edge_set1.get(edge, 0)/weight1 + edge_set2.get(edge, 0)/weight2 for edge in edge_set1.keys() | edge_set2.keys())
                
                return intersection_weight, union_weight
            
            edgelist_1, weights1 = get_edge_set(G1_shrunk, mode)
            edgelist_2, weights2 =  get_edge_set(G2_shrunk, mode)
            intersection, union = weighted_intersection_union(edgelist_1, edgelist_2, weights1, weights2)
            overlap_ratio = intersection/union if union else 0

        else:
            def get_edge_set(graph, mode):
                """Returns a set of edges based on direction mode."""
                edges = set()
                if mode == 'both':
                    for u, v in graph.edges():
                        edges.add(tuple(sorted((u, v))))  # Make undirected by sorting
                else:
                    edges.update(graph.edges())  # Keep directed
                return edges

            edgelist_1, edgelist_2 = get_edge_set(G1_shrunk, mode), get_edge_set(G2_shrunk, mode)
            intersection = edgelist_1 & edgelist_2
            union = edgelist_1 | edgelist_2
            overlap_ratio = len(intersection) / len(union) if union else 0

    if similarity:
        # Step 5: Compute similarity using optimized function
        def calculate_similarity(G1, G2, mode='both'):
            """Computes graph similarity using cosine similarity of weighted edges."""
            common_nodes = set(G1.nodes) & set(G2.nodes)
            similarities = []
            disparities = []
            node_weights = []  # This will store the sum of weights for each node

            for node_A in common_nodes:
                neighbors_B = set(G1.successors(node_A)) | set(G1.predecessors(node_A)) if mode == 'both' else (
                            set(G1.successors(node_A)) if mode == 'out' else set(G1.predecessors(node_A)))
                neighbors_B |= set(G2.successors(node_A)) | set(G2.predecessors(node_A)) if mode == 'both' else (
                            set(G2.successors(node_A)) if mode == 'out' else set(G2.predecessors(node_A)))

                graph1_norm = graph2_norm = graph1_graph2 = 0
                node_weight = 0  # Store the weight for node_A
                G1_node_weight = 0

                for node_B in neighbors_B:
                    weight_G1 = weight_G2 = 0
                    if mode in ('both', 'out'):
                        edge_data = G1.get_edge_data(node_A, node_B)
                        if edge_data:
                            weight_G1 += edge_data.get('weight', 0)
                        edge_data = G2.get_edge_data(node_A, node_B)
                        if edge_data:
                            weight_G2 += edge_data.get('weight', 0)

                    if mode in ('both', 'in'):
                        edge_data = G1.get_edge_data(node_B, node_A)
                        if edge_data:
                            weight_G1 += edge_data.get('weight', 0)
                        edge_data = G2.get_edge_data(node_B, node_A)
                        if edge_data:
                            weight_G2 += edge_data.get('weight', 0)

                    graph1_norm += weight_G1 ** 2
                    graph2_norm += weight_G2 ** 2
                    graph1_graph2 += weight_G1 * weight_G2

                    G1_node_weight += weight_G1
                    node_weight += weight_G1 + weight_G2  # Add weights to node_weight

                if graph1_norm == 0 or graph2_norm == 0:
                    warnings.warn(f"Zero norm encountered for node {node_A}")
                    continue

                similarities.append(graph1_graph2 / np.sqrt(graph1_norm * graph2_norm))
                node_weights.append(node_weight)  # Save the weight for this node
                disparities.append(graph1_norm/(G1_node_weight)**2)

            return similarities, disparities 

        similarity_score, disparity_score = calculate_similarity(G1_shrunk, G2_shrunk, mode)

    return {
        "overlap_ratio": overlap_ratio if overlap else -1,
        "similarity_score": similarity_score if similarity else -1,
    }


def compute_similarity_and_overlap(i, j, first_half_data, second_half_data, mode, similarity= True, overlap=True, weighted_overlap=True):
    """Compute similarity and overlap for a given (i, j) pair."""
    res1 = calculate_preference_measures([first_half_data[i], second_half_data[j]], mode, weighted_overlap, overlap, similarity)
    res2 = calculate_preference_measures([first_half_data[j], second_half_data[i]], mode, weighted_overlap, overlap, similarity)  # Corrected order

    return {
        f"{i}_{j}": (res1["overlap_ratio"], res1["similarity_score"], res1["G1_disparity_score"]),
        f"{j}_{i}": (res2["overlap_ratio"], res2["similarity_score"], res1["G1_disparity_score"])
    }

def parallel_similarity_overlap(first_half_data, second_half_data, mode='both',
                                 similarity= True, overlap=True, weighted_overlap=True):
    """
    Computes similarity and overlap ratios for pairs of graphs using multithreading.

    Args:
        first_half_data (list): List of first set of graphs.
        second_half_data (list): List of second set of graphs.
        direction (str): Graph comparison mode ('both', 'in', or 'out').

    Returns:
        tuple: (overlap_results, similarity_results)
    """
    overlap_results = {}
    similarity_results = {}
    disparity_results = {}

    # graph pairs for threading
    task_list = [(i, j, first_half_data, second_half_data, mode)
                 for i in range(len(first_half_data)) for j in range(len(second_half_data))]

    # parallel execution
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(lambda args: compute_similarity_and_overlap(*args), task_list)))

    for result in results:
        for key, (overlap_res, similarity_res, disparity_res) in result.items():
            overlap_results[key] = overlap_res
            similarity_results[key] = similarity_res
            disparity_results[key] = disparity_res

    return overlap_results, similarity_results, disparity_results



#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# For Simulated Models (working with np matrices)
# Retweet simualtion on underlying network

def add_noise(prob_dist, eta, i):
    """
    Add noise to a given probability distribution by transforming all elements except the i-th element.
    for a non-self-loop case, this will only work if the self_loop element is zero.
    
    Arguments:
    prob_dist : 1D array-like
        A probability distribution of length N.
    eta : float
        A parameter in the range [0, 1] used to add noise.
    i : int
        The index of the element in prob_dist that will not be transformed.
    
    Returns:
    1D numpy array
        The modified probability distribution after noise addition.
    """
    N = len(prob_dist)
    
    # Convert to a numpy array for easier manipulation
    prob_dist = np.array(prob_dist)
    
    # Loop over the elements and modify according to the rule
    for j in range(N):
        if j != i:
            prob_dist[j] = prob_dist[j] * (1 - eta) + eta / (N - 1)
    
    # Return the modified array
    return prob_dist


def simulate_retweet(adj_matrix, steps, eta=0, init_node=None, symmetric = True):
    """simulate_retweet function but without chosen list"""

    N = len(adj_matrix)
    #steps of the order of the nodes (e.g = N)
    if init_node==None:
        init_node = np.random.randint(0,N) #the source of the news/tweet

    current_node = init_node
    path_taken = np.zeros_like(adj_matrix)
    
    steps_taken = []
    counter = 0
    while counter<steps:
    
        probability = adj_matrix[current_node,:] #links connected to current node
        if probability[current_node]!=0:
            raise ValueError('The self loop element of the matrix is not zero.')

        normalization = np.sum(probability)
        if normalization==0:
            warnings.warn('normalization factor for probability distribution is zero.')
            current_node = (current_node+1)%N
            continue
        else:
            probability = probability/normalization
        if len(probability)==0:
            continue
        probability = add_noise(probability, eta, current_node)


        chosen =  np.random.choice(N, size=1, p=probability)[0]

        path_taken[current_node,chosen]+=1
        if symmetric:
            path_taken[chosen, current_node] +=1
        steps_taken.append([current_node,chosen])
        current_node = chosen
        counter += 1

    
    return path_taken, steps_taken



def overlap_coefficient(matrix1, matrix2):
    """
    The Modified Weighted Jaccard Index
    """
    if not isinstance(matrix1, np.ndarray):
        matrix1 = np.array(matrix1)
    if not isinstance(matrix2, np.ndarray):
        matrix2 = np.array(matrix2)
    mat1 = matrix1.copy()
    mat2 = matrix2.copy()
    N1 = np.sum(mat1)
    N2 = np.sum(mat2)
    mat1 = mat1/N1
    mat2 = mat2/N2
    bool_mat1 = np.heaviside(mat1,0)
    bool_mat2 = np.heaviside(mat2,0)

    intersection_bool = bool_mat1*bool_mat2
    intersection = np.sum((mat1+mat2)*intersection_bool)
    union = np.sum(mat1 + mat2)
    
    return intersection/union if union else 0