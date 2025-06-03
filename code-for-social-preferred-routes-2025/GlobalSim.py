import numpy as np
import os
from tqdm import tqdm
import networkx as nx
import warnings

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Global Simulations
def path_graph_initialization(N,m0):
  """
  initializes a path graph wil m0 links in a system of N nodes
  """
  G = nx.path_graph(m0+1)
  links_matrix = nx.to_numpy_array(G)

  init_adj_matrix = np.zeros((N,N))
  for i in range(len(links_matrix)):
    for j in range(len(links_matrix)):
      init_adj_matrix[i,j] = links_matrix[i,j]

  return init_adj_matrix


def create_global_adjacency_matrix(N, init_type="path", m0=5): #checked it,
    """
    Create a symmetric weighted adjacency matrix for an undirected network.
    
    Args:
        N (int): Number of nodes.
        init_type (str): Type of initialization for weights ('path'for path network, 'random' for random links).
        m0 (int): initial number of links.
        
    Returns:
        np.ndarray: Symmetric adjacency matrix (N x N).
    """
    if init_type == "path":
        matrix = path_graph_initialization(N,m0)
    elif init_type == "random":
        matrix = np.zeros((N,N))
        chosen =  np.random.choice(N,(m0,2),replace=True)
        for [i,j] in chosen:
            matrix[i,j] +=1 
            matrix[j,i] +=1
    else:
        raise ValueError("Unsupported distribution type. Use 'random'or 'path'.")
    
    return matrix

#select heads without preference
# def select_heads(N,m,replace=False):
#   """
#   selects random nodes as the head nodes (written as a seperate function for the sake of generality)
#   """
#   return np.random.choice(range(0,N),m, replace=replace)



#select heads with preference
def select_heads(adjacency_matrix,m,replace=False):
    """
    selects random nodes as the head nodes (written as a seperate function for the sake of generality)
    """
    N = len(adjacency_matrix)
    probabilities = np.sum(adjacency_matrix,axis=0)
    probabilities = probabilities/np.sum(probabilities)
    
    return np.random.choice(range(0,N),m, replace=replace, p=probabilities)


def degree_preference(i, adj_matrix, self_loop = False):
    """
    returns the probability distribution of nodes based on preferential attachment
    i is the index of node that the pereference is wirtten for (in this dynamic, only affects the case without self loop)
    """
    degrees = np.sum(adj_matrix, axis=1)

    if not self_loop:
        degrees[i] = 0  #deleted so the tail nodes be selected from nodes other than node i (i.e no self loop)

    normal_coeff = np.sum(degrees)
    if normal_coeff != 0:
        probability = degrees/normal_coeff
    else:
        warnings.warn("All the degrees are zero (self_loop not allowed).")
        N = np.shape(adj_matrix)[0]
        probability = np.ones(N)/(N-1)
        probability[i] = 0

    return probability



def select_tails(heads, adj_matrix, self_loop=False, replace=False, margin=1.5):
  """
  based on the degree_preference, will select tail nodes for the given head nodes
  Beware: the replace argument for the selected head nodes should comply with the replace argument in this function
  """
  tails = []
  N = np.shape(adj_matrix)[0]
  for i in heads:
    probability = degree_preference(i, adj_matrix, self_loop)
    tails.append(np.random.choice(N,size=1, p=probability, replace=replace))

  return tails



def update_adj(links, adj_matrix, d=1, directed=False):
  """
  add d weight to the links selected in a single timestep
  """
  for [head, tail] in links:
    adj_matrix[head, tail] += d
    if not directed:
      adj_matrix[tail, head] += d


def normalize_row(matrix, index):
    """
    Normalize a specific row of a matrix in place to make it sum to 1.
    
    Args:
        matrix (np.ndarray): Input matrix.
        index (int): Row index to normalize.
    """
    row_sum = matrix[index].sum()
    if row_sum > 0:
        matrix[index] /= row_sum

def update_symmetric_matrix(matrix, i, j, mode, value):
    """
    Update a symmetric matrix with specified modes: 'increment' or 'proportional'.
    
    Args:
        matrix (np.ndarray): Symmetric matrix to be updated.
        i (int): Row index of the element to update.
        j (int): Column index of the element to update.
        mode (str): Update mode ('increment', 'proportional').
        value (float): Value to apply in the update.
    """
    if mode == "increment":
        # Increment both [i, j] and [j, i] by the specified value
        matrix[i, j] += value
        matrix[j, i] += value
    
    elif mode == "proportional":
        # Add a proportion of the current value to [i, j] and [j, i]
        matrix[i, j] += matrix[i, j] * value
        matrix[j, i] += matrix[j, i] * value
    
    else:
        raise ValueError("Invalid mode. Use 'increment' or 'proportional'.")


def simulate_global_preference(N, T, m, x, save_steps, output_dir, update_mode='increment',
                               init_type="path", m0=5, replace=False):
    """
    Simulate the described process on a weighted undirected network efficiently without large temporary matrices.
    
    Args:
        N (int): Number of nodes.
        T (int): Number of timesteps.
        m (int): Number of head nodes selected per timestep.
        x (float): Amount to increase the link strength.
        save_steps (list[int]): Timesteps at which to save the adjacency matrix.
        output_dir (str): Directory to save adjacency matrices.
        update_mode (str): Update type of the adjecency matrix:
                        'increment' will only add a constant x at each timestep to the weight of the link,
                        'normalized' is like 'increment' but will normalize the weights before incrementing,
                        'proportional' will add x perent of the current weight of a link to it
        init_type (str): how the m0 weight links is made:
                        'random' will randomly select m0 links and add 1 to their weights,
                        'path' will make a path graph of lenght m0, each link of weight 1.
        m0 (int): sum of the initial link weights at t=0
        params (dict): Parameters for the weight distribution.
        
    Returns:
        np.ndarray: Final adjacency matrix.
    """
    # Create and normalize the initial adjacency matrix
    adjacency_matrix = create_global_adjacency_matrix(N, init_type, m0)
    # saved_matrices = []
    
    for t in tqdm(range(T)):
        # heads = select_heads(N,m, replace=replace)
        heads = select_heads(adjacency_matrix, m, replace=replace)
        tails = select_tails(heads, adjacency_matrix ,replace=replace)
        links = np.column_stack((heads, tails))
        update_adj(links, adjacency_matrix, d=x)
        
        # Apply updates to the adjacency matrix
        for head_node, tail_node in links:
            update_symmetric_matrix(adjacency_matrix, head_node, tail_node, update_mode, x)
        
        # # Normalize the rows of affected nodes
        # affected_nodes = set([node for pair in updates for node in pair])
        # for node in affected_nodes:
        #     normalize_row(adjacency_matrix, node)

        # Save the adjacency matrix if the timestep is in save_steps
        if t in save_steps:
            filename = os.path.join(output_dir, f"adjacency_matrix_t{t}.npy")
            np.save(filename, adjacency_matrix)
            # saved_matrices.append(adjacency_matrix.copy())


def continue_global_simulation(adjacency_matrix, T0, T, m, x, save_steps, output_dir, update_mode='increment', replace=False):
    N = len(adjacency_matrix)

    for t in tqdm(range(T)):
        # heads = select_heads(N,m, replace=replace)
        heads = select_heads(adjacency_matrix,m, replace=replace)
        tails = select_tails(heads, adjacency_matrix ,replace=replace)
        links = np.column_stack((heads, tails))
        update_adj(links, adjacency_matrix, d=x)
        
        # Apply updates to the adjacency matrix
        for head_node, tail_node in links:
            update_symmetric_matrix(adjacency_matrix, head_node, tail_node, update_mode, x)
 
        # Save the adjacency matrix if the timestep is in save_steps
        if t in save_steps:
            filename = os.path.join(output_dir, f"adjacency_matrix_t{t}.npy")
            np.save(filename, adjacency_matrix)