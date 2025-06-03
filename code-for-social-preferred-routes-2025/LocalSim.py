import numpy as np
import os
from tqdm import tqdm
import Preliminary as pr

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Local Simulations


def create_adjacency_matrix(N, distribution="gaussian", params=None): #checked it,
    """
    Create a symmetric weighted adjacency matrix for an undirected network.
    
    Args:
        N (int): Number of nodes.
        distribution (str): Type of distribution for weights ('gaussian', 'uniform', 'powerlaw').
        params (dict): Parameters for the distribution.
        
    Returns:
        np.ndarray: Symmetric adjacency matrix (N x N).
    """
    if params is None:
        params = {}
    
    if distribution == "gaussian":
        mean = params.get("mean", 0.5)
        std = params.get("std", 0.1)
        matrix = np.abs(np.random.normal(mean, std, (N, N)).astype(int))
    elif distribution == "uniform":
        low = params.get("low", 0.1)
        high = params.get("high", 1.0)
        matrix = np.random.uniform(low, high, (N, N)).astype(int)
    elif distribution == "powerlaw":
        exponent = params.get("exponent", 3.0)
        low = params.get("low", 0.01)
        high = params.get("high", 20.0)
        matrix = pr.generate_powerlaw(exponent, low, high, N**2)
        matrix = np.reshape(matrix, (N,N)).astype(int)
    else:
        raise ValueError("Unsupported distribution type. Use 'gaussian', 'uniform', or 'powerlaw'.")
    
    # Symmetrize the matrix to ensure the network is undirected
    matrix = (np.triu(matrix,0) + np.tril(matrix.T,0)) 

    # Set diagonal to zero (no self-loops)
    np.fill_diagonal(matrix, 0)
    
    return matrix

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

#select heads with preference
def select_heads(adjacency_matrix,m,replace=False):
    """
    selects random nodes as the head nodes (written as a seperate function for the sake of generality)
    """
    N = len(adjacency_matrix)
    probabilities = np.sum(adjacency_matrix,axis=0)
    probabilities = probabilities/np.sum(probabilities)
    
    return np.random.choice(range(0,N),m, replace=replace, p=probabilities)



def simulate_local_preference(N, T, m, x, save_steps, output_dir, update_mode='increment',
                               distribution="gaussian", params=None):
    """
    Simulate the described process on a weighted undirected network efficiently without large temporary matrices.
    
    Args:
        N (int): Number of nodes.
        T (int): Number of timesteps.
        m (int): Number of head nodes selected per timestep.
        x (float): Amount to increase the link strength.
        save_steps (list[int]): Timesteps at which to save the adjacency matrix.
        output_dir (str): Directory to save adjacency matrices.
        distribution (str): Distribution type for initial weights ('gaussian', 'uniform', 'powerlaw').
        update (str): Update type of the adjecency matrix ('increment', 'normalized', 'proportional')
        params (dict): Parameters for the weight distribution.
        
    Returns:
        np.ndarray: Final adjacency matrix.
    """
    # Create and normalize the initial adjacency matrix
    adjacency_matrix = create_adjacency_matrix(N, distribution, params)
    # saved_matrices = []
    
    for t in tqdm(range(T)):
        # Select m unique head nodes
        # head_nodes = np.random.choice(N, size=m, replace=False)
        head_nodes = select_heads(adjacency_matrix,m)
        
        # Store links to be updated
        updates = []
        
        for head_node in head_nodes:
            # Get the row for the head node
            probabilities = adjacency_matrix.copy()[head_node,:]
            
            """redundant calculation"""
            # # Normalize the row to create a probability distribution 
            probabilities = probabilities/ probabilities.sum()
            
            """this will automatically avoid self_loop since te diagonal entry is zero."""
            # Select a tail node based on the probability distribution
            tail_node = np.random.choice(N, p=probabilities)
            
            # Store the indices of the head and tail nodes for later updates
            updates.append((head_node, tail_node))
        
        # Apply updates to the adjacency matrix
        for head_node, tail_node in updates:
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


def continue_local_simulation(adj_matrix, T0, T, m, x, save_steps, output_dir, update_mode='increment'):
    N = len(adj_matrix)

    for t in tqdm(range(T0+1,T0+T+1)):
        # Select m unique head nodes
        # head_nodes = np.random.choice(N, size=m, replace=False)

        head_nodes = select_heads(adj_matrix,m)
        
        # Store links to be updated
        updates = []
        
        for head_node in head_nodes:
            # Get the row for the head node
            probabilities = adj_matrix.copy()[head_node,:]
            
            """redundant calculation"""
            # # Normalize the row to create a probability distribution 
            probabilities /= probabilities.sum()
            
            """this will automatically avoid self_loop since te diagonal entry is zero."""
            # Select a tail node based on the probability distribution
            tail_node = np.random.choice(N, p=probabilities)
            
            # Store the indices of the head and tail nodes for later updates
            updates.append((head_node, tail_node))
        
        # Apply updates to the adjacency matrix
        for head_node, tail_node in updates:
            update_symmetric_matrix(adj_matrix, head_node, tail_node, update_mode, x)
        
        # # Normalize the rows of affected nodes
        # affected_nodes = set([node for pair in updates for node in pair])
        # for node in affected_nodes:
        #     normalize_row(adjacency_matrix, node)

        # Save the adjacency matrix if the timestep is in save_steps
        if t in save_steps:
            filename = os.path.join(output_dir, f"adjacency_matrix_t{t}.npy")
            np.save(filename, adj_matrix)
            # saved_matrices.append(adjacency_matrix.copy())