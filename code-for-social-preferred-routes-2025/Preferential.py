import numpy as np
import warnings
import Preliminary as pr
import networkx as nx

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Preferential Selection

def select_heads(N,m,replace=False):
  """
  selects random nodes as the head nodes (written as a seperate function for the sake of generality)
  """
  return np.random.choice(range(0,N),m, replace=replace)


#################### Global Dynamics Functions

def degree_preference(i, adj_matrix, self_loop = False):
  """
  returns the probability distribution of nodes based on preferential attachment
  i is the index of node that the pereference is wirtten for (in this dynamic, only affects the case without self loop)
  """
  degrees = np.sum(adj_matrix, axis=1)

  if not self_loop:
    neighbor_degree = np.delete(degrees, i) #deleted so the tail nodes be selected from nodes other than node i (i.e no self loop)
    normal_coeff = np.sum(neighbor_degree)
    if normal_coeff != 0:
      probability = neighbor_degree/normal_coeff
      probability = np.insert(probability,i,0) #due to the structure of roll function, we need to have probability 0 for node i being chosen as tail
    else:
      warnings.warn("All the degrees are zero (self_loop not allowed).")
      N = np.shape(adj_matrix)[0]
      probability = np.ones(N-1)/(N-1)
      np.insert(probability, i, 0)

  if self_loop:
    normal_coeff = np.sum(degrees)
    if normal_coeff != 0:
      probability = degrees/normal_coeff
    else:
      warnings.warn("All the degrees are zero (self_loop allowed).")
      N = np.shape(adj_matrix)[0]
      probability = np.ones(N)/N

  return probability


def select_tails(heads, adj_matrix, self_loop=False, replace=False, margin=1.5):
  """
  based on the degree_preference, will select tail nodes for the given head nodes
  Beware: the replace argument for the selected head nodes should comply with the replace argument in this function
  """
  tails = []
  N = np.shape(adj_matrix)[0]
  counter_limit = int(N*margin)
  for i in heads:
    probability = degree_preference(i, adj_matrix, self_loop)

    selected = pr.roll(probability)

    if not replace:
      rep = True
      counter = 0
      while (rep==True):
        rep=False
        selected = pr.roll(probability)
        for j in range(len(tails)):
          if tails[j]==selected:
            rep = True
            break
        counter += 1
        if counter>counter_limit:
          rep=False
          warnings.warn("The while-loop in selection without replacement did not converge.")
      tails.append(selected)

    else:
      tails.append(selected)

  return tails


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




######################### Local Functions

def local_degree_preference(i, adj_matrix, self_loop = False):
  """
  returns the probability distribution of nodes based on local preferential attachment (based on the links of a node)
  i is the index of node that the pereference is wirtten for (in this dynamic, only affects the case without self loop)
  """
  links = adj_matrix[i,:]

  if not self_loop:
    links = np.delete(links, i) #deleted so the tail nodes be selected from nodes other than node i (i.e no self loop)
    normal_coeff = np.sum(links)
    if normal_coeff != 0:
      probability = links/normal_coeff
      probability = np.insert(probability,i,0) #due to the structure of roll function, we need to have probability 0 for node i being chosen as tail
    else:
      warnings.warn("All the degrees are zero (self_loop not allowed).")
      N = np.shape(adj_matrix)[0]
      probability = np.ones(N-1)/(N-1)
      np.insert(probability, i, 0)

  if self_loop:
    normal_coeff = np.sum(links)
    if normal_coeff != 0:
      probability = links/normal_coeff
    else:
      warnings.warn("All the degrees are zero (self_loop allowed).")
      N = np.shape(adj_matrix)[0]
      probability = np.ones(N)/N

  return probability


def local_select_tails(heads, adj_matrix, self_loop=False, replace=False, margin=1.5):
  """
  based on the degree_preference, will select tail nodes for the given head nodes
  Beware: the replace argument for the selected head nodes should comply with the replace argument in this function
  """
  tails = []
  N = np.shape(adj_matrix)[0]
  counter_limit = int(N*margin)
  for i in heads:
    probability = local_degree_preference(i, adj_matrix, self_loop)

    selected = pr.roll(probability)

    if not replace:
      rep = True
      counter = 0
      while (rep==True):
        rep=False
        selected = pr.roll(probability)
        for j in range(len(tails)):
          if tails[j]==selected:
            rep = True
            break
        counter += 1
        if counter>counter_limit:
          rep=False
          warnings.warn("The while-loop in of selection without replacement did not converge.")
      tails.append(selected)

    else:
      tails.append(selected)

  return tails


def local_preference_init(N, d0=1, self_loop = False, m0=None, d_m0=None, uniform=False, gaussian=False, powerlaw=False, normal=True):
  """initalizes a fully connected graph with link weights of d0
  will alter randomly chosen m0 links to d_m0 value
  """
  links_matrix = np.ones((N,N))*d0

  # if m0 and d_m0: #only m_0 of the ones in the above will be added d_m0
  #   selected = np.random.randint(0,N,(m0,2))
  #   for [i,j] in selected:
  #     links_matrix[i,j] += d_m0

  if uniform and d0 and d_m0: #d0 is the mean and d_m0 is the breadth of the uniformity
    links_matrix = np.random.uniform(0,1,size=(N,N))*d_m0 + d0

  if powerlaw and d0: #m0 is the exponent, d0 and d_m0 are th elower and upper bounds 
    links_matrix = pr.generate_powerlaw(m0, d0, d_m0, N**2)
    links_matrix = np.reshape(links_matrix, (N,N)) #I've checked, this works well for overall link weight and for each node's weights

  if gaussian and d_m0 and d0: #d0 is th mean and d_M0 is the scale
    links_matrix = np.random.normal(loc=d0,scale=d_m0,size=(N,N))
    links_matrix *= np.sign(links_matrix)

  for i in range(N):
    for j in range(i+1,N):
        links_matrix[j,i] = links_matrix[i,j]


  if not self_loop:
    links_matrix -= links_matrix*np.eye(N)

  if normal:
    for i in range(N):
      links_matrix[i,:] /= np.sum(links_matrix[i,:]) 

  return links_matrix


############################## Functions for both of the dynamics


def update_adj(links, adj_matrix, d=1, directed=False, proportional=False):
  """
  add d weight to the links selected in a single timestep
  """
  if proportional:
    for [head, tail] in links:
      adj_matrix[head, tail] += d*adj_matrix[head, tail] #in this case, d will be the proportinality constant
      if not directed:
        adj_matrix[tail, head] += d*adj_matrix[tail, head] #in this case, d will be the proportinality constant
  else:
    for [head, tail] in links:
      adj_matrix[head, tail] += d
      if not directed:
        adj_matrix[tail, head] += d
    



def evolve(N, m, adj_matrix, d=1, local=False, proportional=False, replace=True, self_loop=False, ti0_track = True, directed=False, margin=1.5):
  """
  run the simulation for one timestep
  if ti0_track is true, will track the statistics of ki0 and ti0
  """
  heads = select_heads(N,m, replace=replace)
  if not local:
    tails = select_tails(heads, adj_matrix ,replace=replace, self_loop=self_loop)
  else:
    tails = local_select_tails(heads, adj_matrix, self_loop=self_loop, replace=False, margin=1.5)
  links = np.column_stack((heads, tails))

  
  ki0 = []
  new_nodes = []
  Lij0 = []
  new_links = []

  if ti0_track:
    unique, indices, counts = np.unique(heads, return_index=True, return_counts=True)
    unique_nodes = [heads[id] for id in indices]
    for i, id in enumerate(unique_nodes):
      if np.sum(adj_matrix[id,:]) == 0:
        ki0.append(counts[i]) #the first non_zero degree of node i (can be anything between 1 and m)
        #len of ki0 will be number of ti0 in this particular timestep
        new_nodes.append(unique_nodes[i])

    unique_l, indices_l, counts_l = np.unique(links, return_index=True, return_counts=True, axis=0)
    unique_links = [[links[id][0],links[id][1]] for id in indices_l]
    for i, [x,y] in enumerate(unique_links):
      if adj_matrix[x,y] == 0:
        Lij0.append(counts_l[i])
        new_links.append(unique_links[i])

  update_adj(links, adj_matrix, d=d, proportional=proportional)


  return ki0, new_nodes, Lij0, new_links



