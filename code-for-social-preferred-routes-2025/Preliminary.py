import numpy as np
import datetime


def roll(massDist,twoD=False):
  """
  given a mass distribution of n probabilities of the form [p1, p2, p3, ..., pn],
  returns the index of the nth probability according to the mass distribution.
  twoD case if for when the array is of the shape [[x1,y1], [x2,y2],...] and we want to chose from xs
  """
  randRoll = np.random.rand()
  summ = 0
  result = 1
  for i in range(len(massDist)):
      if twoD:
        summ += massDist[i][0]
      else:
        summ += massDist[i]
      if randRoll < summ:
          return i



def node_degrees(links, directed = True):
  """
  returns degree sequence of nodes (in and out) from the adjacency matrix
  """
  links = np.array(links)
  matrix_length, matrix_width = links.shape[0], links.shape[1]
  if matrix_length != matrix_width:
    raise Exception('Adjacency matrix should be square.')

  if directed:
    out_distribution = np.zeros(matrix_length)
    in_distribution = np.zeros(matrix_width)
    for i in range(matrix_length):
      out_distribution[i] = np.sum(links[i,:])
      in_distribution[i] = np.sum(links[:,i])
    return in_distribution, out_distribution

  else:
    distribution = np.zeros(links.shape[1])
    for i in range(links.shape[0]):
      distribution[i] = np.sum(links[i,:])
    return distribution



def PDF(arr, log=True, bins=10, nonzero_margin=3):
  """
  makes the probability distribution from a sequence of numbers
  return 2D array of the form [[x_axix point, pdf]]
  """
  if log:
    start = np.log(np.maximum(0.9*np.min(arr),0.001))
    stop = np.log(1.1*np.max(arr))
    print(start,stop)
    bin_edges = np.logspace(start = start, stop = stop, num = bins, base=np.exp(1))
    #max added in start  of logspace to prevent zero in logarithm
    bins = bin_edges 
  counts, bins = np.histogram(arr, bins=bins)
  pdf = np.vstack((bins[:-1],counts))
  zeros = np.where(counts==0)[0]
  nonzero = np.delete(pdf, zeros, axis=1)
  if len(nonzero) != 0:
    return nonzero
  else:
    return pdf


def add_PDF_margin(bin_edge, margin=2):
  """
  returns the extended bin edge with a margin, esepcially for cases where pdf is going to be ensemble averaged
  margin is the number of bin_widths added from both ends
  the rightmost edge is included so that it can be used in numpy histogram
  """
  T = bin_edge[1] - bin_edge[0]
  pre_edge = []
  post_edge = []
  for i in range(margin):
    j = margin - i
    pre_edge.append(bin_edge[0]-j*T)
    post_edge.append(bin_edge[-1]+(i+1)*T)
  new_bin_edge = np.concatenate((pre_edge, bin_edge, post_edge))
  return new_bin_edge

def generate_powerlaw(g, a, b, size=1):
  if a>b:
    c=a
    a=b
    b=c
  uniform = np.random.rand(size)
  low, high = a**g, b**g
  return (low + (high-low)*uniform)**(1/g)


def generate_magic_number():
    """
    generates magic number of the form mm-hh-MM-YY
    """
    now = datetime.now()
    magic_number = f"{now.minute:02d}-{now.hour:02d}-{now.month:02d}-{now.year % 100:02d}"
    return magic_number

