'''
Created on May 5, 2015

@author: asifrehan
'''
import os
import networkx as nx
from networkx.utils import dict_to_numpy_array
import pickle as pkl
import  numpy as np

def get_line_graph(prim_gr):
    '''
    makes a line graph from a NetworkX graph of the network
    '''
    return nx.line_graph(prim_gr)

def mk_dist_matrix(trans_graph):
    '''
    returns the distance matrix for the given line graph
    '''
    d = nx.shortest_path_length(trans_graph)
    idseq = [int(key[2]) for key in d.keys()]
    mapping = dict(zip(d.keys(), idseq))
    dist_mat = dict_to_numpy_array(d, mapping=mapping)
    return dist_mat

def aff_def_func(dist, cutoff=3, weight=0.5):
    '''
    value of cells in affinity matrix defined here. For calculating the
    affinity for each cell use the aff_mat(), which vectorizes this function
    to be applied on each cell of the matrix
    '''
    if dist <= cutoff:
        return weight**dist
    else:
        return 0

def aff_mat(dist_mat):
    '''
    input : distance matrix calculated for a matrix using mk_dist_matrix()
    returns: matrix whose each cell has a calculated affinity value
    '''
    vec_calc_afnty = np.vectorize(aff_def_func)
    return vec_calc_afnty(dist_mat)

def get_degree_mat(affinity_matrix):
    '''
    input : affinity matrix
    returns the calculate the diagonal degree matrix
    '''
    return np.diag(np.sum(affinity_matrix, axis=1))

def get_lapl(dist_mat):
    '''
    calculates the Laplacian matrix using the affinity matrix and the degree
    matrix
    '''
    aff = aff_mat(dist_mat)
    deg = get_degree_mat(aff)
    return deg - aff

def main(nx_graph):
    '''
    input : a NetworkX graph object made from the network SHP file
    returns :  the Laplacian matrix
    '''
    dist_matrix = mk_dist_matrix(get_line_graph(nx_graph))
    Lapl = get_lapl(dist_matrix)
    return Lapl

if __name__ == '__main__':
    data_folder = r'../../data/Relevant_files'
    eta_mg = os.path.join(data_folder, r'MultiGraph.p')
    Lapl = main( pkl.load(open(eta_mg, 'rb')) )
    lapl_matrix_file = os.path.join(data_folder, r'Laplacian_matrix.p')
    pkl.dump(Lapl, open(lapl_matrix_file, 'wb'))
