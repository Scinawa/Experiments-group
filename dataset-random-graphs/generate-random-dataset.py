import numpy as np
import argparse
from tqdm import tqdm
import random
import time
import pickle


import sys
sys.path.append('..')

sys.path.append('/Users/scinawa/workspace/grouptheoretical/multi-orbit-bispectrum-main/')

from spectrum_utils import * 
from utils import *

import networkx as nx



def generate_random_skew(multi_orbit, k_correlation, number_of_graphs, number_of_nodes, p_probability):
    skew_spectrums = {}
    graphs = []

    if multi_orbit:
        for k in range(2,k_correlation):
            skew_spectrums["2orbit-{}-corre-dict".format(k)]=[]
    else:
        for k in range(2,k_correlation):
            skew_spectrums["1orbit-{}-corre-dict".format(k)]=[]




    for _ in tqdm(range(int(number_of_graphs)), desc="Generating skew spectrums of random graphs"):

        nxgraph = nx.fast_gnp_random_graph(int(number_of_nodes),float(p_probability))

        graph = nx.to_numpy_array(nxgraph)
        graphs.append(nxgraph)

        for k in range(2,k_correlation):
            #print("Creating {}-th correlation".format(k))


            if multi_orbit:
                func_ = create_func_on_group_from_matrix_2orbits(np.array(graph))
                skew_spectrums["2orbit-{}-corre-dict".format(k)].append(
                    reduced_k_correlation(func_, k=k, method="extremedyn", vector=True))

            else:
                func_ = create_func_on_group_from_matrix_1orbit(graph)
                skew_spectrums["1orbit-{}-corre-dict".format(k)].append(
                    reduced_k_correlation(func_, k=k, method="extremedyn", vector=True))



    ############ USE WITH CARE!!!!!! IT WILL OVERWRITE THE PERVIOUSLY COMPUTED FEATURES FILE
    if multi_orbit:
        with open("dataset-random-graphs//random-{}-graphs-{}-nodes-multi-orbit-prob-{}.pickle".format(int(number_of_nodes), number_of_nodes, p_probability), 'wb') as handle:
            pickle.dump((graphs, skew_spectrums), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("dataset-random-graphs/random-{}-graphs-{}-nodes-single-orbit-prob-{}.pickle".format(int(number_of_nodes), number_of_nodes, p_probability), 'wb') as handle:
            pickle.dump((graphs, skew_spectrums), handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate skew spectra of random graphs.")
    
    parser.add_argument("--multi_orbit", action="store_true", help="Use multi-orbit correlation (default: False).")
    parser.add_argument("--k-correlation", type=int, default=8, help="Value of k for k-correlation (default: 8).")
    parser.add_argument("number_of_graphs", type=int, help="Number of graphs to generate.")
    parser.add_argument("number_of_nodes", type=int, help="Number of nodes in each graph.")
    parser.add_argument("p_probability", type=float, help="Probability of edge creation.")

    args = parser.parse_args()

    generate_random_skew(args.multi_orbit, args.k_correlation, args.number_of_graphs, args.number_of_nodes, args.p_probability)
    
    
    
    #generate_all_undirected()