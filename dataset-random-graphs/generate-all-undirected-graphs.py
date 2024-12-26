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




def generate_from_g6(filepath, multi_orbit, k_correlation):
    skew_spectrums = {}
    graphs = []

    if multi_orbit:
        for k in range(2,k_correlation):
            skew_spectrums["2orbit-{}-corre-dict".format(k)]=[]
    else:
        for k in range(2,k_correlation):
            skew_spectrums["1orbit-{}-corre-dict".format(k)]=[]

    for _ in tqdm(range(int(sys.argv[1])), desc="Generating skew spectrums of random graphs"):
        nxgraph = nx.fast_gnp_random_graph(int(sys.argv[2]),float(sys.argv[3]))

        graph = nx.to_numpy_array(nxgraph)
        graphs.append(nxgraph)

        for k in range(2,k_correlation):
            print("Creating {}-th correlation".format(k))


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
        with open("mo-megadump-random-graphs-features-{}-{}-{}.pickle".format(int(sys.argv[1]), sys.argv[2], sys.argv[3]), 'wb') as handle:
            pickle.dump((graphs, skew_spectrums), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("so-megadump-random-graphs-features-{}-{}-{}.pickle".format(int(sys.argv[1]), sys.argv[2], sys.argv[3]), 'wb') as handle:
            pickle.dump((graphs, skew_spectrums), handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_all_undirected(multi_orbit=False):
    skew_spectrums = {}
    graphs = []

    for k in range(2,9):
        skew_spectrums["1orbit-{}-corre-dict".format(k)]=[]


    for nxgraph in tqdm(nx.graph_atlas_g(), desc="Generating skew spectrums of all graphs"):
        #nxgraph = nx.fast_gnp_random_graph(int(sys.argv[2]),float(sys.argv[3]))

        if nxgraph.number_of_nodes() == 7:
            graph = nx.to_numpy_array(nxgraph)
            graphs.append(nxgraph)

            for k in range(2,9):
                print("Creating {}-th correlation".format(k))

                try:
                    if multi_orbit == True:
                        print("multi-orbit")
                        func_2o = create_func_on_group_from_matrix_2orbits(np.array(graph))

                        skew_spectrums["2orbit-{}-corre-dict".format(k)].append(
                            reduced_k_correlation(func_2o, k=k, method="extremedyn", vector=True))
                    else:
                        print("single-orbit")
                        func_1o = create_func_on_group_from_matrix_1orbit(graph)

                        skew_spectrums["1orbit-{}-corre-dict".format(k)].append(
                            reduced_k_correlation(func_1o, k=k, method="extremedyn", vector=True))
                except Exception as e:
                    print("Exception: {}".format(e))


    ############ USE WITH CARE!!!!!! IT WILL OVERWRITE THE PERVIOUSLY COMPUTED FEATURES FILE
    with open("megadump-atlas_7.pickle", 'wb') as handle:
        pickle.dump((graphs, skew_spectrums), handle, protocol=pickle.HIGHEST_PROTOCOL)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate skew spectra from a graph6 file.")
    
    parser.add_argument("--multi_orbit", action="store_true", help="Use multi-orbit correlation (default: False).")
    parser.add_argument("--k-correlation", type=int, default=8, help="Value of k for k-correlation (default: 8).")

    args = parser.parse_args()



    generate_from_g6(args.filepath, multi_orbit=args.multi_orbit, k_correlation=args.k_correlation)
    print("Done.")




    ### GENERATE RANDOM GRAPHS
    #generate_random_skew(multi_orbit=True)
    
    
    
    #generate_all_undirected()