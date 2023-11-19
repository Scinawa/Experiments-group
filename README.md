# Experiments-Groups


We have the following datasets:
- `dataset-random-graphs`: The k-reduced-skew-spectra of Erdős–Rényi graphs.
- `dataset-simple-graphs`: some simple graphs downloaded from the link below
- `dataset-all-graphs`: all directed and undirected graph for small (very small :) ) n

We have the following files to create and process datasets:

- `generate-random-dataset.py`: generate the datasets in  `dataset-random-graphs` directory and the dataset for `EXP-Atlas-undirected
- `generate_all_graphs`: folder with HPC code to generate in parallel(!) all directed graphs, eventually skipping some.
- `fromsparse.py:` convert g6 graphs into netnworkx->numpy and gerate k-reduced-skew-spectra


We have the following experiments 
- `EXP-on-graphs.ipynb`: the experiments with all undirected graphs of n=7 nodes
- `EXP-discriminating-important-graphs`: our 3-reduced-skew-spectra can distringuish two important graphs :)
- `EXP-new-QM7-regression.ipynb`: Some examples on how the k-reduced-skew spectra improves the regression task on QM7-datasets

We have the following experimeents with Neural Networks:
- `HGP-SL-extended`: from https://github.com/cszhangzhen/HGP-SL

##### Useful links 
- http://users.cecs.anu.edu.au/%7Ebdm/data/graphs.html

pip install torch_geometric==1.7.2
