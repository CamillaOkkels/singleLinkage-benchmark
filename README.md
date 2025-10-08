# singleLinkage-benchmark

## Supported implementations

We currently support the following implementations:

Exact implementations: 
- [SciPy's Single-Linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.single.html): the popular Single-Linkage implementation in SciPy.
- [VPTreeJava](https://github.com/elki-project/elki/blob/550de994d477ce25b696192f142dfc03e094fa24/elki-clustering/src/main/java/elki/clustering/hierarchical/HeapOfSearchersSingleLink.java#L143): A single-linkage implementation in ELKI using Vantage-Point trees, see [Erich](https://link.springer.com/chapter/10.1007/978-3-031-75823-2_20)

Approximate implementations (these do not guarantee exact clustering results), see [Okkels et al.](https://link.springer.com/chapter/10.1007/978-3-032-06069-3_19?fbclid=IwY2xjawNTD-5leHRuA2FlbQIxMABicmlkETB3SmpYbW1Oa05lVTNYRDlLAR4-bZwl1lfu_SfbDis6F1kOr21S5bZBoAw-Ttl99jKcGXkSQxf6LU4f2Yp0vQ_aem_Ts4he6Ug4XJa_tEmEFT6Cg) for further details, and see [HSSL](https://github.com/CamillaOkkels/HSSL/tree/main) for the source code:
- [HNSWmst]: An algorithm computing the minimum spanning tree on all edges in the HNSW graph.
- [HNSWkruskal]: An algorithm simulating running Kruskal's algorithm on the complete graph induced by the dataset.
- [HNSWhssl]: An approximate single-link algorithm using incremental Heap of Searchers on an HNSW graph.

## Supported datasets

| Dataset    | Size (* --> max. sampling size)     | Min. sampling size | Dimensions |
|------------|-----------|------------|-----------|
| MNIST      | 70,000    | 10,000     | 784       |
| Aloi       | 110,249   | 50,000     | 63        |
| Census*    | 250,000   | 50,000     | 500       | 
| Celeba     | 202,599   | 50,000     | 39        |

# HOWTO 

## Installation

Algorithms are carried out in Docker containers, which means that you will need a running docker installation. See for example [this website](https://www.digitalocean.com/community/tutorial-collections/how-to-install-and-use-docker) to get started.

Assuming you have Python version >= 3.8 installed, run

```bash
python3 -m pip install -r requirements.txt 
```

to install all necessary packages. Starting in a fresh python environment is suggested. 

All implementations can be installed using
```bash
python3 install.py
```

# Running an experiment

The standard way to run an experiment is

```
python3 run.py --dataset <DATASET> --algorithm <ALGORITHM> 
```

This will run all configurations known for algorithm on the dataset. The benchmark also allows to run multiple runs with the same parameter settings as well as whether or not existing results should be overwritten or not. This is done by adding -- run <RUN NUMBER> (-r <RUN NUMBER>, default --> 1) and --overwrite <boolean> (-o <boolean>, default --> true). An example could be:

```
python3 run.py --dataset mnist --algorithm HNSWhssl -r 5 -o true
```

After running the experiments, make sure to fix the file permissions by running something like 

```
sudo chmod -R 777 results/
```
## Algorithm configuration

Algorithm configurations are stored in YAML files. The are available in `benchmark/algorithms/<ALGORITHM>/config.yml.`

An example looks like this:

```yaml
docker-image: dbscan-benchmarks-HNSWhssl
module: HNSWhssl
constructor: HNSWSingleLinkage
name: HNSWhssl
args:
  - [5, 11, 22, 47, 100] # ef (efS)
  - [25, 42, 71, 119, 200] # max_build_heap_size (efC)
  - [10, 14, 19, 26, 37, 51, 72, 100] # lowest_max_degree (M)
```

The `args` part specifies all the experiments that are going to be run. 
The Cartesian product of all the given lists is making up the list of individual runs that are tried out in the experiment. (In the example, 5 * 5 * 8 runs are conducted).
`args` have to match the number of arguments expected by the constructor. 
