# singleLinkage-benchmark

## Supported implementations

We currently support the following implementations:

Exact implementations: 
- [SciPy's Single-Linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.single.html): the popular Single-Linkage implementation in SciPy.
- [VPTreeJava](https://github.com/elki-project/elki/blob/550de994d477ce25b696192f142dfc03e094fa24/elki-clustering/src/main/java/elki/clustering/hierarchical/HeapOfSearchersSingleLink.java#L143): A single-linkage implementation in ELKI using Vantage-Point trees, see [Erich](https://link.springer.com/chapter/10.1007/978-3-031-75823-2_20)

Approximate implementations (these do not guarantee exact clustering results), see [Okkels et al.]() for further details:
- [HNSWmst]: An algorithm computing the minimum spanning tree on all edges in the HNSW graph.
- [HNSWkruskal]: An algorithm simulating running Kruskal's algorithm on the complete graph induced by the dataset.
- [HNSWhssl]: An approximate single-link algorithm using incremental Heap of Searchers on an HNSW graph.
