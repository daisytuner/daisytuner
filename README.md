# Transfer Tuning

This repository implements the [similarity-based transfer tuning algorithm](https://dl.acm.org/doi/abs/10.1145/3577193.3593714), which fuzzy matches loop transformations from online databases using *performance embeddings*.
Additionally, the package provides several utils for benchmarking and optimizing SDFGs.

Installation is as simple as

```bash
# Dependencies
conda install isl nlohmann_json
# Install
pip install daisytuner
```

### Citation

If you use the algorithm, cite us:

```bibtex
@inproceedings{10.1145/3577193.3593714,
    author = {Tr\"{u}mper, Lukas and Ben-Nun, Tal and Schaad, Philipp and Calotoiu, Alexandru and Hoefler, Torsten},
    title = {Performance Embeddings: A Similarity-Based Transfer Tuning Approach to Performance Optimization},
    year = {2023},
    isbn = {9798400700569},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3577193.3593714},
    doi = {10.1145/3577193.3593714},
    booktitle = {Proceedings of the 37th International Conference on Supercomputing},
    pages = {50–62},
    numpages = {13},
    keywords = {compilers, autotuning, performance optimization, peephole optimization, transfer tuning, embeddings},
    location = {Orlando, FL, USA},
    series = {ICS '23}
}
```
