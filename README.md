# Reactive Company Control in Company Shareholding Graphs [VLDB 2022]

This repository contains the Python scrips for generating synthetic graphs and computing their stats.

In order to generate new graphs, you need first to install the required Python packages:
```bash
pip install -r requirements.txt
```

Then you can launch the generating process:
```bash
python utils/graph_generator.py
```

For a given experiment definition, the script generates all the possibile combination of varying parameters:
```python
experiments = {
    'graph_size': {
        'nodes': [100_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000],
        'avg_node_partitions': 5.1,
        'models': [Model.SMALL_WORLD, Model.SCALE_FREE, Model.RANDOM],
        'interconnected': 0.02,
        'seed': 0,
        'perc_del': [0.0002],
        'perc_ins': [0.00024],
    }
```
New experiments can be defined direct within the script file.

> For generating graphs, we used Python 3.8

Some graph examples are reported in this repo. Bigger graphs are way too many and too large to be uploaded on this repo, 
but they can be easily computed by running the Python script as it is.
