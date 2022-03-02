import itertools
import logging
import math
import os.path
import sqlite3
from enum import IntEnum
from operator import itemgetter
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from graph_stats import print_statistics

logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', level=logging.INFO)


def compute_del_ins_random(G, perc_del, perc_ins):
    edges = list(G.edges(data='share'))
    number_of_rows = len(edges)

    random_indices_del = np.random.choice(number_of_rows, size=int(number_of_rows * perc_del), replace=False)
    del_own_set = set(itemgetter(*random_indices_del)(edges))

    random_indices_ins = np.random.choice(number_of_rows, size=int(number_of_rows * perc_ins), replace=False)
    ins_own_set = set(itemgetter(*random_indices_ins)(edges))

    return del_own_set, ins_own_set


def compute_del_ins_centrality(G, perc_del, perc_ins):
    number_of_rows = len(G.edges())

    to_del = int(number_of_rows * perc_del)
    to_ins = int(number_of_rows * perc_ins)

    centrality = nx.degree_centrality(G)
    sort_by_centrality = sorted(centrality.items(), key=itemgetter(1), reverse=True)

    del_own_set = set()
    ins_own_set = set()
    index = 0
    while to_del > len(del_own_set):
        del_own_set = del_own_set.union(set(G.edges(sort_by_centrality[index][0], data='share')))
        index += 1

    while to_ins > len(ins_own_set):
        ins_own_set = ins_own_set.union(set(G.edges(sort_by_centrality[index][0], data='share')))
        index += 1

    return del_own_set, ins_own_set


class Model(IntEnum):
    SCALE_FREE = 1
    SMALL_WORLD = 2
    RANDOM = 3

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)


def generate(number_of_nodes: int, number_of_partitions: int, model: Model, perc_interconnected_nodes: float,
             alpha: float, beta: float, seed: int) -> nx.DiGraph:
    """
    Generate a graph with the given features
    :param number_of_nodes: total number of nodes
    :param number_of_partitions: number of nodes per partition
    :param model: graph topology model
    :param perc_interconnected_nodes: percentage of node in the partition that will receive an incoming edge from a node in another partition
    :param alpha: alpha value for Beta distribution used for extracting random shares
    :param beta: beta value for Beta distribution used for extracting random shares
    :param seed: seed for random number generation
    :return: the final DiGraph
    """

    # Set seed
    np.random.seed(seed)

    G: nx.DiGraph = nx.DiGraph()
    nodes_per_partition: int = math.floor(number_of_nodes / number_of_partitions)
    partitions: List[int] = number_of_partitions * [nodes_per_partition]

    if number_of_nodes % number_of_partitions != 0:
        # This should be a very large partition
        partitions.append(number_of_nodes % number_of_partitions)

    logging.info(
        f"Node per partition: {nodes_per_partition}. Last big partition size: {number_of_nodes % number_of_partitions}")

    for index, partition_size in enumerate(
            tqdm(partitions, total=number_of_partitions, desc="Created partitions", leave=False)):
        offset: int = nodes_per_partition * index
        if model is Model.SCALE_FREE:
            P: nx.DiGraph = nx.barabasi_albert_graph(n=partition_size, m=1, seed=seed + index)
        elif model is Model.SMALL_WORLD:
            P: nx.DiGraph = nx.watts_strogatz_graph(n=partition_size, k=2, p=0.4, seed=seed + index)
        else:
            P: nx.DiGraph = nx.fast_gnp_random_graph(n=partition_size, p=1 / (partition_size - 1),
                                                     seed=seed + index, directed=True)
        G.add_nodes_from((node + offset for node in P.nodes()))
        G.add_weighted_edges_from(((source + offset, target + offset, 0.0) for target, source in P.edges()),
                                  weight='share')

    logging.info(f"Partitions created.")

    #########################################################################################################
    cross_edges: int = int(number_of_nodes * perc_interconnected_nodes * 2)
    cross_nodes_s: List[int] = list(np.random.choice(a=number_of_nodes, size=cross_edges))
    cross_nodes_t: List[int] = list(np.random.choice(a=number_of_nodes, size=cross_edges))

    for s, t in tqdm(zip(cross_nodes_s, cross_nodes_t), desc="Cross edges", total=cross_edges, leave=False):
        G.add_edge(s, t, share=0.0)

    logging.info(f"Added cross-partitions edges.")
    #########################################################################################################

    # For each node set the share of the incoming edges forcing to sum up to 1
    for node in tqdm(G.nodes(), total=number_of_nodes, desc="Assigning shares", leave=False):

        shares: List[float] = np.random.beta(alpha, beta, size=G.in_degree(node))

        shares_sum = np.sum(shares)
        if shares_sum > 1 or shares_sum < 0.4:
            shares: List[float] = shares / np.sum(shares)

        for edge, share in zip(G.in_edges(node), shares):
            G.edges[edge]['share'] = share

    logging.info("Shares assigned.")

    zero_share_edges = [(u, v) for u, v, s in G.edges(data="share") if round(s, 3) == 0]
    G.remove_edges_from(zero_share_edges)
    logging.info(f"Deleted {len(zero_share_edges)} nodes with share equal to 0")

    return G


experiments = {
    'graph_size': {
        'nodes': [100_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000],
        'avg_node_partitions': 5.1,
        'models': [Model.SMALL_WORLD, Model.SCALE_FREE, Model.RANDOM],
        'interconnected': 0.02,
        'seed': 0,
        'perc_del': [0.0002],
        'perc_ins': [0.00024],
    },
    'ownership_random_updates': {
        'nodes': [7_500_000],
        'avg_node_partitions': 3.1,
        'models': [Model.SMALL_WORLD, Model.SCALE_FREE, Model.RANDOM],
        'interconnected': 0.02,
        'seed': 0,
        'perc_del': [1.280e-02, 2.560e-02, 5.120e-02, 1.024e-01, 2.048e-01, 4.096e-01],
        'perc_ins': [1.5360e-02, 3.0720e-02, 6.1440e-02, 1.2288e-01, 2.4576e-01, 4.9152e-01],
    },
    'ownership_centrality_updates': {
        'nodes': [7_500_000],
        'avg_node_partitions': 3.1,
        'models': [Model.SCALE_FREE, Model.SMALL_WORLD, Model.RANDOM],
        'interconnected': 0.1,
        'seed': 0,
        'perc_del': [1.280e-02, 2.560e-02, 5.120e-02],
        'perc_ins': [1.5360e-02, 3.0720e-02, 6.1440e-02],
    }
}

DB_OUTPUT_DIR = './generated/'
SYNTHETIC_OUTPUT_DIR = './generated/'


def main():
    if not os.path.exists(DB_OUTPUT_DIR):
        os.makedirs(DB_OUTPUT_DIR)

    for experiment, settings in experiments.items():
        graphs = itertools.product(zip(settings.get('perc_del'), settings.get('perc_ins')), settings.get('nodes'),
                                   [settings.get('avg_node_partitions')], settings.get('models'))

        for perc, nodes, avg_node_partitions, model in graphs:
            perc_del, perc_ins = perc
            graph_name = f"{experiment}_N{int(nodes / 100_000)}_{model}_D{perc_del}_I{perc_ins}"

            STATS_OUTPUT_DIR = os.path.join(SYNTHETIC_OUTPUT_DIR, experiment)
            if not os.path.exists(STATS_OUTPUT_DIR):
                os.makedirs(STATS_OUTPUT_DIR)

            # GENERATE
            G = generate(number_of_nodes=nodes,
                         number_of_partitions=int(nodes / avg_node_partitions),
                         model=model,
                         perc_interconnected_nodes=settings.get('interconnected'),
                         alpha=2,
                         beta=5,
                         seed=settings.get('seed', 0))

            np.random.seed(settings.get('seed', 0))

            # STATS
            with open(os.path.join(STATS_OUTPUT_DIR, graph_name + ".stats"), 'w') as f:
                f.write(print_statistics(G))
            logging.info("Statistics done!")

            # ADDING END DELETING EDGES
            edges = list(G.edges(data='share'))
            if experiment == 'ownership_centrality_updates':
                del_own_set, ins_own_set = compute_del_ins_centrality(G, perc_del, perc_ins)
            else:
                del_own_set, ins_own_set = compute_del_ins_random(G, perc_del, perc_ins)

            # CREATING DF
            old_own_set = set(edges).difference(ins_own_set)
            new_own_set = set(edges).difference(del_own_set)
            still_own_set = old_own_set.difference(del_own_set)

            # TO SQL
            connection = sqlite3.connect(os.path.join(DB_OUTPUT_DIR, experiment, graph_name + ".sqlite"))
            for table_name, own_set in tqdm([("old_own", old_own_set),
                                             ("del_own", del_own_set),
                                             ("ins_own", ins_own_set),
                                             ("still_own", still_own_set),
                                             ("new_own", new_own_set)],
                                            desc="Tables to SQLite", leave=False):
                table = pd.DataFrame(own_set, columns=["start_id", "end_id", "percentuale"])
                table.to_sql(name=table_name, con=connection, if_exists='replace', index=False)

                connection.execute(f"CREATE INDEX {table_name}_start on {table_name} (start_id);")
                connection.execute(f"CREATE INDEX {table_name}_end on {table_name} (end_id);")
                connection.execute(f"CREATE UNIQUE INDEX {table_name}_start_end on {table_name} (start_id, end_id);")

            connection.commit()
            logging.info("SQL dump done.")


if __name__ == '__main__':
    main()
