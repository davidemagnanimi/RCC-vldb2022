import networkx as nx
import statistics as st
from tqdm import tqdm


def print_statistics(G):
    pbar = tqdm(total=35, desc="Statistics", leave=False)

    # print("### strongly connected components")
    scc = [s for s in nx.strongly_connected_components(G)]
    pbar.update(1)
    # print("### weakly connected components")
    wcc = [w for w in nx.weakly_connected_components(G)]
    pbar.update(1)
    # print("### in-degree, out-degree")
    od = G.out_degree()
    pbar.update(1)
    id = G.in_degree()
    pbar.update(1)

    # number of strongly connected components
    n_cc = len(scc)
    pbar.update(1)
    # number of weakly connected components
    n_wcc = len(wcc)
    pbar.update(1)

    # number of self links
    # print("### self links")
    n_sl = len([e for e in G.edges if e[0] == e[1]])
    pbar.update(1)

    # max out degree
    # print("### max/avg in/out-degree")
    degree_sequence = sorted([d for n, d in od], reverse=True)
    pbar.update(1)
    dmax_out = max(degree_sequence)
    pbar.update(1)

    # max in degree
    degree_sequence = sorted([d for n, d in id], reverse=True)
    pbar.update(1)
    dmax_in = max(degree_sequence)
    pbar.update(1)

    # avg out degree (non-zero)
    degree_sequence = sorted([d for n, d in od if d != 0], reverse=True)
    pbar.update(1)
    davg_out_nz = st.mean(degree_sequence)
    pbar.update(1)

    # avg in degree (non-zero)
    degree_sequence = sorted([d for n, d in id if d != 0], reverse=True)
    pbar.update(1)
    davg_in_nz = st.mean(degree_sequence)
    pbar.update(1)

    # print("### in-degree, out-degree distribution")
    # in-degree distribution
    in_degrees = id
    pbar.update(1)
    in_degrees = dict(in_degrees)
    pbar.update(1)
    in_values = sorted(set(in_degrees.values()))
    pbar.update(1)
    in_hist = [list(in_degrees.values()).count(x) for x in in_values]
    pbar.update(1)
    in_distr = [(v, h) for v, h in zip(in_values, in_hist)]
    pbar.update(1)

    # out-degree distribution
    out_degrees = od
    pbar.update(1)
    out_degrees = dict(out_degrees)
    pbar.update(1)
    out_values = sorted(set(out_degrees.values()))
    pbar.update(1)
    out_hist = [list(out_degrees.values()).count(x) for x in out_values]
    pbar.update(1)
    out_distr = [(v, h) for v, h in zip(out_values, out_hist)]
    pbar.update(1)

    # print("### clustering")
    # average_clustering
    avg_clust = 0  # nx.average_clustering(G)

    # print("### nodes in largest strongly/weakly connected component")
    sorted_scc = sorted(scc, key=len, reverse=True)
    pbar.update(1)
    sorted_wcc = sorted(wcc, key=len, reverse=True)
    pbar.update(1)

    # number of nodes in largest strongly connected component
    number_of_nodes_scc = len(sorted_scc[0])
    # number of nodes in largest weakly connected component
    number_of_nodes_wcc = len(sorted_wcc[0])

    # print("### scc/wcc avg metrics")
    scc_lens = [len(x) for x in sorted_scc]
    pbar.update(1)
    wcc_lens = [len(x) for x in sorted_wcc]
    pbar.update(1)
    scc_avg = st.mean(scc_lens)
    pbar.update(1)
    wcc_avg = st.mean(wcc_lens)
    pbar.update(1)

    # print("### scc size distribution")
    # scc distribution
    scc_values = sorted(set(scc_lens))
    scc_hist = [list(scc_lens).count(x) for x in scc_values]
    scc_dist = [(v, h) for v, h in zip(scc_values, scc_hist)]
    pbar.update(1)

    # print("### wcc size distribution")
    # scc distribution
    wcc_values = sorted(set(wcc_lens))
    wcc_hist = [list(wcc_lens).count(x) for x in wcc_values]
    wcc_dist = [(v, h) for v, h in zip(wcc_values, wcc_hist)]
    pbar.update(1)

    # print("### density")
    density = nx.density(G)
    pbar.update(1)

    # print("### incoming edges per node")
    sum_more_than_one: int = 0
    sum_equal_to_one: int = 0
    sum_equal_to_zero: int = 0
    sum_less_than_one: int = 0

    for node, in_sum in G.nodes(data='sum'):
        sum_of_share: float = round(sum((share for _, _, share in G.in_edges(node, data='share'))), 3)
        if sum_of_share == 0:
            sum_equal_to_zero += 1
        elif sum_of_share > 1:
            sum_more_than_one += 1
        elif sum_of_share < 1:
            sum_less_than_one += 1
        else:
            sum_equal_to_one += 1

    pbar.update(1)

    # print("### generic metrics")

    # print("\n\n\n######################################################")
    stats = nx.info(G) + '\n'
    stats += "Number of strongly connected components: " + str(n_cc) + '\n'
    stats += "Number of weakly connected components: " + str(n_wcc) + '\n'
    stats += "Number of self links: " + str(n_sl) + '\n'
    stats += "Max in degree: " + str(dmax_in) + '\n'
    stats += "Max out degree: " + str(dmax_out) + '\n'
    stats += "Avg non-zero in degree: " + str(davg_in_nz) + '\n'
    stats += "Avg non-zero out degree: " + str(davg_out_nz) + '\n'
    stats += "Nodes in largest scc: " + str(number_of_nodes_scc) + '\n'
    stats += "Nodes in largest wcc: " + str(number_of_nodes_wcc) + '\n'
    stats += "Average clustering: " + str(avg_clust) + '\n'
    stats += "In-degree distribution: " + str(in_distr) + '\n'
    stats += "Out-degree distribution: " + str(out_distr) + '\n'
    stats += "SCC size distribution: " + str(scc_dist) + '\n'
    stats += "WCC size distribution: " + str(wcc_dist) + '\n'
    stats += "AVG nodes in SCC: " + str(scc_avg) + '\n'
    stats += "AVG nodes in WCC: " + str(wcc_avg) + '\n'
    stats += "Density: " + str(density) + '\n'
    stats += f"Nodes with sum of incoming edge shares >1: {sum_more_than_one} \n"
    stats += f"Nodes with sum of incoming edge shares =1: {sum_equal_to_one} \n"
    stats += f"Nodes with sum of incoming edge shares <1: {sum_less_than_one} \n"
    stats += f"Nodes with sum of incoming edge shares =0: {sum_equal_to_zero} \n"

    return stats
