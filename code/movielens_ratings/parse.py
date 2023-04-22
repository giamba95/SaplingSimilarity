import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--similarity', type=str,default="sapling_similarity",
                        help="available similarities: [common_neighbors, jaccard, adamic_adar, resource_allocation, cosine_similarity, sorensen, hub_depressed_index, hub_promoted_index, taxonomy_network, probabilistic_spreading, pearson, sapling_similarity]")
    parser.add_argument('--projection', type=str,default="item-based",
                        help="available: [user-based, item-based]")
    return parser.parse_args()
