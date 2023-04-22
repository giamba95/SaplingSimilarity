import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--gamma', type=float,default=0.5,
                        help="value of the gamma parameter used to combine user-based and item-based approaches")
    parser.add_argument('--dataset', type=str,default='export',
                        help="available datasets: [export, amazon-product, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--similarity', type=str,default="sapling_similarity",
                        help="available similarities: [common_neighbors, jaccard, adamic_adar, resource_allocation, cosine_similarity, sorensen, hub_depressed_index, hub_promoted_index, taxonomy_network, probabilistic_spreading, pearson, sapling_similarity]")
    parser.add_argument('--test', type=str, default="test",
                        help="data to apply the model to: [validation, test]")
    return parser.parse_args()
