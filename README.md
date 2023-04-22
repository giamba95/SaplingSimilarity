# Sapling Similarity Collaborative Filtering
This repository contains the implementation for our paper:
> Sapling Similarity Collaborative Filtering: a memory-based approach with high recommendation quality https://arxiv.org/abs/2210.07039

where we introduce Sapling Similarity and SSCF.

# Python version and libraries
The python and library versions used to test this code are:
- python 3.8.8
- numpy 1.19.5
- pandas 1.2.4
- sklearn 1.0.2

# An example to run SSCF on export data
> python main.py --dataset="export" --test="test" --gamma=0.7 --similarity="sapling_similarity"

We provide five processed datasets:
- export
- amazon-product
- gowalla
- yelp2018
- amazon-book

if test="validation", a validation set is used to evaluate the performance.

gamma is the only parameter in our model and its value ranges from 0 (user-based approach) to 1 (item-based approach).

similarity can be one of the following:
- common_neighbors
- jaccard
- adamic_adar
- resource_allocation
- cosine_similarity
- sorensen
- hub_depressed_index
- hub-promoted_index
- taxonomy_network
- probabilistic_spreading
- pearson
- sapling_similarity

# Rating prediction
We also provide an implementation to predict ratings with movielens data (movielens_rating folder)
