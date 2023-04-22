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

# An example to run the code
To run the code using sapling similarity on export data with gamma = 0.7 and using test data to compute the performance:
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

# Computational time
To test our code we used a computer with the following specifics:
- RAM: 32 GiB
- processor: Intel® Core™ i7-10700 CPU @ 2.90GHz × 16 

with our computer the time required to run the SSCF method is:
- export data: 1.13s
- amazon-product: 6.92s
- gowalla: 512.03s
- yelp2018: 512.90s
- amazon-book 11896.65s

to run the methods we do matrix multiplications and with big datasets the computer may not be able to execute the calculations. for this reason we implemented also a version in which similarity and recommendation matrices are computed in 10 blocks (the number of blocks may be regulated changing the M1 variable in the code). This version is main_light.py and with our computer it is necessary in order to run the code in the amazon-book dataset.

# Rating prediction
We also provide an implementation to predict ratings with movielens data (movielens_rating folder)
