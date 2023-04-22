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
To run the code using sapling similarity on export data with gamma = 0.7:
> python main.py --dataset="export" --test="test" --gamma=0.7 --similarity="sapling_similarity"

We provide five processed datasets:
- export
- amazon-product
- gowalla
- yelp2018
- amazon-book

if test="validation", a validation set is used to evaluate the performance. The validation set we used for the results we show in the paper is located in the corresponding data folder. To generate a new validation set you can run the generate_validation.py code.

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
To test our code we used a computer with the following specifications:
- RAM: 32 GiB
- processor: Intel® Core™ i7-10700 CPU @ 2.90GHz × 16 

with our computer the time required to run the SSCF method is:
- export data: 1.13s
- amazon-product: 6.92s
- gowalla: 512.03s
- yelp2018: 512.90s
- amazon-book 11938.65s

# Code for Amazon-book data
our proposed SSCF model for collaborative filterings requires the measure of 5 matrices: a similarity matrix for the users, a similarity matrix for the items, and three recommendation matrices: one for the user-based approach, one for the item-based approach and one for the final SSCF model. If we deal with a big dataset, depending on the specifications of the computer, it could not be possible to compute a whole matrix. For instance our computer is not able to run the main.py code when using the amazon-book dataset.
We implemented a version main_light.py in which matrices computation is divided in 10 steps (the number of steps may be regulated changing the M1 variable in the code). in each step we compute only a fraction of the columns (or of the rows) of the matrices. using main_light.py our computer can run the SSCF model on amazon-book data.
> python main_light.py --dataset="amazon-book" --test="test" --gamma=0.9

# Rating prediction
We also provide an implementation to predict ratings with movielens data (movielens_rating folder)
