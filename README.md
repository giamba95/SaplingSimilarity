# Sapling Similarity and SSCF
This repository contains the implementation for our paper:
> Sapling Similarity Collaborative Filtering: a memory-based approach with high recommendation quality https://arxiv.org/abs/2210.07039

where introduce Sapling Similarity and SSCF.
The python and library versions used to test this code are:
- python 3.9.1
- numpy 1.21.5
- pandas 1.2.0
- sklearn 0.24.0

similarities.py contains the code of the 12 similarity metrics used in the paper:
- Common Neighbors
- Jaccard Index
- Adamic/Adar
- Preferential Attachment
- Resource Allocation Index
- Cosine Similarity
- Sorensen Index
- Hub Depressed Index
- Hub Promoted Index
- Taxonomy Network
- Probabilistic Spreading
- Sapling Similarity

main.py contains the code to reproduce the results in our paper.
To run the code simply write **python build.py**.
The code will ask you which dataset to use, which similarity metric and if it is a user-based or item-based collaborative filtering.
To use the code for the Amazon and Movielens dataset you have to run the create_test.py code first (**python create_test.py**) in order to select which products or movies will be used for the test.
