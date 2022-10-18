# SaplingSimilarity
This repository contains the code to build memory-based collaborative filtering using different similarity metrics and riproduce the results shown in https://arxiv.org/abs/2210.07039, where we introduce the Sapling Similarity. The python and library versions used to test this code are:
- python 3.9.1
- numpy 1.21.5
- pandas 1.2.0
- sklearn 0.24.0

similarity_metrics.py contains the code of the 12 similarity metrics used in the paper:
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

we provide three datasets to test the different similarity metrics:
- Export Data with the Revealed Comparative Advantage values of 169 countries in 5040 products
- Movielens Data with 5950 users and which of 2811 movies they like
- Amazon Data with 4881 users and 2686 products they wrote a good review 

build.py contains the code to build the collaborative filtering and test its performance.\\
To run the code simply write **python build.py**.\\
The code will ask you which dataset to use, which similarity metric and if it is a user-based or item-based collaborative filtering.\\
To use the code for the Amazon and Movielens dataset you have to run the create_test.py code first (**python create_edgelist.py**) in order to select which products or movies will be used for the test.
