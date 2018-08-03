#!/bin/bash

#################### kNN ####################
# local mode for knn
# spark-submit  \
#     --master local[4] \
#     --deploy-mode client \
#     --num-executors 2 \
#     --executor-cores 2 \
#     kNN_Implementation_final.py \
#     --input hdfs://localhost:9000/user/rzhu9225/ \
#     --d 50 \
#     --k 10

# cluster mode for knn
# spark-submit \
#   --master yarn \
#   --deploy-mode client \
#   --num-executors 5 \
#   --executor-cores 4 \
#   kNN_Implementation_final.py \
#   --input hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/ \
#   --d 50 \
#   --k 5


#################### Decision Tree ####################
# local mode for decision tree
# spark-submit  \
#     --master local[4] \
#     RandomForest.py \
#     --d 20

# cluster mode for decision tree
spark-submit \
 --master yarn \
 --deploy-mode client \
 --num-executors 5 \
 --executor-cores 4 \
 RandomForest.py  \
 --d 20


#################### Multilayer Perceptron ####################
# local mode for Multilayer Perceptron
# spark-submit  \
#     --master local[4] \
#     MultilayerPerceptron_final.py \
#     --input hdfs://localhost:9000/user/rzhu9225/ \
#     --size 100

# cluster mode for Multilayer Perceptron
# spark-submit \
#   --master yarn \
#   --deploy-mode client \
#   --num-executors 5 \
#   --executor-cores 4 \
#   MultilayerPerceptron_final.py  \
#   --input hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/ \
#   --size 100
