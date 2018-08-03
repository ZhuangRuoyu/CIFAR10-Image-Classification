#!/usr/bin/python

"""
Random Forest in Spark ML library exploration.

In order to submit this to spark, please ssh to SIT
lab cluster and use bash command:

# cluster mode for decision tree
spark-submit \
 --master yarn \
 --deploy-mode client \
 --num-executors 5 \
 --executor-cores 4 \
 RandomForest.py  \
 --d 100
 --n 500
"""

# import libraries
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from time import time as runtime
import argparse
import pickle
from io import BytesIO

def load_data(filename):
    train_curr = spark.sparkContext.binaryFiles(filename)
    dfs = train_curr.values().map(lambda p: pickle.load(BytesIO(p), encoding='latin1'))
    test = dfs.collect()[0]
    image_rdd = spark.sparkContext.parallelize(zip(test['data'], test['labels']))
    image_df = image_rdd.map(lambda x: (Vectors.dense(x[0]), x[1])).toDF(['features', 'label'])
    return image_df

# main algorithm
if __name__ == "__main__":

    # input arguments needed
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", help="reduced dimension with PCA", default='100')
    parser.add_argument("--n", help="number of trees", default='100')
    args = parser.parse_args()
    d = int(args.d)
    n = int(args.n)

    # start Spark session
    spark = SparkSession \
        .builder \
        .appName("Random Forest with d=" + args.d + "n=" + args.n) \
        .getOrCreate()

    # load the data
    test_path = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/user/rzhu9225/cifar10/test_batch"
    train_path = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/user/rzhu9225/cifar10/data_batch_"

    t0 = runtime()
    test_df = load_data(test_path)
    for i in range(5):
        filename = train_path + str(i+1)
        image_df = load_data(filename)
        if i==0:
            train_df = image_df
        else:
            train_df = train_df.union(image_df)

    t1= runtime()
    print("Data loaded. Runtime: %.1fs" % (t1-t0))

    ##################### Preprocessing #####################
    # PCA
    pca = PCA(k=d, inputCol="features", outputCol="pca")

    ##################### Decision Tree #####################
    # Train a Random Forest model.
    rf = RandomForestClassifier(labelCol="label", featuresCol="pca", \
                                numTrees=n, seed=1234, maxDepth=30, \
                                minInstancePerNode=5)

    ##################### Pipelined Model #####################
    pipeline_rf = Pipeline(stages=[pca, rf])

    # build pipelined model with train data
    model_rf = pipeline_rf.fit(train_df)

    ##################### Prediction #####################
    # make predictions
    result_rf = model_rf.transform(test_df)

    ##################### Evaluation #####################
    # compute accuracy
    evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(result_rf)

    print("\n+-------------------+")
    print("| Accuracy = %.2f%% |" % (100*accuracy))
    print("+-------------------+\n")
