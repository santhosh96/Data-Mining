from pyspark import SparkContext, SparkConf
from collections import OrderedDict
import multiprocessing
import pandas as pd

import sys
import json
import time

# Creating the configuration for context with app name as yelpAnalysis and master URL as local
conf = SparkConf().setAppName("yelpPartitionAnalysisVariedPartitions").setMaster("local[*]")
# Creating the context object
sc = SparkContext(conf=conf)

# Dictionary for saving the output
result = OrderedDict()

# FilePath of the dataset
filePath = sys.argv[1]

data = []

# Reading the json file as text data
# reviewData = sc.textFile(filePath).repartition(8)

for i in range(1,41):

    partitions = 8 * i

    reviewData = sc.textFile(filePath).repartition(partitions)

    result[i] = OrderedDict()

    n_partition = reviewData.getNumPartitions()
    result[i]["n_partition"] = reviewData.getNumPartitions() 

    n_items = reviewData.mapPartitions(lambda length : [sum(1 for _ in length)]).collect()
    result[i]["n_items"] = n_items

    start = time.time()

    # Map task
    businessId = reviewData.map(lambda review : (review[1:-1].replace("\"","").split("business_id:")[1].split(",")[0],1))

    # Reduce task
    businessIdTopReviews = businessId.reduceByKey(lambda a, b: a+b).takeOrdered(10, key = lambda x: -x[1])

    end = time.time()
    exe_time = end - start
    result[i]["exe_time"] = exe_time

    data.append((n_partition, n_items, exe_time))

    del reviewData

df = pd.DataFrame(data, columns = ["partitionCount", "itemCount", "execTime"])

df.to_csv("analysis.csv")

# Output file name for the result
outputPath = sys.argv[2]
# Creating the json file for the output
with open(outputPath, 'w') as fp:
    json.dump(result, fp)


