from pyspark import SparkContext, SparkConf
from collections import OrderedDict
import multiprocessing

import sys
import json
import time

# spark-submit santhosh_narayanan_task2.py ../data/review.json output/output2.json 8

# Creating the configuration for context with app name as yelpAnalysis and master URL as local
conf = SparkConf().setAppName("yelpPartitionAnalysis").setMaster("local[*]")
# Creating the context object
sc = SparkContext(conf=conf)

# Dictionary for saving the output
result = OrderedDict()

result["default"] = OrderedDict()
result["customized"] = OrderedDict()
result["explanation"] = "Since the dataset is not too large, we can have the data split into partitions equivalent to number of cores (8 in my case) in the system, and each cores can execute the whole chunk. If the number of partitions is more (160 created by default in my machine), there is an extra time required to move the data from local disk to main memory an process"

# FilePath of the dataset
filePath = sys.argv[1]

# USING DEFAULT PARTITION

# Reading the json file as text data

reviewData = sc.textFile(filePath)

result["default"]["n_partition"] = reviewData.getNumPartitions()
result["default"]["n_items"] = reviewData.mapPartitions(lambda length : [sum(1 for _ in length)]).collect()

start = time.time()

# Map task
businessId = reviewData.map(json.loads).map(lambda bussn : (bussn["business_id"],1)).persist()

# Reduce task
businessIdTopReviews = businessId.reduceByKey(lambda a, b: a+b).takeOrdered(10, key = lambda x: -x[1])

end = time.time()

result["default"]["exe_time"] = end - start

del reviewData

# USING CUSTOM PARTITION

reviewData = sc.textFile(filePath).repartition(int(sys.argv[3]))

result["customized"]["n_partition"] = reviewData.getNumPartitions()
result["customized"]["n_items"] = reviewData.mapPartitions(lambda length : [sum(1 for _ in length)]).collect()

start_c = time.time()

# Map task
businessId = reviewData.map(json.loads).map(lambda bussn : (bussn["business_id"],1)).persist()

# Reduce task
businessIdTopReviews = businessId.reduceByKey(lambda a, b: a+b).takeOrdered(10, key = lambda x: -x[1])

end_c = time.time()

result["customized"]["exe_time"] = end_c - start_c

# Output file name for the result
outputPath = sys.argv[2]

# Creating the json file for the output
with open(outputPath, 'w') as fp:
    json.dump(result, fp)


