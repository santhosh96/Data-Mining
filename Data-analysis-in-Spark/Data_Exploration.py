from pyspark import SparkContext, SparkConf
from collections import OrderedDict
import multiprocessing

import sys
import json

# spark-submit santhosh_narayanan_task1.py ../data/review.json output/output_1.json

# Creating the configuration for context with app name as yelpAnalysis and master URL as local
conf = SparkConf().setAppName("yelpAnalysis").setMaster("local[*]")
# Creating the context object
sc = SparkContext(conf=conf)

# Dictionary for saving the output
result = OrderedDict()

# FilePath of the dataset
filePath = sys.argv[1]

# Reading the json file as text data
reviewData = sc.textFile(filePath)

# Count number of reviews
date = reviewData.map(json.loads).map(lambda review : review["date"]).persist()
result["n_review"] = date.count()

# Number of reviews given in year 2018
result["n_review_2018"] = date.map(lambda date : date.split(" ")[0] \
                              .split("-")[0]).filter(lambda year : year == '2018').count()

# Number of distinct users who wrote reviews
userId = reviewData.map(json.loads).map(lambda user : user["user_id"]).persist()
result["n_user"] = userId.distinct().count()

# Top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
top10Users = userId.map(lambda user : (user,1)) \
                   .reduceByKey(lambda u1, u2 : u1+u2) \
                   .sortByKey() \
                   .takeOrdered(10, key = lambda x: -x[1])

result["top10_user"] = [list(i) for i in top10Users]

# Number of distinct businesses that have been reviewed
businessId = reviewData.map(json.loads).map(lambda bussn : bussn["business_id"]).persist()
result["n_business"] = businessId.distinct().count()

# Top 10 businesses that had the largest numbers of reviews and the number of reviews they had
top10Business = businessId.map(lambda bussn : (bussn,1)) \
                                     .reduceByKey(lambda b1, b2 : b1+b2) \
                                     .sortByKey() \
                                     .takeOrdered(10, key = lambda x: -x[1])

result["top10_business"] = [list(i) for i in top10Business]

# Output file name for the result
outputPath = sys.argv[2]
# Creating the json file for the output
with open(outputPath, 'w') as fp:
    json.dump(result, fp)