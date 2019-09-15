from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from collections import OrderedDict
import sys
import json

# Setting the python for driver and executor
# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3'

# Creating the configuration for context with app name as yelpAnalysis and master URL as local
conf = SparkConf().setAppName("yelpAnalysis").setMaster("local[*]")
# Creating the context object
sc = SparkContext(conf=conf)

# Dictionary for saving the output
result = OrderedDict()

# FilePath of the dataset
filePath = sys.argv[1]

print("file path : ",filePath)

# Reading the json file as text data
reviewData = sc.textFile(filePath)

# Count number of reviews
result["n_review"] = reviewData.count()
print("count : ", result["n_review"])

# Number of reviews given in year 2018
# result["n_review_2018"] = reviewData.map(lambda review : review.replace("\"","").split("date:")[1]).filter(lambda year : year[:4] == "2018").count()

result["n_review_2018"] = reviewData.map(lambda review : review[1:-1].split(",")[-1]
                                                        .split(" ")[0]
                                                        .split(":")[1]
                                                        .split("-")[0].replace("\"", "")).filter(lambda year : year[:4] == "2018").count()


# Generating (user_id, 1) tuples
userId = reviewData.map(lambda review : (review[1:-1].replace("\"","").split("user_id:")[1].split(",")[0],1))

# Number of distinct users who wrote reviews
result["n_user"] = userId.distinct().count()

# Top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
userIdTopReviews = userId.reduceByKey(lambda a, b: a+b).takeOrdered(10, key = lambda x: -x[1])
result["top10_user"] = [list(i) for i in userIdTopReviews]

# Generating (business_id, 1) tuples
businessId = reviewData.map(lambda review : (review[1:-1].replace("\"","").split("business_id:")[1].split(",")[0],1))

# Number of distinct businesses that have been reviewed
result["n_business"] = businessId.distinct().count()

# Top 10 businesses that had the largest numbers of reviews and the number of reviews they had
businessIdTopReviews = businessId.reduceByKey(lambda a, b: a+b).takeOrdered(10, key = lambda x: -x[1])
result["top10_business"] = [list(i) for i in businessIdTopReviews]


# Output file name for the result
outputPath = sys.argv[2]
# Creating the json file for the output
with open(outputPath, 'w') as fp:
    json.dump(result, fp)





