from pyspark import SparkContext, SparkConf
from collections import OrderedDict
import multiprocessing

import sys
import json
import time

# Creating the configuration for context with app name as yelpAnalysis and master URL as local
conf = SparkConf().setAppName("yelpAnalysis").setMaster("local[*]")
# Creating the context object
sc = SparkContext(conf=conf)

outputDict = OrderedDict()

# File path for review data
reviewFilePath = sys.argv[1]

# File path for business data
businessFilepath = sys.argv[2]


# Reading the dataset
reviewData = sc.textFile(reviewFilePath)
businessData = sc.textFile(businessFilepath)

# Mapping tuples of (business_id, city)
busCity = businessData.map(json.loads).map(lambda bussncity : (bussncity["business_id"],bussncity["city"])).persist()

# Mapping tuple of (business_id, (stars, 1))
busReview = reviewData.map(json.loads).map(lambda bussnReview : (bussnReview["business_id"], (float(bussnReview["stars"]), 1))).persist()

# Joined data for city and stars based on business_id
joinData = busCity.join(busReview).map(lambda x:(x[1][0],(float(x[1][1][0]), x[1][1][1])))

# Finding the average of stars with respect to city and 
result = joinData.reduceByKey(lambda rev1, rev2: (rev1[0] + rev2[0], rev1[1] + rev2[1])) \
                 .mapValues(lambda rev : rev[0] / rev[1]).sortByKey() \
                 .sortBy(lambda x : -x[1])

# Collecting the result RDD

start = time.time()

printData = result.collect()

for i in range(10):
    print(printData[i])

end = time.time()

outputDict["m1"] = end-start

# Taking only 10 rows from result RDD

start = time.time()

printDataTake = result.take(10)

for data in printDataTake:
    print(data)

end = time.time()

outputDict["m2"] = end-start

outputDict["explanation"] = "The collect method, collects all the data from the worker node and in this case performs sorting on all the data and then we print 10 result, while the take method with number of elements only picks (10 in this case) those elements and sorts them accordingly, hence this takes less time as compared to collect"

avgCityOutputPath = sys.argv[3]

f = open(avgCityOutputPath,"w+")
f.writelines("city,stars\n")

for tup in printData:
    f.writelines(str(tup[0])+","+str(tup[1])+'\n')

f.close()

# Output file name for the result
outputPath = sys.argv[4]
# Creating the json file for the output
with open(outputPath, 'w') as fp:
    json.dump(outputDict, fp)

# time spark-submit santhosh_narayanan_task3.py ../data/review.json ../data/business.json output/output3_1.txt output/output3_2.json