from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from collections import OrderedDict
from pyspark.sql.functions import desc

conf = SparkConf().setAppName("yelpAnalysis").setMaster("local[*]")
sc = SparkContext(conf=conf)

result = OrderedDict()

df = sc.read.json("../data/review.json")

result["n_review"] = df.count()

result["n_review_2018"] = df.filter("date like '2018%'").count()

result["n_user"] = df.select("user_id").distinct().count()

result["top10_user"] = df.groupBy("user_id").count().sort(desc("count")).show(10)

result["n_business"] = df.select("business_id").distinct().count()





