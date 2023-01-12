from __future__ import print_function
import random
from pyspark import SparkContext
import numpy as np
from pyspark.sql import SparkSession

l = []
sample_size = 1000
#create random points
for i in range(sample_size):
    a = random.random()
    b = random.random()
    l.append([a, b])
spark = SparkSession.builder.getOrCreate()
#count the points
rdd = spark.sparkContext.parallelize(l)
rdd = rdd.filter(lambda x: x[0]*x[0]+x[1]*x[1] < 1).count()

#because random.random() only return positive number, so we only count the
#points in ((0,0),(1,1)), so now we need to times 4
print(4.0 * (rdd / sample_size))