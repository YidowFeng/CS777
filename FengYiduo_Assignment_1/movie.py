from __future__ import print_function
from operator import add
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sc = SparkContext.getOrCreate()

ratings = sc.textFile('C:/Users/Yidow/Desktop/cs777-new/FengYiduo_Assignment_1/ml-10M100K/ratings.dat')
ratings = ratings.map(lambda line: line.split("::"))

rate_tmp = ratings.map(lambda x: (float(x[2]),1))
rate_tmp = rate_tmp.reduceByKey(add)
re = rate_tmp.collect()
indices = np.arange(len(re))
word, frequency = zip(*re)
plt.bar(indices, frequency, color='r')
plt.xticks(indices, word, rotation='vertical')
plt.tight_layout()
plt.show()

print(re)