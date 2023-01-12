from __future__ import print_function
from operator import add
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sc = SparkContext.getOrCreate()

def parseLine(line):
    field = line.split(',')
    pos = field[1]
    follower = field[2]
    following = field[3]
    return (pos, follower, following)
line = sc.textFile('C:/Users/Yidow/Desktop/cs777-new/InstagramUserStats.csv')
rdd = line.map(parseLine)

fler_avg = rdd.map(lambda x: (int(int(x[0])/100), int(x[1])))
fler_avg = fler_avg.mapValues(lambda x: (x,1)).reduceByKey(lambda x,y:(x[0]+y[0], x[1]+y[1]))
fler_avg = fler_avg.mapValues(lambda x: x[0]/x[1])
print(fler_avg.collect())

fling_avg = rdd.map(lambda x: (int(int(x[0])/100), int(x[2])))
fling_avg = fling_avg.mapValues(lambda x: (x,1)).reduceByKey(lambda x,y:(x[0]+y[0], x[1]+y[1]))
fling_avg = fling_avg.mapValues(lambda x: x[0]/x[1])
print(fler_avg.collect())