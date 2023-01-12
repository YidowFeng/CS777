from __future__ import print_function
from operator import add
from pyspark import SparkContext
import time
import os
import sys
import requests
import matplotlib.pyplot as plt


#Exception Handling and removing wrong datalines
def isfloat(value):
    try:
        float(value)
        return True
    except:
         return False
def correctRows(p):
    if(len(p)==17):
        if(isfloat(p[5]) and isfloat(p[11])):
            if(float(p[5])!=0 and float(p[11])!= 0):
                return p
    return 0
if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    #read file
    file = sc.textFile(sys.argv[1])
    #split line
    file = file.map(lambda line: line.split(",")).filter(correctRows)

    re2 = file.filter(lambda x: x != '0')
    re2 = re2.map(lambda line: (line[1], float(line[16])/(float(line[4])/60) if float(line[4])/60 else 0))
    re2 = re2.mapValues(lambda x: (x, 1)).reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1]))
    re2 = re2.mapValues(lambda x: x[0] / x[1])
    re2 = re2.top(10, key=lambda p: p[1])
    task_2 = sc.parallelize(re2).coalesce(1).saveAsTextFile(sys.argv[2])
    print(re2)
    sc.stop()