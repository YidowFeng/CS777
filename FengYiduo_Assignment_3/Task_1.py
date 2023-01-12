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
    #start spark
    sc = SparkContext.getOrCreate()
    #read file
    file = sc.textFile(sys.argv[1])
    #split line
    file = file.map(lambda line: line.split(",")).filter(correctRows)
    #medallion and hack license

    ids = file.map(lambda x:((x[0],x[1]))).distinct()
    ids = ids.map(lambda line: (line[0],1))
    re1 = ids.reduceByKey(add).top(10,lambda x:x[1])
    task_1  = sc.parallelize(re1).coalesce(1).saveAsTextFile(sys.argv[2])
    print(re1)
    sc.stop()
