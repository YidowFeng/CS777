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
    #file = sc.textFile("C:/Users/Yidow/Desktop/cs777-new/FengYiduo_Assignmrnt_3/taxi-data-sorted-small.csv/taxi-data-sorted-small.csv")
    #split line
    file = file.map(lambda line: line.split(",")).filter(correctRows)
    re3 = file.filter(lambda x: x != '0')
    re3 = file.map(lambda x: (x[2].split(" ")[1].split(":")[0], float(x[12])/float(x[5])))
    re3 = re3.mapValues(lambda x: (x,1)).reduceByKey(lambda x,y:(x[0]+y[0], x[1]+y[1]))
    re3 = re3.mapValues(lambda x: x[0]/x[1])
    re3 = re3.top(1, key=lambda p: p[1])
    task_3 = sc.parallelize(re3).coalesce(1).saveAsTextFile(sys.argv[2])
    # print(re3)
    print(re3)
    sc.stop()