from __future__ import print_function
from operator import add
from pyspark import SparkContext
import time
import os
import sys
import requests
import matplotlib.pyplot as plt
import numpy as np

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
                if(float(p[4]) >= 120 and float(p[4]) <= 3600):
                    if(float(p[11]) >= 3 and float(p[11]) <= 200):
                        if(float(p[5]) >= 1 and float(p[5]) <= 50):
                            if(float(p[15]) >= 3):
                                return p
    return 0
#
if __name__ == "__main__":
    #start spark
    sc = SparkContext.getOrCreate()
    #read file
    #file = sc.textFile("C:/Users/Yidow/Desktop/cs777-new/FengYiduo_Assignment_3/taxi-data-sorted-small.csv/taxi-data-sorted-small.csv")
    file = sc.textFile(sys.argv[1])
    #split line
    file = file.map(lambda line: line.split(",")).filter(correctRows)

    #linear coefficient
    file_linearR = file.map(lambda x: (float(x[5]), float(x[11])))
    file_linearR = file_linearR.map(lambda x: (x[0], x[1], x[0]*x[1], x[0]*x[0]))
    n = file_linearR.count()
    x_sum = file_linearR.flatMap(lambda x: (0,x[0])).reduce(add)
    y_sum = file_linearR.flatMap(lambda x: (0,x[1])).reduce(add)
    xy_sum = file_linearR.flatMap(lambda x: (0,x[2])).reduce(add)
    xx_sum = file_linearR.flatMap(lambda x: (0,x[3])).reduce(add)
    # print([x_sum,y_sum,xy_sum,xx_sum])

    m = (n*xy_sum - x_sum*y_sum)/(n*xx_sum-(x_sum**2))
    b = (xx_sum*y_sum - x_sum*xy_sum)/(n*xx_sum-(x_sum**2))
    re_1 = sc.parallelize(["m = ", m, "b = ", b]).coalesce(1).saveAsTextFile(sys.argv[2])

