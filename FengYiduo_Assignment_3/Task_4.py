from __future__ import print_function
from operator import add
from pyspark import SparkContext
import time
import os
import sys
import requests
import matplotlib.pyplot as plt

# Exception Handling and removing wrong data lines
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
    ###---------------------------------------------------------------
    #What percentage of taxi customers pay with cash and what percentage use electronic cards?
    ###-----------------------------------------------------------------
    sc = SparkContext.getOrCreate()
    #read file
    file = sc.textFile(sys.argv[1])
    #file = sc.textFile("C:/Users/Yidow/Desktop/cs777-new/FengYiduo_Assignmrnt_3/taxi-data-sorted-small.csv/taxi-data-sorted-small.csv")
    #split line
    file = file.map(lambda line: line.split(",")).filter(correctRows)
    tmp = file.map(lambda x: (x[2].split(" ")[1].split(":")[0], x[10]))

    re4_csh = tmp.filter(lambda x:x[1] == "CSH")
    re4_csh = re4_csh.map(lambda x: (x[0], 1)).reduceByKey(add)

    re4_crd = tmp.filter(lambda x:x[1] == "CRD")
    re4_crd = re4_crd.map(lambda x: (x[0], 1)).reduceByKey(add)

    re4_1 = re4_csh.join(re4_crd)
    re4_1 = re4_1.map(lambda x: (x[0], 0 if len(x[1]) == 1 else x[1][1]/(x[1][0]+x[1][1])))
    task_4_1 = re4_1.coalesce(1).saveAsTextFile(sys.argv[2])
    print(re4_1.collect())

    ###---------------------------------------------------------------
    # top-10 efficient taxi divers
    ###-----------------------------------------------------------------
    tmp_2 = tmp = file.map(lambda x: (x[1], (float(x[12])+float(x[14]))/float(x[5])))
    re4_2 = tmp_2.mapValues(lambda x: (x, 1)).reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1]))
    re4_2 = re4_2.mapValues(lambda x: x[0] / x[1])
    re4_2 = re4_2.top(10, key=lambda p: p[1])
    task_4_2 = sc.parallelize(re4_2).coalesce(1).saveAsTextFile(sys.argv[3])
    print(re4_2)

    sc.stop()





