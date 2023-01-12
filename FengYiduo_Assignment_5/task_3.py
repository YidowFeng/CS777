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
    # split line
    file = file.map(lambda line: line.split(",")).filter(correctRows)
    #file_GD = file.map(lambda x: (float(x[4]), float(x[5]), float(x[11]), float(x[12]), float(x[16]))))
    file_GD = file.map(lambda x: (np.array([float(x[4]), float(x[5]), float(x[11]), float(x[12])]), float(x[16])))
    n = float(file_GD.count())
    learningRate = 0.0001
    numIteration = 100
    m = np.array([0.1, 0.1, 0.1, 0.1])
    b = 0.1
    cost = np.Inf
    # def cost():
    re_3 = []
    for i in range(numIteration):
        # Vectorizatoin
        file_GD = file_GD.map(lambda x: (x[0], (x[1] - (np.dot(x[0], m)+b))))
        cost_new = file_GD.flatMap(lambda x: (0, x[1]**2)).reduce(add)

        m_derivative = file_GD.flatMap(lambda x: (0, (-1)*x[0]*x[1])).reduce(add)
        m_derivative = (2/n) * m_derivative
        m = m - learningRate * m_derivative

        b_derivative = file_GD.flatMap(lambda x: (0, (-1)*x[1])).reduce(add)
        b_derivative = (2 / n) * b_derivative
        b = b - learningRate * b_derivative
        # bold driver update learning rate
        if(cost_new >= cost):
            learningRate = learningRate/2.0
        else:
            learningRate = learningRate * 1.01
        cost = cost_new
        print(i, " m=", m, " b=", b, " cost=", cost)
        re_3.append([i, " m=", m, " b=", b, " cost=", cost])
    task_3 = sc.parallelize(re_3).coalesce(1).saveAsTextFile(sys.argv[2])