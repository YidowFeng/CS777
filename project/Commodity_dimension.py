import os
import sys
import requests
from operator import add

from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sc = SparkContext(appName="TermProject-dimension")
spark = SparkSession.builder.getOrCreate()

newDF = [StructField('userId', StringType()),
         StructField('goodsId', StringType()),
         StructField('categoryId', StringType()),
         StructField('behaviorType', StringType()),
         StructField('timeStamp', IntegerType()),
         StructField('date', StringType()),
         StructField('time', StringType())]
finalStruct = StructType(fields=newDF)
dataframe = spark.read.csv('UserBehavior.csv', schema=finalStruct)
dataframe.registerTempTable("user_behavior")

salemostgoods = dataframe\
    .withColumn("pv", when(dataframe.behaviorType == "pv", 1).otherwise(0)) \
    .withColumn("fav", when(dataframe.behaviorType == "fav", 1).otherwise(0))\
    .withColumn("cart", when(dataframe.behaviorType == "cart", 1).otherwise(0))\
    .withColumn("buy", when(dataframe.behaviorType == "buy", 1).otherwise(0))
# salemostgoods.printSchema()
salemostgoods = salemostgoods.groupBy('goodsId').agg({'pv': 'sum','fav': 'sum','cart': 'sum','buy': 'sum'}).sort('sum(buy)',ascending=False)
salemostgoods.show(10)
salemostgoods.write.option("header", True).csv("./salemostgoods")

salemostcategorysql = \
    "select categoryId, "\
    "sum(case when behaviorType = 'pv' then 1 else 0 end) as pv, "\
    "sum(case when behaviorType = 'fav' then 1 else 0 end) as fav, "\
    "sum(case when behaviorType = 'cart' then 1 else 0 end) as cart, "\
    "sum(case when behaviorType = 'buy' then 1 else 0 end) as buy "\
    "from user_behavior "\
    "group by categoryId "\
    "order by buy desc "

salemostcategory = spark.sql(salemostcategorysql)
salemostcategory.show(10)
salemostcategory.write.option("header", True).csv("./salemostcategory")
