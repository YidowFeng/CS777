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

# sc = SparkContext(appName="TermProject-RFM")
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

R = "select userId, "\
    "datediff('2017-12-04', max(date)) as R, "\
    "dense_rank() over(order by datediff('2017-12-04', max(date))) as R_rank "\
    "from user_behavior "\
    "where behaviorType = 'buy' "\
    "group by userId " \
    "order by R_rank desc"

Rdata = spark.sql(R)
Rdata.show(10)
Rdata.write.option("header",True).csv("./Rdata")

F = \
    "select userId, "\
    "count(1) as F,"\
    "dense_rank() over(order by count(1) desc) as F_rank "\
    "from user_behavior "\
    "where behaviorType = 'buy' "\
    "group by userId "\
    "order by F_rank desc"

Fdata = spark.sql(F)
Fdata.show(10)
Fdata.write.option("header",True).csv("./Fdata")


RFM = "\
with cte as(\
select userId,\
       datediff('2017-12-04', max(date)) as R,\
       dense_rank() over(order by datediff('2017-12-04', max(date))) as R_rank,\
       count(1) as F,\
       dense_rank() over(order by count(1) desc) as F_rank \
from user_behavior \
where behaviorType = 'buy' \
group by userId) \
select userId, R, R_rank, R_score, F, F_rank, F_score,  R_score + F_score AS score \
from( \
select *, \
       case ntile(5) over(order by R_rank) when 1 then 5\
                                           when 2 then 4\
                                           when 3 then 3\
                                           when 4 then 2\
                                           when 5 then 1\
       end as R_score,\
       case ntile(5) over(order by F_rank) when 1 then 5\
                                           when 2 then 4\
                                           when 3 then 3\
                                           when 4 then 2\
                                           when 5 then 1\
       end as F_score \
from cte \
) as a \
order by score desc"

RFMdata = spark.sql(RFM)
RFMdata.show(10)
RFMdata.write.option("header",True).csv("./RFMdata")
