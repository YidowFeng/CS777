from __future__ import print_function
from operator import add
from pyspark import SparkContext
import sys
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #start spark
    spark = SparkSession.builder.getOrCreate()
    #read file
    #dataframe = spark.read.csv("C:/Users/Yidow/Desktop/cs777-new/project/UserBehaviorSmall.csv.bz2", header='true')
    dataframe = spark.read.csv(
        sys.argv[1],
        header='true')
    #split line
    dataframe.show(3)
    dataframe.createOrReplaceTempView("user_behavior")
    df_hr = spark.sql("select hour(timestamp) as hour, \
    sum(case when behaviorType = 'pv' then 1 else 0 end) as pv, \
    sum(case when behaviorType = 'fav' then 1 else 0 end) as fav,\
       sum(case when behaviorType = 'cart' then 1 else 0 end) as cart,\
       sum(case when behaviorType = 'buy' then 1 else 0 end) as buy \
       from user_behavior group by hour(timestamp) order by hour")
    df_hr.show(3)
    pdf_hr = (df_hr.toPandas())
    pdf_hr.plot()
    plt.savefig('df_hr.pdf')

    df_week = spark.sql("select pmod(datediff(timestamp, '1920-01-01') - 3, 7) as weekday,\
       sum(case when behaviorType = 'pv' then 1 else 0 end) as pv, \
       sum(case when behaviorType = 'fav' then 1 else 0 end) as fav, \
       sum(case when behaviorType = 'cart' then 1 else 0 end) as cart,\
       sum(case when behaviorType = 'buy' then 1 else 0 end) as buy \
        from user_behavior\
        where date(timestamp) between '2017-11-27' and '2017-12-03'\
        group by pmod(datediff(timestamp, '1920-01-01') - 3, 7)\
        order by weekday")
    df_week.show(3)
    pdf_week = (df_week.toPandas())
    pdf_week.plot.bar(rot=0)
    plt.savefig('df_week_hist.pdf')



