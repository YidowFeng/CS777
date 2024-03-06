from __future__ import print_function
from operator import add
import sys
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql.types import *
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
    #dataframe.createOrReplaceTempView("user_behavior")
    # dataframe = dataframe.withColumn("goodsId", dataframe["goodsId"].cast(IntegerType()))
    # dataframe = dataframe.withColumn("categoryId",
    #                                  dataframe["categoryId"].cast(IntegerType()))
    # dataframe = dataframe.withColumn("time",
    #                                  hour(dataframe["time"]).cast(IntegerType()))
    dataframe = spark.sql("select hour(timestamp) as hour, \
        float(categoryId), behaviorType from user_behavior")
    dataframe = dataframe.withColumn("behaviorType", F.when(dataframe["behaviorType"] == "pv", 0).\
                                     when(dataframe["behaviorType"] == "fav", 1).\
                                     when(dataframe["behaviorType"] == "cart", 1).\
                                     when(dataframe["behaviorType"] == "buy", 1))
    dataframe.show(3)
    assembler = VectorAssembler(
        inputCols=["categoryId", "hour"],
        outputCol="Features")

    output = assembler.transform(dataframe)
    output.select("Features").show()

    finalised_data = output.select("Features",
                                   "behaviorType")

    train_data, test_data = finalised_data.randomSplit([0.5, 0.5])

    regressor = LogisticRegression(featuresCol='Features',
                                 labelCol='behaviorType')
    regressor = regressor.fit(train_data)

    # pred_results = round(regressor.evaluate(test_data))
    # pred_results.predictions.show()
    lrn_summary = regressor.summary
    lrn_summary.predictions.show()
    #evaluate
    lrn_summary.predictions.describe().show()
    eval = BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                         labelCol="behaviorType")
    auc = eval.evaluate(lrn_summary.predictions)
    print("accuracy: ", auc)