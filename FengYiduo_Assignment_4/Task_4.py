import sys
import re
import numpy as np
from pyspark import SparkContext
from numpy import dot
from numpy.linalg import norm
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from operator import add
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
# In[2]:

spark = SparkSession.builder.getOrCreate()
dictionary_df = dictionary.toDF()
#allDocsAsNumpyArraysTFidf.map(lambda x:(x[0],x[1].tolist()))
#array_lst = allDocsAsNumpyArraysTFidf.map(lambda x: x[1]).collect()
allDocsAsNumpyArraysTFidf_col = StructType([
    StructField('name', StringType(), True),
    StructField('array', ArrayType(IntegerType()), True)
])
# featuresRDD_df = spark.createDataFrame(featuresRDD, schema = featuresRDD_col)
allDocsAsNumpyArraysTFidf_df = spark.createDataFrame(allDocsAsNumpyArraysTFidf, schema = allDocsAsNumpyArraysTFidf_col)

def getPrediction_df (textInput, k):
    # Create an dataframe out of the textIput
    data = textInput.split()
    myDoc = spark.createDataFrame(data, StringType())
    # Flat map the text to (word, 1) pair for each word in the doc
    wordsInThatDoc = myDoc.withColumn("count", lit(1))
    # This will give us a set of (word, (dictionaryPos, 1)) pairs
    allDictionaryWordsInThatDoc = dictionary_df.join (wordsInThatDoc, dictionary_df._1 == wordsInThatDoc.value,"inner")
    allDictionaryWordsInThatDoc = allDictionaryWordsInThatDoc.drop("value")
    allDictionaryWordsInThatDoc.show()
    # Get tf array for the input string
    #print(list(allDictionaryWordsInThatDoc.select("_2")))
    myArray = np.array(allDictionaryWordsInThatDoc.select("_2").collect())

    #Get the tf * idf array for the input string
    myArray = np.multiply (myArray, idfArray)
    array_lst = allDocsAsNumpyArraysTFidf.map(lambda x: x[1]).collect()
    #array_lst = np.array(array_lst)
    myArray = np.squeeze(np.asarray(myArray))
    array_lst = np.squeeze(np.asarray(array_lst))
    #print(featuresRDD_df.show(2))
    # Get the distance from the input text string to all database documents, using cosine similarity (np.dot() )
    #distances = featuresRDD_df.withColumn( "distance", cousinSim (featuresRDD_df.select("_2"), myArray) )
    # print(myArray)
    # print(array_lst)
    distances = allDocsAsNumpyArraysTFidf_df.withColumn( "distance", np.dot (array_lst, myArray) )
    # get the top k distances
    distances.orderby("array")
    topK = distances.head(k)

    # and transform the top k distances into a set of (docID, 1) pairs
    docIDRepresented = spark.createDataFrame(topK.docID, IntegerType())

    # now, for each docID, get the count of the number of times this document ID appeared in the top k
    numTimes = docIDRepresented.groupby(add)

    # Return the top 1 of them.
    # Ask yourself: Why we are using twice top() operation here?
    return numTimes.orderby(_2).head(k)




print(getPrediction_df('How many goals Vancouver score last year?', 10))