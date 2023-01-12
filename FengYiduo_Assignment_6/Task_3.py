from __future__ import print_function

import re
import sys
import numpy as np
from operator import add

from pyspark import SparkContext

def freqArray (listOfIndices, numberofwords):
	returnVal = np.zeros (20000)
	for index in listOfIndices:
		returnVal[index] = returnVal[index] + 1
	returnVal = np.divide(returnVal, numberofwords)
	return returnVal


def buildArray(listOfIndices):
	returnVal = np.zeros(20000)

	for index in listOfIndices:
		returnVal[index] = returnVal[index] + 1

	mysum = np.sum(returnVal)

	returnVal = np.divide(returnVal, mysum)

	return returnVal


def build_zero_one_array(listOfIndices):
	returnVal = np.zeros(20000)

	for index in listOfIndices:
		if returnVal[index] == 0: returnVal[index] = 1

	return returnVal


def stringVector(x):
	returnVal = str(x[0])
	for j in x[1]:
		returnVal += ',' + str(j)
	return returnVal


def cousinSim(x, y):
	normA = np.linalg.norm(x)
	normB = np.linalg.norm(y)
	return np.dot(x, y) / (normA * normB)

def sigmod(x):
	return 1 / (1 + np.exp(-x))


if __name__ == "__main__":

	sc = SparkContext(appName="LogisticRegression")
	# d_corpus = sc.textFile(sys.argv[1], 1)
	d_corpus = sc.textFile(
		"C:/Users/Yidow/Desktop/cs777-new/FengYiduo_Assignment_6/SmallTrainingData.txt",
		1)
	d_keyAndText = d_corpus.map(lambda x: (
	x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))
	regex = re.compile('[^a-zA-Z]')

	d_keyAndListOfWords = d_keyAndText.map(
		lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	# print(d_keyAndListOfWords.take(1))
	# [('AU35', ['consideration', 'of', 'an', 'application'

	# -----------------------------------------------

	# Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
	# to ("word1", 1) ("word2", 1)...
	allWords = d_keyAndListOfWords.flatMap(lambda x: x[1])
	allWords = allWords.map(lambda x: (x, 1))

	# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
	allCounts = allWords.reduceByKey(add)

	# Get the top 20,000 words in a local array in a sorted format based on frequency
	# If you want to run it on your laptio, it may a longer time for top 20k words.
	topWords = allCounts.top(20000, key=lambda x: x[1])

	#
	# print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))

	# We'll create a RDD that has a set of (word, dictNum) pairs
	# start by creating an RDD that has the number 0 through 20000
	# 20000 is the number of words that will be in our dictionary
	topWordsK = sc.parallelize(range(20000))

	# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
	# ("NextMostCommon", 2), ...
	# the number will be the spot in the dictionary used to tell us
	# where the word is located
	dictionary = topWordsK.map(lambda x: (topWords[x][0], x))
	# print("dictionary")
	# print(dictionary.take(2))
	dictionary_tmp = dictionary.collectAsMap()
	# ——————————————————————————————————————————————————————————————————————————
	# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
	# ("word1", docID), ("word2", docId), ...
	allWordsWithDocID = d_keyAndListOfWords.flatMap(
		lambda x: ((j, x[0]) for j in x[1]))
	# print(allWordsWithDocID.take(2))
	# [('consideration', 'AU35'), ('of', 'AU35')]

	# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
	allDictionaryWords = allWordsWithDocID.join(dictionary)
	# allDictionaryWords = allWordsWithDocID.join(dictionary)
	# print(allDictionaryWords.take(2))
	# 934380

	# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
	justDocAndPos = allDictionaryWords.map(lambda x: x[1])
	# print(justDocAndPos.take(2))
	# [('431949', 1), ('431949', 1)]

	# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
	# print(allDictionaryWordsInEachDoc.take(2))

	# The following line this gets us a set of
	# (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	# and converts the dictionary positions to a bag-of-words numpy array...
	# ....................................
	allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(
		lambda x: (x[0], buildArray(x[1])))
	# print(allDocsAsNumpyArrays.take(2))

	# Now, create a version of allDocsAsNumpyArrays where, in the array,
	# every entry is either zero or one.
	# A zero means that the word does not occur,
	# and a one means that it does.

	zeroOrOne = allDictionaryWordsInEachDoc.mapValues(build_zero_one_array)
	# print(zeroOrOne.take(2))
	# Now, add up all of those arrays into a single array, where the
	# i^th entry tells us how many
	# individual documents the i^th word in the dictionary appeared in
	dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
	numberOfDocs = d_keyAndListOfWords.count()
	# Create an array of 20,000 entries, each entry with the value numberOfDocs (number of docs)
	multiplier = np.full(20000, numberOfDocs)

	# Get the version of dfArray where the i^th entry is the inverse-document frequency for the
	# i^th word in the corpus
	idfArray = np.log(np.divide(np.full(20000, numberOfDocs), dfArray))

	# Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
	allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(
		lambda x: (x[0], np.multiply(x[1], idfArray)))
	# print("TF-idf matrix")
	# print(allDocsAsNumpyArraysTFidf.take(2))
	# #

	# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs

	# ----------------------------------------------------------------

	train = allDocsAsNumpyArrays.map(
		lambda x: (1 if x[0][:2] == 'AU' else 0, np.append(x[1], 1)))
	train.cache()
	sizeofLine = train.count()

	cost = 0
	costList = []
	coefficient_size = len(train.take(1)[0][1])
	coefficient = np.zeros(coefficient_size)
	max_iteration = 100
	learning_rate = 29.75
	regularization = 0.0002475

	for i in range(max_iteration):
		cur_coef = coefficient

		pred = train.sample(False, 256 / sizeofLine, 1 + i)

		pred = pred.treeAggregate((np.zeros(coefficient_size), 0, 0),
								  lambda x, y: (x[0] + y[1] * (-y[0] + (
											  1 - 1 / (np.exp(
										  np.dot(y[1], cur_coef)) + 1))),
												x[1] + y[0] * (-(np.dot(y[1],
																		cur_coef))) + np.log(
													1 + np.exp(np.dot(y[1],
																	  cur_coef))),
												x[2] + 1), add)
		gradient = (1.0 / pred[2]) * pred[0] + 2 * regularization * coefficient
		coefficient = coefficient - learning_rate * gradient

		curr_cost = pred[1] + regularization * (np.square(coefficient).sum())
		cost = curr_cost
		costList.append(curr_cost)
		print(i, " cost: ", curr_cost)


	#--------------------------------------------------------------------------------
	# d_corpus = sc.textFile(sys.argv[1], 1)
	test_corpus = sc.textFile(
		"C:/Users/Yidow/Desktop/cs777-new/FengYiduo_Assignment_6/TestingData.txt",
		1)
	test_keyAndText = test_corpus.map(lambda x: (
		x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))
	regex = re.compile('[^a-zA-Z]')

	test_keyAndListOfWords = test_keyAndText.map(
		lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	# Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
	# to ("word1", 1) ("word2", 1)...
	test_allWords = test_keyAndListOfWords.flatMap(lambda x: x[1])
	test_allWords = test_allWords.map(lambda x: (x, 1))

	# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
	test_allCounts = test_allWords.reduceByKey(add)

	# Get the top 20,000 words in a local array in a sorted format based on frequency
	# If you want to run it on your laptio, it may a longer time for top 20k words.
	test_topWords = test_allCounts.top(20000, key=lambda x: x[1])

	#
	# print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))

	# We'll create a RDD that has a set of (word, dictNum) pairs
	# start by creating an RDD that has the number 0 through 20000
	# 20000 is the number of words that will be in our dictionary
	test_topWordsK = sc.parallelize(range(20000))

	# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
	# ("NextMostCommon", 2), ...
	# the number will be the spot in the dictionary used to tell us
	# where the word is located
	test_dictionary = test_topWordsK.map(lambda x: (topWords[x][0], x))
	# print("dictionary")
	# print(dictionary.take(2))
	test_dictionary_tmp = test_dictionary.collectAsMap()
	# ——————————————————————————————————————————————————————————————————————————
	# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
	# ("word1", docID), ("word2", docId), ...
	test_allWordsWithDocID = test_keyAndListOfWords.flatMap(
		lambda x: ((j, x[0]) for j in x[1]))
	# print(allWordsWithDocID.take(2))
	# [('consideration', 'AU35'), ('of', 'AU35')]

	# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
	test_allDictionaryWords = test_allWordsWithDocID.join(test_dictionary)
	# allDictionaryWords = allWordsWithDocID.join(dictionary)
	# print(allDictionaryWords.take(2))
	# 934380

	# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
	test_justDocAndPos = test_allDictionaryWords.map(lambda x: x[1])
	# print(justDocAndPos.take(2))
	# [('431949', 1), ('431949', 1)]

	# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	test_allDictionaryWordsInEachDoc = test_justDocAndPos.groupByKey()
	# print(allDictionaryWordsInEachDoc.take(2))

	# The following line this gets us a set of
	# (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	# and converts the dictionary positions to a bag-of-words numpy array...
	# ....................................
	test_allDocsAsNumpyArrays = test_allDictionaryWordsInEachDoc.map(
		lambda x: (x[0], buildArray(x[1])))

	test = allDocsAsNumpyArrays.map(
		lambda x: (x[0], 1 if x[0][:2] == 'AU' else 0, x[1]))
	test.cache()

	test_re = test.map(lambda x: (x[0], x[1], (np.dot(x[2], coefficient[:-1]))))
	test_re = test_re.map(lambda x: (x[0], x[1], (x[2] > 0).astype(int)))

	TP = test_re.filter(lambda x: x[1] == 1 and x[2] == 1).count()
	FP = test_re.filter(lambda x: x[1] == 0 and x[2] == 1).count()
	FN = test_re.filter(lambda x: x[1] == 1 and x[2] == 0).count()
	TN = test_re.filter(lambda x: x[1] == 0 and x[2] == 0).count()

	F1 = TP / (TP + ((FP + FN)/2.00))

	print("TP: ", TP)
	print("FP: ", FP)
	print("FN: ", FN)
	print("TN: ", TN)
	print("F1: ", F1)