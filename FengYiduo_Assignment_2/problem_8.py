from __future__ import print_function
from pyspark import SparkContext
#--------------------------------------------------
#Assume that the order of column is [”rownumber, first name",
#"last name", "course", "score"]
#--------------------------------------------------
sc = SparkContext.getOrCreate()
lines = sc.textFile("C:/Users/Yidow/Desktop/cs777-new/students.csv")
lines = lines.map(lambda line: line.split(","))
print(lines.collect())

#1 Min grade of each student
min_grade = lines.map(lambda x: (x[1]+x[2], x[4])).reduceByKey(min)
print(min_grade.collect())
#2 Max grade of each student
max_grade = lines.map(lambda x: (x[1]+x[2], x[4])).reduceByKey(max)
print(max_grade.collect())
#3 GPA
GPA = lines.map(lambda x: (x[1]+x[2], float(x[4])))
GPA = GPA.mapValues(lambda v: (v, 1)).reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])).mapValues(lambda v: v[0]/v[1])
print(GPA.collect())
#4 Number of courses taken
course = lines.map(lambda x: (x[1]+x[2], 1)).reduceByKey(lambda a, b: a+b)
print(course.collect())