from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *

import csv
import StringIO

def transformXbarm(sc, sqlContext):
	lines = sc.textFile('hdfs:///user/w205/final/xbarm.csv')

	parts = lines.map(lambda l: list(csv.reader(StringIO.StringIO(l), delimiter=','))[0])

	rawData = parts.map(lambda t: (t[0], t[1], t[2], t[3], t[4]))

	schemaString = 'product_id IPU_version x_xbarm y_xbarm z_xbarm'
	fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
	schema = StructType(fields)

	schemaData = sqlContext.createDataFrame(rawData, schema)
	schemaData.registerTempTable('xbarm')
	sqlContext.cacheTable('xbarm')

	results = sqlContext.sql('SELECT * FROM xbarm')
	results.show()

	results.rdd.saveAsTextFile('hdfs:///user/w205/final/xbarm.txt')
	return
