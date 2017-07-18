from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *

import csv
import StringIO

def transformQC(sc, sqlContext):
        lines = sc.textFile('hdfs:///user/w205/final/qc_results.csv')
        
	parts = lines.map(lambda l: list(csv.reader(StringIO.StringIO(l), delimiter=','))[0])
	
	rawData = parts.map(lambda t: (t[0], t[1], t[2], t[3], t[4], t[5]))

	# create dataframe
	schemaString = 'product_id qc_lot_number date x_value y_value z_value'
	fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
	schema = StructType(fields)

	schemaData = sqlContext.createDataFrame(rawData, schema)
	schemaData.registerTempTable('qc_results')

	results = sqlContext.sql('SELECT * FROM qc_results')
	results.show()

	results.rdd.saveAsTextFile('hdfs:///user/w205/final/qc_results.txt')
	return
