from pyspark import SparkConf
from pyspark.sql import SparkSession

spark = SparkSession \
        .builder \
        .appName('MyApp') \
        .config('spark.mongodb.input.uri', 'mongodb://127.0.0.1/SparkSpamFilter.emails')\
        .config('spark.mongodb.output.uri', 'mongodb://127.0.0.1/SparkSpamFilter.emails')\
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:2.4.2') \
        .getOrCreate()


def readData ():
        df = spark.read.format('com.mongodb.spark.sql.DefaultSource').load()
        return df.drop('_id')



# data = readData()
# data.show()




