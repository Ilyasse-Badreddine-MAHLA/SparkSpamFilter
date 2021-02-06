from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from read import readData
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


spark = SparkSession.builder.appName('nlp').getOrCreate()

data = readData()
data.show()

data = data.withColumn('length',length(data['text']))
data.show()
data.groupby('class').mean().show()


tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
ham_spam_to_num = StringIndexer(inputCol='class',outputCol='label')

clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')

nb = NaiveBayes()
data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
cleaner = data_prep_pipe.fit(data)
clean_data = cleaner.transform(data)
clean_data = clean_data.select(['label','features'])
clean_data.show()
(training,testing) = clean_data.randomSplit([0.7,0.3])
spam_predictor = nb.fit(training)
data.printSchema()
test_results = spam_predictor.transform(testing)
test_results.show()

acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting spam was: {}".format(acc))



data_array =  np.array(test_results.select("prediction").collect())
data_array1 =  np.array(test_results.select("label").collect())
names = ['spam prediction','spam', 'ham prediction','ham']
values = [np.count_nonzero(data_array == 1), np.count_nonzero(data_array1 == 1),np.count_nonzero(data_array == 0), np.count_nonzero(data_array1 == 0)]
plt.figure(figsize=(18, 8))
plt.subplot(131)
plt.bar(names, values)
plt.ylabel('total')
plt.suptitle('Filtrage des emails')
# plt.show()

print(confusion_matrix(data_array,data_array1))
cf_matrix=confusion_matrix(data_array,data_array1)

sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')

print("done.")