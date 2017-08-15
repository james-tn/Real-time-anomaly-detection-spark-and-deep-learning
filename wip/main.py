import numpy as np
import pandas as pd
import tensorflow as tf

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure, output_file, show
from bokeh.models import Span

from tensorflow.python.framework import random_seed

<<<<<<< HEAD
=======
#import findspark
#findspark.init()
>>>>>>> origin/master
import pyspark
from pyspark.sql.functions import *

from pyspark import SparkContext
sc = SparkContext()

from pyspark.sql import SparkSession

<<<<<<< HEAD

spark = SparkSession \
    .builder \
    .appName("W205 Final bokeh") \
    .config("spark.driver.memory", "10g") \
    .enableHiveSupport() \
    .getOrCreate()


qc_sql = spark.sql("select * from qc_data")
=======
spark = SparkSession.builder.appName("W205 Final").config("spark.some.config.option", "some-value").getOrCreate()

data_path = 'C:/Temp/w205finalprojdata/'

qc_raw = spark.read.csv(data_path + 'Specific QC Lot Number Results.csv', header =False)
qc_raw = qc_raw.select(col("_c0").alias("Prod_ID"), col("_c1").alias("QC_Lot"), col("_c2").alias("QC_Time"),col("_c3").alias("W-X"), col("_c4").alias("W-Y"),col("_c5").alias("W-Z") )
qc_raw.write.saveAsTable("QC_Raw")

machine_raw = spark.read.csv(data_path + 'Reference Set A Result Set.csv')
machine_raw = machine_raw.select(col("_c0").alias("Prod_ID"), col("_c1").alias("Ser_No"), col("_c2").alias("Model_ID"),col("_c3").alias("IPU_Ver"), col("_c4").alias("Upgrade_Date") )
machine_raw.write.saveAsTable("Machine_Raw")
>>>>>>> origin/master




qc_data =qc_sql.toPandas()

qc_data = qc_data[((qc_data['W-X']!='0')&(qc_data['W-Y']!='0')&(qc_data['W-Z']!='0'))]
qc_data['Prod_ID'] = qc_data['Prod_ID'].astype(str)
qc_data['W-X'] = qc_data['W-X'].astype(int)
qc_data['W-Y'] = qc_data['W-Y'].astype(int)
qc_data['W-Z'] = qc_data['W-Z'].astype(int)
qc_data['QC_Time'] = pd.to_datetime(qc_data['QC_Time'])
qc_data[['Ser_No', 'Model_ID']] = qc_data[['Ser_No', 'Model_ID']].bfill()
qc_data['IPU_Ver'] = qc_data['IPU_Ver'].fillna(value ='unknown_ver')

transformed_qc_data = pd.get_dummies(qc_data[['QC_Lot', 'W-X', 'W-Y', 'W-Z', 'IPU_Ver', 'Model_ID']],columns =['IPU_Ver', 'Model_ID', 'QC_Lot']).values
train_data = transformed_qc_data[:2000000]


queue =[]        
sess = tf.Session()
new_saver = tf.train.import_meta_graph('sysmex_anomaly_model1.meta')

#new_saver.restore(sess,tf.train.latest_checkpoint('./'))

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
outputs = tf.get_collection("outputs")[0]


def predict(act_w, input_stream, threshold):

    y_pred = sess.run(outputs, feed_dict={X: input_stream})
    print("ypred shape", y_pred.shape)
    pred = y_pred.reshape(50,-1)[-1]
    print("pred is", pred)
    print("act_w is", act_w)
    var = np.sum(np.abs(act_w-pred))/3
    flag=var>threshold
    
    return pred, flag

def gen_data(num=1, threshold=500):
    global queue
    X_act, act_w = mock_data.next_batch(num)
    act_w = act_w.flatten()
    queue.append(X_act.flatten())
    pred_data=[]
    flag = None
    if(len(queue)>50):
        queue = queue[1:]
    if(len(queue)==50):
        pred_data,flag = predict(act_w, np.array(queue)[np.newaxis, :,:],threshold)
    return pred_data, act_w, flag


class DataSet(object):

  def __init__(self,
               data,
               indices,
               n_steps,
               fea_size,
               seed=None):
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    self._num_examples = int(len(indices)/n_steps)
    self._indices = indices
#     self._labels = labels
    self._n_steps = n_steps
    self._data = data

    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._fea_size = fea_size
  @property
  def indices(self):
    return self._indices

  @property
  def fea_size(self):
    return self._fea_size

  @property
  def n_steps(self):
    return self._n_steps

  @property
  def num_examples(self):
    return self._num_examples
  @property
  def data(self):
    return self._data


  @property
  def epochs_completed(self):
    return self._epochs_completed


  def load(self, indices=[], batch_size=1):
    n_events= np.zeros(shape=(batch_size,self._n_steps,self._fea_size))
    m_events= np.zeros(shape=(batch_size,self._n_steps,3))

    
    for i,j in zip(indices,range(batch_size)):
#         print("start is", start)
#         print("end is", end)
#         print("i is", i)
#         print("")

        n_event = self._data[i*self._n_steps:(i+1)*self._n_steps]
        n_events[j] = np.array(n_event)
        m_event = self._data[i*self._n_steps+1:(i+1)*self._n_steps+1,0:3]
        m_events[j] = np.array(m_event)
#         print('n event shape is', n_event.shape)
#         print('m event shape is', m_event.shape)

#         print('n event is', n_event)
#         print('m event is', m_event)
#     print('n event shape', n_events.shape)
#     print('m event shape', m_events.shape)
#     print('n events is', n_events)
#     print('m events is', m_events)

    return n_events, m_events
   

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    if self._epochs_completed == 0 and start == 0:
      perm0 = np.arange(self._num_examples)
      self._indices = self.indices[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      events_rest_part = self._indices[start:self._num_examples]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      events_new_part = self._indices[start:end]

      return self.load(np.concatenate((events_rest_part, events_new_part), axis =0), batch_size)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
        
#     n_events= np.zeros(shape=(batch_size,self._n_steps,41), dtype=np.uint8)
#     m_events= np.zeros(shape=(batch_size,self._n_steps,3), dtype=np.uint8)

    
#     for i,j in zip(range(start,end),range(batch_size)):
# #         print("start is", start)
# #         print("end is", end)
# #         print("i is", i)
# #         print("")

#         n_event = self._data[i*self._n_steps:(i+1)*self._n_steps]
#         n_events[j] = np.array(n_event)
#         m_event = self._data[i*self._n_steps+1:(i+1)*self._n_steps+1,0:3]
#         m_events[j] = np.array(m_event)
    return self.load((start,end), batch_size)

mock_data = DataSet(train_data,np.arange(0,2000000), 1,41)

my_data = dict(x=[], y1=[], y2=[], y3=[],y4=[], y5=[], y6=[])
source = ColumnDataSource(my_data)

fig = Figure()
fig.line(source=source, x="x", y="y1", line_width=2, alpha=.85, color="red", legend="W-X actual")
fig.line(source=source, x="x", y="y2", line_width=2, alpha=.85, color="blue",  legend="W-Y actual")
fig.line(source=source, x="x", y="y3", line_width=2, alpha=.85, color="green",  legend="W-Z actual")

fig.line(source=source, x="x", y="y4", line_width=2,line_dash='dashed', alpha=.85, color="red", legend="W-X norm")
fig.line(source=source, x="x", y="y5", line_width=2,line_dash='dashed', alpha=.85, color="blue",  legend="W-Y norm")
fig.line(source=source, x="x", y="y6", line_width=2,line_dash='dashed', alpha=.85, color="green" , legend="W-Z norm")


ct = 0
w_x = 0
w_y = 0
w_z = 0

w_x_p = 0
w_y_p = 0
w_z_p = 0

def update_data():
    global ct, w_x, w_y, w_z, w_x_p, w_y_p, w_z_p
    ct += 1
    pred_data, act_w, flag = gen_data(1,200)
    
    w_x = act_w[0]
    w_y = act_w[1]
    w_z = act_w[2]

    if(len(pred_data)> 0):
        w_x_p = pred_data[0]
        w_y_p = pred_data[1]
        w_z_p = pred_data[2]
    if(flag):
        anomaly = Span(location=ct,dimension='height', line_color='red',line_dash='dashed', line_width=3)
        fig.add_layout(anomaly)
        
    new_data = dict(x=[ct], y1=[w_x], y2=[w_y], y3=[w_z], y4=[w_x_p], y5=[w_y_p], y6=[w_z_p])
    source.stream(new_data, 200)

curdoc().add_root(fig)
curdoc().add_periodic_callback(update_data, 80)