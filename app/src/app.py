#app.py

import matplotlib.dates as mdt
import math
import datetime as dt
import numpy as np 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
with warnings.catch_warnings():
  warnings.filterwarnings("ignore",category=FutureWarning)
  import tensorflow as tf 

import matplotlib.pyplot as plt 
import matplotlib.animation as manimation
from celluloid import Camera




print("##################")
print("##  Starting..  ##")
print("##################")

print(tf.config.experimental.list_physical_devices())

logdir = './tensorboard'
if tf.gfile.Exists(logdir):
  tf.io.gfile.rmtree(logdir)
tf.io.gfile.makedirs(logdir)

class PrimeiraLayDeNilton():
  def __init__(self, num_points, num_feat):
    fig = plt.figure()
    self.camera = Camera(fig)
    np.random.seed(101) 
    tf.compat.v1.set_random_seed(101) 

    self.num_feat = num_feat
    self.n = num_points # Number of data points 

    self.X = tf.compat.v1.placeholder(tf.float64, shape=(self.n,self.num_feat+1), name="X") 
    self.Y = tf.compat.v1.placeholder(tf.float64, shape=(self.n, 1), name="Y") 
    self.accumulated_weight = tf.compat.v1.placeholder(tf.float64, name ="acc_weight", shape=(self.num_feat+1, self.num_feat+1)) 
    self.iter_W  = tf.compat.v1.placeholder(tf.float64, name ="iter_W", shape=(self.num_feat+1, 1)) 
    #w = np.random.random((self.num_feat+1,1))
    w = np.ones((self.num_feat+1,1))
    with tf.name_scope('weights'):
      self.W = tf.Variable(initial_value=w, name = "W", shape=(self.num_feat+1, 1)) 
      self.variable_summaries(self.W)
      self.weight_summaries(self.W)
    
    self.merged = tf.compat.v1.summary.merge_all()
    self.learning_rate = 0.01
    self.training_epochs = 2050
    self.saver = tf.compat.v1.train.Saver()
    self.data_history = []
    self.weight_history= []

  def add_data_history(self, data):
    self.data_history.append(data)
    if len(self.data_history) > 10:
      self.data_history = self.data_history[-10:]

  def add_weight_history(self,weight):
    self.weight_history.append(weight)

  def variable_summaries(self, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.compat.v1.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.compat.v1.summary.scalar('stddev', stddev)
      tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
      tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
      tf.compat.v1.summary.histogram('histogram', var)

  def weight_summaries(self, var):
    with tf.name_scope('weights'):
      feats = self.num_feat+1
      for i in range(feats):
        base_vector = [0.]*feats
        base_vector[i]=1.
        projection = tf.matmul(tf.cast([base_vector],tf.float64),self.W)
        tf.compat.v1.summary.scalar("x{}".format(i),projection[0][0])


  def set_conditions(self):

    with tf.name_scope("acc_weight_update") as scope:
      #accumulated_weight update
      self.accumulated_weight_updater = tf.matmul(self.accumulated_weight, tf.linalg.tensor_diag(tf.reshape(tf.transpose(self.iter_W),[-1])), name="acc_weight_update")

    with tf.name_scope("pre_compute_x") as scope:
      #pre_compute_x
      self.pre_compute_x = tf.matmul(self.X, self.accumulated_weight, name="pre_compute_x")

    #self.weight_restriction = 
    #comput_prediction
    #self.compute_prediction = tf.matmul

    with tf.name_scope("Hypothesis") as scope:
      # Hypothesis 
      self.y_pred = tf.matmul(self.pre_compute_x, self.W, name='y_pred')

    with tf.name_scope("compute_cost") as scope:
      # Mean Squared Error Cost Function 
      self.cost = tf.reduce_sum(
                    tf.square(
                      tf.subtract(self.y_pred,self.Y)))

    with tf.name_scope("Adam_opt") as scope:
      # Gradient Descent Optimizer 
      self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate, name="Adam_opt").minimize(self.cost, name="minCost") 


  def start_calculations(self, new_data_set_count, step_limit):
    
    #iter_weight = np.ones((self.num_feat+1,1))
    accumulated_weight = np.eye(self.num_feat+1)
    change_sum = 0
    iter_index = -1
    #print("init_weight",iter_weight.T[0])

    self.set_conditions()

    # Global Variables Initializer 
    init = tf.compat.v1.global_variables_initializer() 

    # Starting the Tensorflow Session 
    with tf.compat.v1.Session() as sess: 
               
      # Initializing the Variables 
      sess.run(init) 
      summary_writer = tf.compat.v1.summary.FileWriter(logdir, sess.graph)
      
      for incoming_new_values in range(1,new_data_set_count+1):    
        iter_index +=1 
        if True:
          ################################################################
          # Genrating random linear data 
          # There will be n data points ranging from 0 to 50 
          x = []
          #x = [np.linspace(0, 10, self.n), np.linspace(0, 10, self.n)**2]
          for n_feat in range(1,self.num_feat+1):
            x.append(np.linspace(0, 10, self.n)**n_feat)
          change_rate = (incoming_new_values)**0.5
          #print("x0",x)
          if np.random.random()<0.9:
            change_rate *= 2  
          else:
            change_rate *=10
          if np.random.random()<0.22:
            change_sum += (incoming_new_values**0.5)*3
            
          y = [(np.random.uniform(-200, 200, self.n)*change_rate)+change_sum]
          y = [(x[0]+2*(x[1]))*change_rate]
          y += np.random.uniform(-200, 200, self.n)

          #print("rand_y",y)

          x = np.transpose(x)
          x = np.c_[x, np.ones(self.n)]
          #print("xT",x)
          y += np.random.uniform(0, 0, self.n) 
          y = y.T 
          self.add_data_history(y)
          #print("x.shape  \t=",x.shape)
          #print("y.shape \t=",y.shape)
          #print("W.shape \t=",self.W.shape)
          #print("y_pred.shape \t=",self.y_pred.shape)
          #print("Y.shape \t=",self.Y.shape)
          ################################################################

        # Using past t-1 predicion to fit the new data points        
        #accumulated_weight = np.matmul(accumulated_weight, np.diag(iter_weight.T[0]))
        #iter_weight = sess.run(self.W)
        #accumulated_weight = sess.run(self.accumulated_weight)
        #accumulated_weight = tf.matmul(accumulated_weight, tf.linalg.tensor_diag(iter_weight.T[0]), name="accumulated_weight")
        #accumulated_weight = sess.run(self.accumulated_weight_updater, feed_dict={self.accumulated_weight:accumulated_weight, self.iter_W:iter_weight})
        #self.accumulated_weight.assign(accumulated_weight)
        #print("accumulated_weight\n",accumulated_weight)
        #pre_compute_x =tf.matmul(x, accumulated_weight, name="pre_compute_x")
        
        #print("pre_compute_x\n",pre_compute_x)
        
        #print("accumulated_weight=\n",accumulated_weight, "\niter_weight=\n",iter_weight)
        #weight_past = weight
        #print("precomp=\n",pre_compute_x)

        # Iterating through all the epochs  
        for epoch in range(self.training_epochs): 
          
          # Feeding each data point into the optimizer using Feed Dictionary 
          sess.run(self.optimizer, feed_dict = {self.X : x, self.Y : y, self.accumulated_weight: accumulated_weight}) 

        # Storing necessary values to be used outside the Session 
        training_cost = sess.run(self.cost, feed_dict ={self.X: x, self.Y: y, self.accumulated_weight: accumulated_weight}) 
        iter_weight = sess.run(self.W) 

        centered_weights = iter_weight-np.ones((self.num_feat+1,1))
        step = np.linalg.norm(centered_weights)
        if step > step_limit and iter_index:
          original_weight = iter_weight
          correction_factor = step_limit/step
          iter_weight = (centered_weights*correction_factor)+np.ones((self.num_feat+1,1))
          print("Oops, this is running away! step:{}. correction_factor={}".format(step,correction_factor))
          print("original_weight={}\niter_weight=\t{}".format(original_weight.T,iter_weight.T))
        
        

        # Calculating the predictions 
        self.add_weight_history(iter_weight)
        accumulated_weight = sess.run(self.accumulated_weight_updater, feed_dict={self.accumulated_weight:accumulated_weight, self.iter_W:iter_weight})
        predictions = np.matmul(x,
                    np.matmul(accumulated_weight,
                        np.ones((self.num_feat+1,1))
                  ))
        print("============================================")
        print(">> Iter",iter_index,"Training cost =", training_cost)
        #print("predictions=",predictions.T, "\ny=", y.T)
        print(">> iter_weight=",iter_weight.T)
        print(">> accumulated_weight=\n",accumulated_weight)
        print("============================================")
        summary = sess.run(self.merged)
        summary_writer.add_summary(summary,iter_index)
        self.saver.save(sess,logdir+"/model.cktp",iter_index)

        idx = 0
        for data_points in self.data_history[::-1]:
          alpha = 1-(idx/10)
          idx += 1
          points, = plt.plot(x.T[0], data_points.T[0], 'ro', alpha=alpha)
        line, = plt.plot(x.T[0], predictions.T[0], 'b')
        plt.title('Linear Regression Result')
        plt.legend([points, line],('Original data', 'Fitted line'))
        #plt.show()
        self.camera.snap()
    
    self.camera.snap()
    animation = self.camera.animate(interval=10)
    an_writer = manimation.writers['ffmpeg']
    writer = an_writer(fps=2, metadata=dict(artist='Nilton Duarte'), bitrate=1000)
    animation.save('evo.mp4', writer=writer)
    animation.save('evo.gif', writer='imagemagick', fps=2)



calc = PrimeiraLayDeNilton(num_points=10, num_feat=3)
#calc.set_conditions()
calc.start_calculations(new_data_set_count=40, step_limit=5)