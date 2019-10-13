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

    w = np.random.random((self.num_feat+1,1))
    self.W = tf.Variable(initial_value=w, name = "W", shape=(self.num_feat+1, 1)) 

    self.learning_rate = 0.01
    self.training_epochs = 2050
    self.data_history = []

  def add_data_history(self, data):
    self.data_history.append(data)
    if len(self.data_history) > 10:
      self.data_history = self.data_history[-10:]

  def add_weight_history(self,weight):
    pass

  def set_conditions(self):
    # Hypothesis 
    self.y_pred = tf.matmul(self.X, self.W, name='y_pred')

    # Mean Squared Error Cost Function 
    self.cost = tf.reduce_sum(
       tf.square(self.y_pred-self.Y)
      , name='cost')

    # Gradient Descent Optimizer 
    self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate, name="AdamOpt").minimize(self.cost) 

  def start_calculations(self, new_data_set_count, step_limit):
    
    iter_weight = np.random.random((self.num_feat+1,1))
    accumulated_weight = np.eye(self.num_feat+1)
    change_sum = 0
    iter_index = 0
    for incoming_new_values in range(1,new_data_set_count):
      iter_index +=1 
      w = np.ones((self.num_feat+1,1))
      self.W = tf.Variable(initial_value=w, name = "W", shape=(self.num_feat+1, 1)) 
      self.set_conditions()

      # Global Variables Initializer 
      init = tf.compat.v1.global_variables_initializer() 

      # Starting the Tensorflow Session 
      with tf.compat.v1.Session() as sess: 
        # Initializing the Variables 
        sess.run(init) 
        summary_writer = tf.compat.v1.summary.FileWriter('./tensorboard', sess.graph)
        # Genrating random linear data 
        # There will be n data points ranging from 0 to 50 
        x = [np.linspace(0, 20, self.n), np.linspace(0, 30, self.n)]
        change_rate = (incoming_new_values)**0.5
        if np.random.random()<0.9:
          change_rate *= 2  
        else:
          change_rate *=10
        if np.random.random()<0.22:
          change_sum += (incoming_new_values**0.5)*3
          
        y = [(np.random.uniform(-200, 200, self.n)*change_rate)+change_sum]

        # Adding noise to the random linear data 
        #x[0] += np.random.uniform(-0, 0, self.n)
        #x[1] += np.random.uniform(-0, 0, self.n)
        x = np.transpose(x)
        x = np.c_[x, np.ones(self.n)]
        y += np.random.uniform(0, 0, self.n) 
        y = y.T 
        self.add_data_history(y)
        #print("x.shape  \t=",x.shape)
        #print("y.shape \t=",y.shape)
        #print("W.shape \t=",self.W.shape)
        #print("y_pred.shape \t=",self.y_pred.shape)
        #print("Y.shape \t=",self.Y.shape)
        #print("diff     \t=",(self.y_pred-self.Y).shape)

        # Using past t-1 predicion to fit the new data points        
        accumulated_weight = np.matmul(accumulated_weight, np.diag(iter_weight.T[0]))
        pre_compute_x =np.matmul(x, accumulated_weight)
        
        #print("accumulated_weight=\n",accumulated_weight, "\niter_weight=\n",iter_weight)
        #weight_past = weight
        #print("precomp=\n",pre_compute_x)

        # Iterating through all the epochs  
        for epoch in range(self.training_epochs): 
          
          # Feeding each data point into the optimizer using Feed Dictionary 
          sess.run(self.optimizer, feed_dict = {self.X : pre_compute_x, self.Y : y}) 

        # Storing necessary values to be used outside the Session 
        training_cost = sess.run(self.cost, feed_dict ={self.X: pre_compute_x, self.Y: y}) 
        iter_weight = sess.run(self.W) 

        centered_weights = iter_weight-np.ones((self.num_feat+1,1))
        step = np.linalg.norm(centered_weights)
        if step > step_limit and iter_index:
          original_weight = iter_weight
          correction_factor = step_limit/step
          iter_weight = (centered_weights*correction_factor)+np.ones((self.num_feat+1,1))
          print("Oops, this is running away! step:{}. correction_factor={}".format(step,correction_factor))
          print("original_weight=\n{}\niter_weight=\n{}".format(original_weight,iter_weight))
        
        

        # Calculating the predictions 
        predictions = np.matmul(pre_compute_x,iter_weight)
        print("============================================")
        print("== Training cost =", training_cost)
        #print("predictions=",predictions.T, "\ny=", y.T, "\nweight=\n",iter_weight.T)
        print("============================================")

        # Plotting the Results 
        #print("x=",x.T[0],"\ny=", y.T[0], "\nŷ=",predictions.T[0])
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
    writer = an_writer(fps=4, metadata=dict(artist='Nilton Duarte'), bitrate=1000)
    animation.save('evo.mp4', writer=writer)
    animation.save('evo.gif', writer='imagemagick', fps=4)



calc = PrimeiraLayDeNilton(20,2)
#calc.set_conditions()
calc.start_calculations(10, 0)