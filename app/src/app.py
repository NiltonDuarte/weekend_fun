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




class PrimeiraLayDeNilton():
  def __init__(self, num_points, num_feat):
    fig = plt.figure()
    self.camera = Camera(fig)
    np.random.seed(101) 
    tf.compat.v1.set_random_seed(101) 

    self.num_feat = num_feat
    self.n = num_points # Number of data points 

    self.X = tf.compat.v1.placeholder(tf.float64, shape=(self.n,self.num_feat+1)) 
    self.Y = tf.compat.v1.placeholder(tf.float64, shape=(self.n, 1)) 

    w = np.random.random((self.num_feat+1,1))
    self.W = tf.Variable(initial_value=w, name = "W", shape=(self.num_feat+1, 1)) 

    self.learning_rate = 0.01
    self.training_epochs = 150
    self.data_history = []

  def add_data_history(self, data):
    self.data_history.append(data)
    if len(self.data_history) > 10:
      self.data_history = self.data_history[-10:]

  def set_conditions(self):
    # Hypothesis 
    self.y_pred = tf.matmul(self.X, self.W)

    # Mean Squared Error Cost Function 
    self.cost = tf.reduce_sum(
       tf.square(self.y_pred-self.Y)
      )

    # Gradient Descent Optimizer 
    self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.cost) 

  def start_calculations(self, new_data_set_count, step_limit):
    
    self.set_conditions()
    # Genrating random linear data 
    # There will be n data points ranging from 0 to 50 
    x = [np.linspace(0, 50, self.n), np.linspace(0, 50, self.n)]
    y = [np.linspace(0, 50, self.n)] 

    # Adding noise to the random linear data 
    x[0] += np.random.uniform(-4, 4, self.n)
    x[1] += np.random.uniform(-4, 4, self.n)
    x = np.transpose(x)
    x = np.c_[x, np.ones(self.n)]
    y += np.random.uniform(-0, 0, self.n) 
    y = y.T
    self.add_data_history(y)
    print("x.shape  \t=",x.shape)
    print("y.shape \t=",y.shape)
    print("W.shape \t=",self.W.shape)
    print("y_pred.shape \t=",self.y_pred.shape)
    print("Y.shape \t=",self.Y.shape)
    print("diff     \t=",(self.y_pred-self.Y).shape)

    # Global Variables Initializer 
    init = tf.compat.v1.global_variables_initializer() 

    # Starting the Tensorflow Session 
    with tf.compat.v1.Session() as sess: 
      
      ### First round
      # Initializing the Variables 
      sess.run(init) 
      for epoch in range(self.training_epochs): 
        
        # Feeding each data point into the optimizer using Feed Dictionary 
        # for (_x, _y) in zip(x, y):
          
        #   _x = np.array([_x])
        #   _y = np.array([_y])
          #print("zip",_x, _y)
        sess.run(self.optimizer, feed_dict = {self.X : x, self.Y : y}) 

        # Displaying the result after every 50 epochs 
        if (epoch) % 50 == 0: 
          # Calculating the cost a every epoch 
          c = sess.run(self.cost, feed_dict = {self.X : x, self.Y : y})
          print("Epoch", (epoch), ": cost =", c, "\nW =\n", sess.run(self.W)) 
          print("============================================")

      # Storing necessary values to be used outside the Session 
      training_cost = sess.run(self.cost, feed_dict ={self.X: x, self.Y: y})
      weight = sess.run(self.W) 
      print("============================================")
      print("Training cost =", training_cost, "Weight =\n", weight,'\n') 
      print("============================================")
      ### Iterate round
    

    w = np.ones((self.num_feat+1,1))
    self.W = tf.Variable(initial_value=w, name = "W", shape=(self.num_feat+1, 1)) 

    self.set_conditions()
    init = tf.compat.v1.global_variables_initializer() 

    # Starting the Tensorflow Session 
    with tf.compat.v1.Session() as sess: 
      # Initializing the Variables 
      sess.run(init) 

      for incoming_new_values in range(1,new_data_set_count):
        # Genrating random linear data 
        # There will be n data points ranging from 0 to 50 
        x = [np.linspace(0, 20, self.n), np.linspace(0, 30, self.n)]
        change_rate = (incoming_new_values)**0.5
        if np.random.random()<0.9:
          change_rate *= 1  
        else:
          change_rate *=10
        y = [np.linspace(0, 50, self.n)*change_rate] 

        # Adding noise to the random linear data 
        x[0] += np.random.uniform(-0, 0, self.n)
        x[1] += np.random.uniform(-0, 0, self.n)
        x = np.transpose(x)
        x = np.c_[x, np.ones(self.n)]
        y += np.random.uniform(-32, 32, self.n) 
        y = y.T 
        self.add_data_history(y)
        # Using past t-1 predicion to fit the new data points
        predictions_t1 =np.matmul(x, np.diag(weight.T[0]))
        #print("preds=\n",predictions_t1)

        # Iterating through all the epochs  
        for epoch in range(self.training_epochs): 
          
          # Feeding each data point into the optimizer using Feed Dictionary 
          sess.run(self.optimizer, feed_dict = {self.X : predictions_t1, self.Y : y}) 

          # Displaying the result after every 50 epochs 
          if (epoch) % 50 == 0: 
            # Calculating the cost a every epoch 
            c = sess.run(self.cost, feed_dict = {self.X : predictions_t1, self.Y : y}) 
            #print("Epoch", (epoch), ": cost =", c, "\nW =\n", sess.run(self.W)) 
            #print("============================================")

        # Storing necessary values to be used outside the Session 
        training_cost = sess.run(self.cost, feed_dict ={self.X: predictions_t1, self.Y: y}) 
        weight = sess.run(self.W) 

        centered_weights = weight-np.ones((self.num_feat+1,1))
        step = np.linalg.norm(centered_weights)
        if step > step_limit:
          original_weight = weight
          correction_factor = step_limit/step
          weight = (centered_weights*correction_factor)+np.ones((self.num_feat+1,1))
          print("Oops, this is running away! step:{}. correction_factor={}".format(step,correction_factor))
          print("original_weight=\n{}\nnew_weight=\n{}".format(original_weight,weight))
        

        # Calculating the predictions 
        predictions = np.matmul(predictions_t1,weight)
        print("============================================")
        print("Training cost =", training_cost) 
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
    writer = an_writer(fps=2, metadata=dict(artist='Nilton Duarte'), bitrate=1000)
    animation.save('evo.mp4', writer=writer)



calc = PrimeiraLayDeNilton(30,2)
#calc.set_conditions()
calc.start_calculations(20, 1.5)