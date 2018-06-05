# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:00:56 2018

@author: sarac
"""
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import LSTM


class DQNAgent:
    def __init__(self, state_size):
        self.memory = deque(maxlen=4000)
        self.learning_rate = 0.001
        self.state_size = state_size
        self.epsilon = 1.0  # exploration rate
        self.gamma = 0.95   # discount rate 
        self.predict_size = 6
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        
    
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(LSTM(30, return_sequences = True, input_shape = (self.state_size,1), activation='relu'))
        model.add(LSTM(18, return_sequences = True,  activation='relu'))
        model.add(LSTM(6, activation='relu'))
        model.add(Dense(self.predict_size, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, pred_prices, next_state, done):
        self.memory.append((state, pred_prices, next_state, done))
   
    def act(self, state, train = True):

        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) #return action 
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        if not done : 
            for state, pred_prices, next_state, done in minibatch:
                target = self.model.predict(state)
                curr_price = state[0][self.state_size-1]
                for i in range(self.predict_size,1):
                    fut_price = next_state[0][self.state_size-i]
                    reward = self.reward_estimator()
                    print('day : ' , i, 'accuracy : ', reward)
                    target[0][i] = reward    
                self.model.fit(state, target, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min : 
            self.epsilon *= self.epsilon_decay
        
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
    def reward_estimator(self, last_price, pred, target):
        reward = 0 
        real_trend = target - last_price
        pred_trend = pred - last_price
        err_trend = abs(real-trend - pred_trend)/abs(real_trend)
        if pred_trend*real_trend > 0 :
            reward = 1-err_trend  
        else : 
            reward = -(1-err_trend)
        return reward * 100