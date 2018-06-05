# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:11:15 2018

@author: sarac
"""

from PredictAgent import DQNAgent
import datetime
import random 
import numpy as np
import pandas as pd 

EPISODES = 300
MARGIN = 1000

start_index = 45    #2010.01.01 00:00
end_index = 3161+1  #2011.12.30 20:00
dataset = pd.read_csv('EURUSD_4H.csv')
train_data = dataset.iloc[start_index:end_index,5:6]

train_data = np.array(train_data)
state_size = 60
X_train = [] 
all_index = end_index-start_index
for i in range(state_size, all_index):
    X_train.append(train_data[i-state_size:i,0])
X_train = np.array(X_train)


class TrainEnvironment:
    def __init__(self, data, num_index):
        self.train_data = data
        self.train_index = 0 
        self.end_index = num_index-1
        self.discount_rate = 0.95

    def reset(self):
        self.train_index = 0         
        return [self.train_data[self.train_index]]
    
    def done_check(self):
        if self.train_index + 1 == self.end_index :
            return True
        else :
            return False
        
    def step(self):
        skip = 6
        self.train_index += skip
        if self.train_index >= self.end_index-1 : 
            self.train_index = self.end_index-1 
        ns = [self.train_data[self.train_index]]
        done = self.done_check()
        return ns, done

#########################################################################################################
# Train     
#########################################################################################################         
def watch_result(episode ,s_time, e_time, c_index, all_index, action, reward, profit):
    print('-------------------- Check -------------------------')
    print('start time: ' + s_time)  
    print('counter : ', c_index,'/', all_index,' of episode : ', episode, '/', EPISODES)
    print('action : ', action)
    print('current profit : ', profit*MARGIN)
    print('reward (all profit): ', reward)
    print('end_time: ' + e_time)
    print('-------------------End Check -----------------------')

    
if __name__ == "__main__":
    
    agent = DQNAgent(state_size)
    #agent.load("agent_model.h5")
    num_index = all_index - state_size
    env = TrainEnvironment(X_train, num_index)
    batch_size = 20 
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, (1, state_size, 1))
        
        for t in range(end_index-start_index):
            start_time = str(datetime.datetime.now().time())
            pred_prices = agent.act(state)
            next_state, done = env.step()
            next_state = np.reshape(next_state, (1,state_size,1))
            agent.remember(state, pred_prices, next_state, done)
            print('state : ', state)
            print('pred_price :', pred_prices)
            state = next_state       
            if done:
                agent.update_target_model()
                print('----------------------------- Episode Result -----------------------')
                print("episode: {}/{}, time: {}, e: {:.2}"
                      .format(e, EPISODES, t, agent.epsilon))
                print('----------------------------- End Episode --------------------------')
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            end_time = str(datetime.datetime.now().time())
             
                     
    agent.save("agent_model.h5")
                      
    