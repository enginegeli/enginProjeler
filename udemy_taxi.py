#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 12:49:17 2021

@author: enginegeli
"""


# kütüphanelerimizi import edelim 

import gym

import numpy as np

import random

import matplotlib.pyplot as plt


# environmentimizi yaratalım


env=gym.make("Taxi-v3").env







#Q-table

q_table=np.zeros([env.observation_space.n,env.action_space.n])





# Hyperparameter



alpha=0.1

gamma=0.9

epsilon=0.1



# Plotting metrix

# her bir adımda kazandığı ödülü reward liste  ekleyelim

reward_list=[]

dropout_list=[]



episode_number=100000



for i in range(1,episode_number):

   

    #initialize environment

   

    state=env.reset()

    reward_count=0

    droput_count=0

   

   

    while True:

       

       

        # exploit vs explore to find action

        # %10=explore, %90=exploit

        if random.uniform(0,1) < epsilon:

            action=env.action_space.sample()

       

        else:

            action=np.argmax(q_table[state])

       

       

        # action process and take reward/observation

        next_state,reward,done,_=env.step(action) # step metodu action'ımızı gerçekleştirmemizi sağlar

       

               

        # Q-learning Function

        old_value=q_table[state,action]

        next_max=np.max(q_table[next_state])

       

        next_value=(1-alpha)*old_value+alpha*(reward+gamma*next_max)

               

       

        # Q table Update

        q_table[state,action]=next_value

       

        # Update State

        state=next_state

       

        # Find wrong dropouts

        if reward == -10:

            droput_count +=1

           

       

        if done: break

       

        reward_count+=reward

   

    if i%100 ==0:

       

        dropout_list.append(droput_count)

        reward_list.append(reward_count)

        print("Episode:{}, Reward:{}, wrong dropout:{}".format(i,reward_count,droput_count))
        env.render()

   
#%% visualize

fig,axs = plt.subplots(1,2)

axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")

axs[1].plot(dropout_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("dropouts")

axs[0].grid(True)
axs[1].grid(True)

plt.show()

#%%  q table
"""
Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
"""
env.s = env.encode(0,0,3,4)

env.render( )





#%%

env.s = env.encode(4,4,4,3)
# en mantıklı hareket sola gidip yolcuyu bırakması ve rewardı en yuksek olan da o cıkıyor
env.render( )



