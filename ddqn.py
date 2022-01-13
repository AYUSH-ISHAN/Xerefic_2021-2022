import gym_robot
import gym
import numpy as np
import random
from gym import error, spaces
from gym import utils
from gym import spaces, logger
from gym.utils import seeding
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from environment import Vasuki
#%matplotlib inline
import matplotlib.pyplot as plt
from collections import deque
import sys
import cv2
import time
#Number of episodes
episodes=200

class DQN():
    def __init__(self):
        self.load_model=False
        self.action_size=3
        self.state_size= 5
        self.discount_factor=0.99
        self.learning_rate=0.001
        #Exploration Exploitation trade off
        self.epsilon=1.0
        self.epsilon_decay=0.999
        self.epsilon_min=0.01

        self.train_start = 1000

        self.batch_size = 64
        #Build models
        self.model=self.build_model()
        self.target_model=self.build_model()
        #set target models parameters to parameters of model
        self.target_model.set_weights(self.model.get_weights())
        #episode memeory
        self.memory = deque(maxlen=2000)
        if self.load_model:
            self.epsilon=0.0001
            self.model.load_weights('dqn.h5')

    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    # Policy
    def get_action(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.float32(state).reshape((1,5))
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        #target = self.model.predict(state)[0]
        batch_size=self.batch_size
        mini_batch = random.sample(self.memory, batch_size)
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []
        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])
        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)
        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

if __name__ == '__main__' :
    config = {'n': 8, 'rewards': {'Food': 4, 'Movement': -1, 'Illegal': -2}, 'game_length': 100}
    env = Vasuki(**config)
    agentA=DQN()
    agentB=DQN()
    plt.ion()
    scores, episode = [], []
    for e in range(episodes):
        done=False
        score = 0
        env.reset()
        stateA = env.agentA
        stateB = env.agentB
        scoreA=0
        scoreB=0
        while not done:
            StateA=[]
            StateA.append(stateA['head'])
            StateA.append(stateA['score'])
            StateA.append(stateA['state'][0])
            StateA.append(stateA['state'][1])
            StateA.append(stateA['velocity'])
            #print(StateB)
            StateB=[]
            StateB.append(stateB['head'])
            StateB.append(stateB['score'])
            StateB.append(stateB['state'][0])
            StateB.append(stateB['state'][1])
            StateB.append(stateB['velocity'])

            actionA=agentA.get_action(StateA)
            actionB=agentB.get_action(StateB)

            rewardA, rewardB, done, info=env.step({'actionA':actionA,'actionB':actionB})
            scoreA+=rewardA
            scoreB+=rewardB
            stateA=info['agentA']
            stateB=info['agentB']

            next_StateA=[]
            next_StateA.append(stateA['head'])
            next_StateA.append(stateA['score'])
            next_StateA.append(stateA['state'][0])
            next_StateA.append(stateA['state'][1])
            next_StateA.append(stateA['velocity'])
            next_StateB=[]
            next_StateB.append(stateB['head'])
            next_StateB.append(stateB['score'])
            next_StateB.append(stateB['state'][0])
            next_StateB.append(stateB['state'][1])
            next_StateB.append(stateB['velocity'])

            agentA.append_sample(StateA, actionA, rewardA, next_StateA, done)
            agentB.append_sample(StateB, actionB, rewardB, next_StateB, done)

            agentA.train_model()
            agentB.train_model()
            if done:
                #set target models parameters to parameters of model
                agentA.target_model.set_weights(agentA.model.get_weights())
                agentB.target_model.set_weights(agentB.model.get_weights())

                print("episode:", e, "scoreA:", scoreA, "scoreB:",scoreB,"  memory length:",
                      len(agentA.memory), "  epsilon:", agentA.epsilon)
        #save weights
        agentA.model.save_weights('dqn1.h5')
        agentB.model.save_weights('dqn2.h5')
