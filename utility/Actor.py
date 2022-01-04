# In this file we will declare the actor-critic model to feed in the DDPG.
from keras.layers.core import Flatten
from tensorflow.keras import Sequential
from keras.layers import Dense,Input, Activation, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.models import Model
#import gym
import tensorflow as tf
from environment import Vasuki
import cv2

config = {'n': 8, 'rewards': {'Food': 4, 'Movement': -1, 'Illegal': -2}, 'game_length': 100} # Should not change for evaluation
env = Vasuki(**config)
#num_states = env.observation_space.shape[0]
#("Size of State Space ->  {}".format(num_states))
#num_actions = env.action_space.shape
#print("Size of Action Space ->  {}".format(num_actions))

#env.action_space.shape

upper_bound = 0   #env.action_space.high[0]
lower_bound = 3    #env.action_space.low[0]

class Actor():

    def __init__(self):
        self.state_dim = 5#num_states # in lunar lander we have 8 states
        self.action_dim = 1#num_actions   # in lunar lander we have 1 action.
        self.model = self.A_model()

    def A_model(self):
        
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """

        # print(f'shape of input in the Actor Network -> {self.state_dim} and for action -> {self.action_dim}')
        #print("Actor input dimensions: ",self.state_dim)
        inputs = Input(shape=self.state_dim)
        x = Dense(500, activation="relu")(inputs)  # or can do   x = Dense(500, activation='relu')
        x = Dense(100, activation="relu")(x)  # or can do   x = Dense(500, activation='relu')
        output = Dense(1, activation="relu")(x)  # or can do   x = Dense(500, activation='relu')

        # return output
        return Model(inputs, output)

    





