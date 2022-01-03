# In this file we will declare the actor-critic model to feed in the DDPG.
from keras.layers.core import Flatten
from tensorflow.keras import Sequential
from keras.layers import Dense,Input, Activation, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.models import Model
import gym
import tensorflow as tf

env = gym.make('LunarLander-v2')
num_states = env.observation_space.shape[0]
num_actions = 1 #env.action_space.shape

upper_bound = 1   #env.action_space.high[0]
lower_bound = -1    #env.action_space.low[0]

class Actor():

    def __init__(self):
        self.state_dim = num_states # in lunar lander we have 8 states
        self.action_dim = num_actions   # in lunar lander we have 1 action.
        self.model = self.A_model()

    def A_model(self):
        
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """

        # print(f'shape of input in the Actor Network -> {self.state_dim} and for action -> {self.action_dim}')

        inputs = Input(shape=self.state_dim)
        x = Dense(500, activation="relu")(inputs)  # or can do   x = Dense(500, activation='relu')
        x = Dense(100, activation="relu")(x)  # or can do   x = Dense(500, activation='relu')
        output = Dense(1, activation="relu")(x)  # or can do   x = Dense(500, activation='relu')

        # return output
        return Model(inputs, output)

    





