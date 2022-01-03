# In this file we will declare the actor-critic model to feed in the DDPG.
import tensorflow as tf
from keras.layers.core import Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Concatenate
from keras.layers import Dense,Input, Activation, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.models import Model
import gym

env = gym.make('LunarLander-v2')
num_states = env.observation_space.shape[0]
num_actions = 1 #env.action_space.shape

upper_bound = 1    # env.action_space.high[0]
lower_bound = -1    # env.action_space.low[0]

class Critic():

    def __init__(self):
        self.state_dim = num_states #num_states  # in lunar lander we have 8 states
        self.action_dim = num_actions   # in lunar lander we have 1 action.

    def C_model(self):

        # print(f'shape of input in the Critic Network -> {self.state_dim} and for action -> {self.action_dim}')
        
        state_in = Input(shape=self.state_dim)
        state_out = Dense(25, activation="relu")(state_in)
        state_out = Dense(50, activation="relu")(state_out)
        # state_out = tf.transpose(state_out)

        action_in = Input(shape=self.action_dim)
        # print("action_in -> ",action_in)
        action = Dense(50, activation="relu")(action_in)
        # action = tf.transpose(action)
        # take the transpose of the action and then feed in Concatination for dimension matching.
        # x = concatenate([state_out, action])
        
        action = tf.reshape(action, (50, 1))
        state_out = tf.reshape(state_out, (50, 8))

        x = Concatenate()([state_out, action])
        x = Dense(100, activation="relu")(x)
        x = Dense(100, activation="relu")(x)
        output = Dense(1)(x)

        # return output

        return  Model([state_in, action_in], output)

    












