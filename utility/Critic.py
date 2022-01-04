# In this file we will declare the actor-critic model to feed in the DDPG.
import tensorflow as tf
from keras.layers.core import Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Concatenate
from keras.layers import Dense,Input, Activation, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.models import Model
from environment import Vasuki
import cv2

config = {'n': 8, 'rewards': {'Food': 4, 'Movement': -1, 'Illegal': -2}, 'game_length': 100} # Should not change for evaluation
env = Vasuki(**config)
#num_states = env.observation_space.shape[0]
#print("Size of State Space ->  {}".format(num_states))
#num_actions = env.action_space.shape
#print("Size of Action Space ->  {}".format(num_actions))
upper_bound = 0    # env.action_space.high[0]
lower_bound = 3    # env.action_space.low[0]

class Critic():

    def __init__(self):
        self.state_dim = 5#num_states #num_states  # in lunar lander we have 8 states
        self.action_dim = 1#num_actions   # in lunar lander we have 1 action.

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
        state_out = tf.reshape(state_out, (50, 5))  #(50, state_dim)

        x = Concatenate()([state_out, action])
        x = Dense(100, activation="relu")(x)
        x = Dense(100, activation="relu")(x)
        output = Dense(1)(x)

        # return output

        return  Model([state_in, action_in], output)

    












