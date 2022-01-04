import gym
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from numpy import random
from DDPG.utility.Actor import Actor
from DDPG.utility.Critic import Critic   # importing actor and critic model from the file.
from keras.layers import Dense,Input
from tensorflow.keras.layers import Concatenate
from utility.utils import *
from collections import deque
from tqdm import tqdm
import random
from tensorflow.keras.optimizers import Adam
from environment import Vasuki
import cv2

config = {'n': 8, 'rewards': {'Food': 4, 'Movement': -1, 'Illegal': -2}, 'game_length': 100} # Should not change for evaluation
env = Vasuki(**config)
#num_states = env.observation_space.shape[0]
#print("Size of State Space ->  {}".format(num_states))
#num_actions = env.action_space.shape
#print("Size of Action Space ->  {}".format(num_actions))

upper_bound = 0   # env.action_space.high[0]
lower_bound = 3    # env.action_space.low[0]
Actor = Actor()
Critic = Critic()
Orn_Uhl = Ornstein_Uhlenbeck()

REPLAY_MEMORY_SIZE = 50_000
EPISODES = 20
NUM_TRANS = 10
MIN_Replay = 5_000
MINI_BATCH_SIZE = 1000
GAMMA = 0.98
tau = 0.005

class DDPG():

    def __init__(self):

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)   # to initialise a replay buffer.
        self.Critic = Critic.C_model()
        self.Actor = Actor.A_model()
        self.target_Critic = Critic.C_model()
        self.target_Actor = Actor.A_model()
        self.trans_per_iter = NUM_TRANS
        self.min_replay = MIN_Replay
        self.state_dim = 5#num_states  # in lunar lander we have 8 states
        self.action_dim = 1#num_actions   # in lunar lander we have 1 action.

    def update_memory(self, transitions):

        self.replay_memory.append(transitions)

    # def A_model(self):
        
    #     """ Actor Network for Policy function Approximation, using a tanh
    #     activation for continuous control. We add parameter noise to encourage
    #     exploration, and balance it with Layer Normalization.
    #     """

    #     inputs = Input(shape=self.state_dim)
    #     x = Dense(500, activation="relu")(inputs)  # or can do   x = Dense(500, activation='relu')
    #     x = Dense(100, activation="relu")(x)  # or can do   x = Dense(500, activation='relu')
    #     output = Dense(1, activation="relu")(x)  # or can do   x = Dense(500, activation='relu')

    #     return output
    #     # return Model(inputs, output)

    # def C_model(self):
        
    #     state_in = Input(shape=self.state_dim)
    #     state_out = Dense(25, activation="relu")(state_in)
    #     state_out = Dense(50, activation="relu")(state_out)
    #     # state_out = tf.transpose(state_out)

    #     action_in = Input(shape=self.action_dim)
    #     action = Dense(50, activation="relu")(action_in)
    #     # action = tf.transpose(action)
    #     # take the transpose of the action and then feed in Concatination for dimension matching.
    #     # x = concatenate([state_out, action])
        
    #     action = tf.reshape(action, (50, 1))
    #     state_out = tf.reshape(state_out, (50, 8))

    #     x = Concatenate()([state_out, action])
    #     x = Dense(100, activation="relu")(x)
    #     x = Dense(100, activation="relu")(x)
    #     output = Dense(1)(x)

    #     return output
    #     # return Model([state_in, action_in], output)

    def Policy(self, state, noise):  
        '''
        # argument is state at a given parameter theta.
        # also add the Ornstein_Uhlenbeck random data
        '''
        # print("Input shape -> ", tf.shape(state))
        # print("state -> ", state)
        state=np.expand_dims(state,axis=0)
        #print("State in Actor: ", state)
        sampled_actions = self.Actor.predict(state)
        # sampled_actions = self.Actor.predict(state)

        # print("input shape -> ", tf.shape(sampled_actions))
        # print("sampled actions -> ", sampled_actions)
        sampled_actions = sampled_actions + noise     
                        # converted tensorflow tensor to numpy array
        action = np.clip(sampled_actions[0][0], lower_bound, upper_bound)   # clipping between the lower and upper bound

        return action

    def train_and_update(self):

        '''
        From here I will Monitor the whole update thing and helper funtions calling.
        '''
        if len(self.replay_memory) < self.min_replay:
            # print("I am here !!")
            return

        miniBatch = random.sample(self.replay_memory, MINI_BATCH_SIZE)
        # miniBatch is of format DDPG.update_mamory() in the episode running step.

        state = [trans[0] for trans in miniBatch]
        action = [trans[1] for trans in miniBatch]
        reward = [trans[2] for trans in miniBatch]
        future_state = [trans[3] for trans in miniBatch]
        
        ''' If any error occurs put the trainable = True in these cases.'''


        with tf.GradientTape() as tape:
            
            print("Future State: ", future_state)
            target_action = self.target_Actor(future_state)
            Y = reward + GAMMA*self.target_Critic(future_state, target_action)
            critic_val = self.Critic([state, action])
            Loss = tf.math.reduce_mean(tf.math.square(Y, critic_val))
            
            '''
            Update the critic after minimising this loss.
            '''

        C_gradient = tape.gradient(Loss, self.Critic.trainable_variables)

        Adam.apply_gradients(zip(C_gradient, self.Critic.trainable_variables))

        ''' Training and Updating the Actor Model '''

        with tf.GradientTape() as tape:

            A_actions = self.Actor(state)
            critic_val = self.Critic([state, A_actions])

            A_Loss = -tf.math.reduce_mean(critic_val)

        A_grad = tape.gradient(A_Loss, self.Actor.trainable_variables)
        Adam.apply_gradients(zip(A_grad, self.Actor.trainable_variables))

        ''' Updating the weights '''

        for (t_w, w) in zip(self.target_Actor.weights, self.Actor.weights):

            t_w.assign(w * tau + (1 - tau) * t_w)
        
        for (t_w, w) in zip(self.target_Critic.weights, self.Critic.weights):

            t_w.assign(w * tau + (1 - tau) * t_w)


agentA = DDPG()
agentB = DDPG()
episode_list_A = []  # In this we will store the rewards from each episodes.
episode_list_B = []
avg_reward_list = []


for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episodes'):

    steps = 1  # initilising the steps per episodes or not.
    act_noise = Orn_Uhl.OU(steps)  # aame noises passed to both the agents.


    episode_reward_A = 0
    episode_reward_B = 0
    env.reset()
    current_state_A = env.agentA
    Current_StateA = []
    Current_StateA.append(current_state_A['head'])
    Current_StateA.append(current_state_A['score'])
    Current_StateA.append(current_state_A['state'][0])
    Current_StateA.append(current_state_A['state'][1])
    Current_StateA.append(current_state_A['velocity'])
    #print(StateB)
    current_state_B = env.agentB
    Current_StateB=[]
    Current_StateB.append(current_state_B['head'])
    Current_StateB.append(current_state_B['score'])
    Current_StateB.append(current_state_B['state'][0])
    Current_StateB.append(current_state_B['state'][1])
    Current_StateB.append(current_state_B['velocity'])
        

    done = False  ## this waits till terminal step
    while not done:

        # print(current_state)
        actionA = int(agentA.Policy(Current_StateA, act_noise))
        actionB = int(agentB.Policy(Current_StateB, act_noise))
        # print(action)
        action = {'actionA' : actionA, 'actionB' : actionB}
        
        rewardA, rewardB, done, new_info = env.step(action)

        new_state_A = new_info['agentA']
        New_State_A = []
        New_State_A.append(new_state_A['head'])
        New_State_A.append(new_state_A['score'])
        New_State_A.append(new_state_A['state'][0])
        New_State_A.append(new_state_A['state'][1])
        New_State_A.append(new_state_A['velocity'])
        new_state_B = new_info['agentB']
        New_State_B = []
        New_State_B.append(new_state_B['head'])
        New_State_B.append(new_state_B['score'])
        New_State_B.append(new_state_B['state'][0])
        New_State_B.append(new_state_B['state'][1])
        New_State_B.append(new_state_B['velocity'])
        
        agentA.update_memory([Current_StateA, actionA, rewardA, New_State_A])
        agentB.update_memory([Current_StateB, actionB, rewardB, New_State_B])
        ''' Training and Updating the Critic Model '''

        episode_reward_A += rewardA
        episode_reward_B += rewardB

        agentA.train_and_update()
        agentB.train_and_update()
        
        # if episode % 100 == 0:
        #     img = env.render(actionA, actionB)  
        #     cv2.imshow('render', img)
        #     cv2.waitKey(1)
        #     cv2.destroyAllWindows()

       
    episode_list_A.append(episode_reward_A)
    episode_list_B.append(episode_reward_B)
    averages_reward_A = np.mean(episode_list_A[-5:])  # average the reward per 100 epochs or last 100 epochs
    averages_reward_B = np.mean(episode_list_B[-5:])
    print(f'loss A in {episode} EPISODE -> {episode_reward_A}\nAVERAGE REWARD of A-> {averages_reward_A}')
    print(f'loss B in {episode} EPISODE -> {episode_reward_B}\nAVERAGE REWARD of B-> {averages_reward_B}')
    avg_reward_list.append([averages_reward_A, episode_reward_B])

    agentA.Actor.save_weights('ddpg_agent_A_ACTOR.h5')
    agentA.Critic.save_weights('ddpg_agent_A_CRITIC.h5')
    agentB.Actor.save_weights('ddpg_agent_B_ACTOR.h5')
    agentB.Critic.save_weights('ddpg_agent_B_CRITIC.h5')

plt.plot(episode_list_A)
plt.xlabel("Episodes")
plt.ylabel("Episodic Reward")
plt.show()

plt.plot(episode_list_B)
plt.xlabel("Episodes")
plt.ylabel("Episodic Reward")
plt.show()

plt.plot(avg_reward_list)
plt.xlabel("Episodes")
plt.ylabel("Avg Episodic Reward (per 5 episodes)")
plt.show()
'''
I THINK I AM DONE !!
'''
###########################################################################################################################






