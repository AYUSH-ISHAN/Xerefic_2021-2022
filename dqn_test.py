###  first implement in tensorflow and then go for pytorch

from keras.layers import Activation, Flatten
from tensorflow.keras.optimizers import Adam
import random
from keras import Sequential
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from collections import deque
from keras.layers import Dense, Conv2D
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt
from environment import Vasuki
import cv2

config = {'n': 8, 'rewards': {'Food': 4, 'Movement': -1, 'Illegal': -2}, 'game_length': 100} # Should not change for evaluation
env = Vasuki(**config)
#env = gym.make('LunarLander-v2')
DISCOUNT = 0.99
EPISODES = 20
EPSILON = 1
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.99975
GAMMA = 0.1
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
VERBOSE = 1
num_action_spaces = 4  
MINI_BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
AGGREGATE_STATS_EVERY = 50  # saving stats after 50 epochs

EP_REWARD = []  # In this we will store the rewards from each episodes.

# can save model only with val_loss (validation loss)

# callbacks = [
#              ModelCheckpoint("model.h5", save_best_only=True, verbose=1),
#              ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, min_lr=1e-6, verbose=1),
#              EarlyStopping(monitor='val_loss', patience=5, verbose=1)
# ]

class DQNAgent():
    def __init__(self):
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.model = self.DQN()
        self.target_model = self.DQN()
        self.target_update_counter = 0

    def update_memory(self, transition):
        self.replay_memory.append(transition)

    def DQN(self):
        '''
        Here, we can do Input shapes = env.n_observation  or action_space.n
        This will help us in getting the direct result of observation space and 
        action space dimensions.
        '''

        '''
        May be we can include some dropouts, Pooling layers and all for better results 
        '''
        model = Sequential()
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        # model.Flatten()
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(3))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def Q_values(self, state):

        return self.model(np.array(state).reshape(-1, state.shape)/255)[0]

    def train(self, step, terminal_state):
        
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            #print("*****************Stuck here*******************")
            return

        #print("****************************hello*******************************")
        
        minibatch = random.sample(self.replay_memory, MINI_BATCH_SIZE)
        #print("transitions:", [transitions[0] for transitions in minibatch])
        current_states = np.array([transitions[0] for transitions in minibatch])#/255
        current_q_list = self.model.predict(current_states)

        new_states = np.array([transitions[3] for transitions in minibatch])#/255
        next_q_list = self.target_model.predict(new_states)

        X = []
        y = []

        for index, (current_state, action, reward, next_current_state, done) in enumerate(minibatch):

            if not done:
                future_q_max = np.max(next_q_list[index])  # max Q of action that is to be taken
                next_q = reward + DISCOUNT*future_q_max
            else:
                next_q = reward

            ## update the qs values to the current Q table 

            current_qs = current_q_list[index]
            current_qs[action] = next_q
                
            X.append(current_state)
            y.append(current_qs)

            '''Also introduce its own custom callback -- as did in Semantic Seg.'''
            self.model.fit(np.array(X)/255, np.array(y), batch_size=MINI_BATCH_SIZE, shuffle=False) #, callbacks=callbacks)

            '''we will set the weights of target model as that of original model
            used and then comparing it'''

            if terminal_state:
                self.target_update_counter += 1

            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0
        
    def fetch_Qs(self, state):
        # print("State in fetch Qs is : ", state)
        # print("QQQQQQQQQQQQQQQQQQQQQQQQQ", self.model.predict(state))
        '''just print and see what is the return of predict'''
        print(self.model.predict((state)/255)[0])#.reshape(-1, *state.shape)/255))
        return self.model.predict((state)/255)[0]#.reshape(-1, *state.shape)/255)[0]
        

##  after this we will run episodes to train our model.

agentA = DQNAgent()
agentB = DQNAgent()
episode_list_A = []  # In this we will store the rewards from each episodes.
episode_list_B = []
avg_reward_list = []

for episode in tqdm(range(1, EPISODES+1), ascii='True', unit='episodes'):
        
        '''
        1. Look at Lunar Lander github code to see how its coded.
        2. Look at Coursera RL assignemts and see how they have the Lunar Lander Model
        '''

        episode_reward_A = 0
        episode_reward_B = 0
        step = 1

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

        print("Current state A", Current_StateA)
        print("Current State B", Current_StateB)
        

        done = False  ## this waits till terminal step
        while not done:
            print(done)
            if np.random.random() > EPSILON:
                actionA = np.argmax(agentA.fetch_Qs(Current_StateA))
                actionB = np.argmax(agentB.fetch_Qs(Current_StateB))
            else:
                actionA = np.random.randint(0, 3)
                actionB = np.random.randint(0, 3)
            
            action = {'actionA' : actionA, 'actionB' : actionB}
           # new_state, reward, done, _ = env.step(action)
            rewardA, rewardB, done, new_info = env.step(action)

            episode_reward_A += rewardA
            episode_reward_B += rewardB

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

            # if episode % 100:
            #     img = env.render(actionA, actionB)  
            #     cv2.imshow('render', img)
            #     cv2.waitKey(1)
            #     cv2.destroyAllWindows()

            ''' You can use SHOW_PREVIEW or similarly
                AGGREGATE STATS EVERY more detailing
            '''
            
            agentA.update_memory([Current_StateA, actionA, rewardA, New_State_A, done])
            agentA.train(step, done)
            agentB.update_memory([Current_StateB, actionB, rewardB, New_State_B, done])
            agentB.train(step, done)
            Current_StateA = New_State_A
            Current_StateB = New_State_B
            step+=1
        
        EP_REWARD.append([episode_reward_A, episode_reward_B])

        '''
            In this subsection update the Logs + Savce the model.
            Below this we will do the gradient descent step.
        '''
        # agentA.model.compile(optimizer='adam', loss='mse', metrics=['accuracy']]
        # Epsilon decay step:

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)
        episode_list_A.append(episode_reward_A)
        episode_list_B.append(episode_reward_B)
        averages_reward_A = np.mean(episode_list_A[-5:])  # average the reward per 100 epochs or last 100 epochs
        averages_reward_B = np.mean(episode_list_B[-5:])
        print(f' of {episode} EPISODE -> {episode_reward_A}\nAVERAGE REWARD of A-> {averages_reward_A}')
        print(f' of {episode} EPISODE -> {episode_reward_B}\nAVERAGE REWARD of B-> {averages_reward_B}')
        avg_reward_list.append([averages_reward_A, episode_reward_B])

        agentA.model.save_weights('dqn_agent_A.h5')
        agentB.model.save_weights('dqn_agent_B.h5')

plt.plot(episode_list_A)
plt.xlabel("Episodes")
plt.ylabel("Episodic Reward of A")
plt.show()

plt.plot(episode_list_B)
plt.xlabel("Episodes")
plt.ylabel("Episodic Reward of B")
plt.show()

plt.plot(avg_reward_list)
plt.xlabel("Episodes")
plt.ylabel("Avg Episodic Reward (per 5 episodes)")
plt.show()
'''
I THINK I AM DONE !!
'''
###########################################################################################################################






