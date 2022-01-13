<h1>Xerefic – Shaastra 2021-22<h2>

<h2>Problem Statement :</h2>

<h3>Game Description:</h3>
  
The environment consists of two snakes (agents) and 4 food locations at any instant. The snakes (agents) can move in three directions; namely left, right or straight ahead. The objective of the game is to possess a greater score than the opponent either by consuming the food or by colliding with the opponent.
A breief description on the environment is given below:
<h4>State Space:</h4>
    • The state space is characterised by a(8 x 8)Grid (Continious Space).
    • At any instant of time,(4)random coordinates out of(8)fixed coordinates possess food.
<h4>Action Space:</h4>
    • The agent may choose one of the three possible moves; left, right, forward at any instant.
    • Depending on the position of the agent, the move may or may not be executed.
        ◦ For instance, if the agent lies on the first row and is facing North, and decides to move left, the move will be determined illegal and the agent will not be displaced. Although the move does not take place, the agent will be turned to face West.
        ◦ That is, the agent will first turn to left and then try to move. Since the move is illegal, the agent stays put.
<h4>Rules:</h4>
    1. The agent must eat the food to grow.
    2. If the agent collides with the opponent:
        ◦ Let(s1)and(s2)be the scores of the two agents.
        ◦ If(s1 > s2,)r1 = 5 s2/(s1-s2) and r2 = -3 s2/(s1-s2)
        ◦ If(s1 < s2,)r1 = -3 s1/(s2-s1) and r2 = 5 s1/(s2-s1)
    3. After collison, the agent with the lesser score is randomly respawned.
<h4>Reward System:</h4>
  <ul>
    <li>-1 for legal moves</li>
    <li>-2 for illegal moves</li>
    <li>+4 for consuming food</li>
    <li>Collision
      <ul>
      <li>If (s1 > s2,) r1 = 5 s2/(s1-s2) and r2 = -3 s2/(s1-s2)</li>
      <li>If (s1 < s2,) r1 = -3 s1/(s2-s2) and r2 = 5 s1/(s2-s1)</li>
      </ul></li>
  </ul>
<h4>Evaluation Metric:</h4>
    • Every game lasts for a maximum of (game_length = 100) iterations.
    • The agent with the greater score wins the game.
    • Play runs = 1000 games against the opponent.
    • The agent with higher number of victories wins the bracket.

<h3>Our Approach :</h3>

The given system resembles with the descrete actions spaced environement of the atari games or the open AI gym envs. So, a natural choice would be to use the architecture based on nueral networks. In this environement there are two agents playing against other that means it is a competative environement with descrete action space. On basis of this analysis we chose DQN od deep Q networks as our architecture with epsiolon greedy as its policy as shown below.
In this DQN algorithm, we are udating the target algorithm after every 5 training steps and the training steps starts when our algorithm has a minimum of 1000 units of memory size and maximum it can go upto 50, 000 units. By this we ensure we have enough state sapce or the data for both the agents to give tough fight to each other. 

          Here, is the list of hyperparameters which we have used : 

          DISCOUNT = 0.99    # this is the discount factor
          EPISODES = 20
          EPSILON = 1
          MIN_EPSILON = 0.001
          EPSILON_DECAY = 0.99975
          GAMMA = 0.1
          REPLAY_MEMORY_SIZE = 50_000
          MIN_REPLAY_MEMORY_SIZE = 1_000
          VERBOSE = 1
          num_action_spaces = 3
          MINI_BATCH_SIZE = 64
          UPDATE_TARGET_EVERY = 5   # this is the parameter which takes care of updating of target network.


<h3>Model Architecture :</h3>

          Model: "sequential_1"
          _________________________________________________________________
          Layer (type)                 Output Shape              Param #   
          =================================================================
          dense_8 (Dense)              (None, 256)               1536      
          _________________________________________________________________
          activation_6 (Activation)    (None, 256)               0         
          _________________________________________________________________
          dense_9 (Dense)              (None, 256)               65792     
          _________________________________________________________________
          activation_7 (Activation)    (None, 256)               0         
          _________________________________________________________________
          dense_10 (Dense)             (None, 64)                16448     
          _________________________________________________________________
          activation_8 (Activation)    (None, 64)                0         
          _________________________________________________________________
          dense_11 (Dense)             (None, 3)                 195       
          =================================================================
          Total params: 83,971
          Trainable params: 83,971
          Non-trainable params: 0


We have also done further modification in this DQN is that we have been decreasing the epsilon using the below command.

        if EPSILON > MIN_EPSILON:
          EPSILON *= EPSILON_DECAY
          EPSILON = max(MIN_EPSILON, EPSILON)

Also the optimizer which we used up was Adam with learning rate of 0.001.




***************************************************
