REPO FOR PATH_FINDING ALGORITHM
================================================================
A* and Reinforcement Learning
----------------------------------------------------------------
# Simple_RL: 
* run simple_RL_train.py: Train the DQN in the simple_RL_env  
* Save the neural network so that it can be reloaded without training
* If the results are not ideal, try test it more times. The network might fall into overfitting or the accuracy just doesn't go up. 

# Something in the Path_Utils folder:
* Run Astar.py: Using AStar algorithm to find the best way to get to the target and avoid obstacles
* Run Testing.py: Simulate how the Jetbot reacts according to the AStar solutions. 

# Improvement in this RL environment:
* `States:` expand 3 into 5 -> take the relative position to the target into account 
* note: I tried to use AStar to teach it but the results seems not to be so good. Also, I tried to use different reward function, the results are not so good either.

# Others:
* Contributed by Weibo Huang 
* Reference: MorvanZhou
