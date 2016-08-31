Solving Random Mazes using Asynchronous Deep Reinforcement Learning
====================================================================

Requirement for running this program is installation of Tensorflow, which is freely available. If you are a Windows user it is strongly encouraged to install the Windows 10 Bash with Ubuntu Update to easily work with Tensorflow. A helpful link can be found here, http://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/

The basic idea of this project is to use deep reinforcement learning to solve any number of 5x5 randomly generated mazes, where the start location is always at the bottom left cell and the goal is always at the center of the maze.

In order to solve any number of randomly generated mazes the agent needs to know as much about the maze as possible such as where all its walls are located, beyond just knowing where it its self is located in the maze.

In order to deal with high dimensional state spaces the agent uses and trains with a convolutional neural network, similar to how Deep Mind trained with a program to play Atari games with just raw pixel data.

Each maze state consits of a 9x9 2D image showing walls/pegs with -1, open spaces with 0, and the agent with 1. An example of a possible state is shown below.
  0     1     2     3     4
  -------------------------
0|0  0  0  0  0  0  0  0  0
 |0 -1  0 -1 -1 -1 -1 -1  0
1|0 -1  0  0  0  0  0 -1  0
 |0 -1 -1 -1  0 -1  0 -1  0
2|0  0  0 -1  0 -1  0  0  0
 |0 -1  0 -1  0 -1 -1 -1  0
3|0 -1  0  0  0  0  1 -1  0
 |0 -1 -1 -1 -1 -1  0 -1  0
4|0  0  0  0  0  0  0  0  0

This Maze is called the "I maze" because it looks like a hollowed out I. It has 4 entrances into its center goal area, the agent can be seen entering from the bottom right entrance. The agent "1" is located at the (3,3) cell which has cell (0,0) at top left. Notice that there are two optimal paths from the start to the goal, each 6 moves long. 

deep reinforcement learning example, based on the Q-function.
- Rules: The agent (black box) has to reach the center goal and then it starts over with a new random maze.
- Rewards: Each step gives a negative reward of -0.1. Running into a wall gets -1. Reaching the center goal gets +10.
- Actions: There are only 4 actions. Up, Down, Right, Left.

The Learning Policy follows this algorithm.
Initialize action -value function Q with random weights 
For N trials DO:
	Set agent to start of maze
	Generate new random maze
	While agent has not reached center of maze
		For each possible cell position Pick random action for cell position 
			Feed forward cell position state in Q and see reward, and next state from
			following random action
			Set y_j= r_j 					if next state is terminal
 				 r_j + gamma* maxa'Q(s_(j+1) , a ) 	for non-terminal state
		End for
	Perform gradient decent and minimize Loss as defined in Learner.py
	Feed forward agents state in Q and choice action with highest value from Q
	End while
End for


The little triangles represent the values of the Q network for each state and each action. Green is positive and red is negative.

# Run
Run the file Learner.py

# Demo
Not yet Avaliable
