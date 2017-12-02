from __future__ import print_function
__author__ = 'Aaron Brown'
import World
import threading
import time
import random
import numpy as np
import tensorflow as tf
import floodfill
from collections import deque

actions = World.actions

#Turn the GUI on, or off if training
gui_display = True

if(not gui_display):
	World.gui_off()

# The variables below hold all the trainable weights for our CNN. For each, the

GAMMA = 0.8 # decay rate of past observations
BATCH = 25 # size of minibatch

#<s,a,r,s'>
state_input_1 = tf.placeholder(
	tf.float32,
	[None,9,9,1])

action_input = tf.placeholder(
	tf.bool,
	shape=(BATCH,4))

reward_input = tf.placeholder(
	tf.float32,
	shape=(BATCH))

max_val_input = tf.placeholder(
	tf.float32,
	shape=(BATCH))

terminal_input = tf.placeholder(
	tf.float32,
	shape=(BATCH))

conv1_weights = tf.Variable(
	tf.truncated_normal([5, 5, 1, 16],  # 5x5 filter, depth 16.
											stddev=0.1))
conv1_biases = tf.Variable(tf.zeros([16]))
conv2_weights = tf.Variable(
	tf.truncated_normal([3, 3, 16, 32], # 3x3 filter, depth 32
											stddev=0.1))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[32]))

conv3_weights = tf.Variable(
	tf.truncated_normal([2, 2, 32, 64], # 3x3 filter, depth 64
											stddev=0.1))
conv3_biases = tf.Variable(tf.constant(0.1, shape=[64]))

fc1_weights = tf.Variable(  # fully connected, depth 128.
	tf.truncated_normal([256, 512],
											stddev=0.1))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
fc2_weights = tf.Variable(
	tf.truncated_normal([512, 4],
											stddev=0.1))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[4]))

def network(data):
	conv = tf.nn.conv2d(data,
											conv1_weights,
											strides=[1, 1, 1, 1],
											padding='VALID')

	# Bias and rectified linear non-linearity.
	relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

	conv = tf.nn.conv2d(relu,
											conv2_weights,
											strides=[1, 1, 1, 1],
											padding='VALID')
	relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))

	conv = tf.nn.conv2d(relu,
											conv3_weights,
											strides=[1, 1, 1, 1],
											padding='VALID')
	relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))

	# fully connected layers.
	relu_flat = tf.reshape(relu, [-1,256])

	# Fully connected layer. Note that the '+' operation automatically
	# broadcasts the biases.
	hidden = tf.nn.relu(tf.matmul(relu_flat, fc1_weights) + fc1_biases)

	return tf.matmul(hidden, fc2_weights) + fc2_biases

sess = tf.InteractiveSession()
sess.as_default()

# L = .5[r + discount * max a' Q(s', a') - Q(s, a)]^2
#      |------target-------|  |prediction|

# Do a feedforward pass for the current state s to get predicted Q-values for all actions.
action_array_1 = network(state_input_1)
# Do a feedforward pass for the next state s' and calculate maximum overall network outputs max a' Q(s', a').
# Set Q-value target for action to r + discount * max a' Q(s', a') (use the max calculated in step 2). 
# For all other actions, set the Q-value target to the same as originally returned from step 1, making the error 0 for those outputs.

# tt = rr + discount * max(a') Q(ss',aa') or rr if terminal state
tt = reward_input + terminal_input * (GAMMA * max_val_input) 
tt = tf.reshape(tt,(BATCH,1))
target_prep = tf.tile(tt,[1,4])
try:
	# Tensorflow < 1.4.0
	target = tf.select(action_input, target_prep, action_array_1)
except AttributeError:
	# Tensorflow >= 1.4.0
	target = tf.where(action_input, target_prep, action_array_1)

# loss is .5(tt - Q(ss,aa))^2
try:
	# Tensorflow < 1.0
	Qerror = tf.sub(target, action_array_1)
	loss = .5*tf.reduce_sum(tf.mul(Qerror, Qerror))
except AttributeError:
	# Tensorflow >= 1.0
	Qerror = tf.subtract(target, action_array_1)
	loss = .5*tf.reduce_sum(tf.multiply(Qerror, Qerror))

# Update the weights using backpropagation.
optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

# saving and loading networks
saver = tf.train.Saver()
tf.initialize_all_variables().run()

checkpoint = tf.train.get_checkpoint_state("saved_networks")
if checkpoint and checkpoint.model_checkpoint_path:
	saver.restore(sess, checkpoint.model_checkpoint_path)
	print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
	print("Could not find old network weights")

def see_action(action,i,j):
	
	if action == actions[0]:
		reward, s2, t = World.see_move(0, -1,i,j)
	elif action == actions[1]:
		reward, s2, t= World.see_move(1, 0,i,j)
	elif action == actions[2]:
		reward, s2, t = World.see_move(0, 1,i,j)
	elif action == actions[3]:
		reward, s2, t = World.see_move(-1, 0,i,j)
	else:
		return

	return reward, s2, t

def do_action(action):
	
	if action == actions[0]:
		World.do_move(0, -1)
	elif action == actions[1]:
		World.do_move(1, 0)
	elif action == actions[2]:
		World.do_move(0, 1)
	elif action == actions[3]:
		World.do_move(-1, 0)
	else:
		return

#update the visual network arrow display in GUI
def network_triangles():
	D = deque()
	for i in range(World.x):
		for j in range(World.y):
			state_peek_1 = World.get_state((i,j))
			state_peek_1 = np.reshape(state_peek_1,(1, 9, 9, 1)).astype(np.float32)
			feed_dict = {state_input_1: state_peek_1}
			values_1 = sess.run(action_array_1, feed_dict=feed_dict)
			state_peek_1 = np.reshape(state_peek_1,(9, 9, 1)).astype(np.float32)

			random_index = np.random.choice(4,1)
			try_index = random_index[0]
			try_act = actions[try_index]

			try_act_prep = np.reshape([False, False, False, False],(4)).astype(np.bool)
			try_act_prep[try_index] = True

			reward, s2, terminal = see_action(try_act,i,j)

			state_peek_2 = np.reshape(s2,(1, 9, 9, 1)).astype(np.float32)
			feed_dict = {state_input_1: state_peek_2}
			values_2 = sess.run(action_array_1, feed_dict=feed_dict)

			max_val_data = np.amax(values_2)

			D.append((state_peek_1, try_act_prep, reward, max_val_data, terminal))

			if(gui_display):
				for action in actions:
					World.set_cell_score(i,j,action,values_1)

	return D

def run():
	#initalize variables
	trials = 1 
	moves = 1
	t = 0
	hit_one = True

	#t0_floodfill = time.time()
	floodfill.FloodFillValues()
	#t1_floodfill = time.time()

	#print('running floodfill took {}'.format(t1_floodfill-t0_floodfill))

	opt_moves = floodfill.get_value(0,4)

	sub_trials = 1

	# variables used for running tests, note that some of these are not really compatiable with each other. Sort of hacked together for testing purposes
	train = True # used to train the network
	maze_space = -1 # number of saved mazes to iterate through, -1 means no iteration and always use new maze every time
	save_trial = 500 # save network off after every so many trials, -1 to disable save
	number_trial = -1 # number of trials to run, -1 for indefinite
	max_moves = -1 # max number of moves before restarting, -1 for no limit


	World.set_maze_size(maze_space)

	while trials < number_trial or (number_trial == -1):

		# run transitions multiple times to get collection of <s,a,r,s'> data thats equal to BATCH_SIZE

		# update current state
		state_1 = World.get_state(World.player)

		#print(state_1)

		state_peek = np.reshape(state_1,(1, 9, 9, 1)).astype(np.float32)
		feed_dict = {state_input_1: state_peek}
		#t0_network = time.time()
		net_out_1 = sess.run(action_array_1, feed_dict=feed_dict)
		#t1_network = time.time()

		#print('running the network took {}'.format(t1_network-t0_network))
		
		#World.get_pos_from_state(state_peek)
		#print(net_out_1[0])
		
		max_index = np.argmax(net_out_1[0])

		max_act = actions[max_index]
			
		do_action(max_act)

		# Check if the game has restarted
		if World.has_restarted() or (moves > max_moves and max_moves > 0):

			if(moves==opt_moves or (trials < maze_space or maze_space < 0)):
				trials+=1
				hit_one = True

			if(moves < max_moves or max_moves == -1):
				sub_trials+=1
			moves = 0

			#DEBUG
			print('at trial {}'.format(trials))
			#print('at subtrial {}'.format(sub_trials))
				
			World.restart_game(trials)

			#recalculate optimum number of moves
			#t0_floodfill = time.time()
			floodfill.FloodFillValues()
			#t1_floodfill = time.time()
			#print('running floodfill took {}'.format(t1_floodfill-t0_floodfill))
			opt_moves = floodfill.get_value(0,4)

			# save progress every so many iterations
			if save_trial > 0 and trials % save_trial == 0 and hit_one:
				saver.save(sess, 'saved_networks/' + 'async_maze' + '-dqn', global_step = t)
				
				print('completed trial {}'.format(trials))
				#subtrials is used as a reference in certain testing areas
				#print('took {} subtrials'.format(sub_trials))

				hit_one = False
				
				sub_trials = 1

		# update weights and minimize loss function for BATCH_SIZE amount of data points

		# sample a minibatch to train on
		if(train):
			D = network_triangles()
			minibatch = random.sample(D, BATCH)

			s1_update = [d[0] for d in minibatch]
			a_update  = [d[1] for d in minibatch]
			r_update  = [d[2] for d in minibatch]
			mv_update = [d[3] for d in minibatch]
			term      = [d[4] for d in minibatch]

			feed_dict = {state_input_1: s1_update, action_input: a_update, reward_input: r_update, max_val_input: mv_update, terminal_input: term}

			_, my_loss, start, _end_, my_tt = sess.run([optimizer, loss, action_array_1, target, tt], feed_dict=feed_dict)


		# MODIFY THIS SLEEP IF THE GAME IS GOING TOO FAST.
		#if gui_display:
			#time.sleep(1.0)
		moves += 1
		t += 1

	#log = open(".\optimal_policy.txt", "w")
	#print(get_policy(), file = log)
	#Test for maze completion without training
	if(max_moves > 0):
		print('completed trial {}'.format(trials))
		print('took {} subtrials'.format(sub_trials))

t = threading.Thread(target=run)
t.daemon = True
t.start()
World.start_game()
