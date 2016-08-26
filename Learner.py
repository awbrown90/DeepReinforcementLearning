from __future__ import print_function
__author__ = 'Aaron Brown'
import World
import threading
import time
import random
import numpy as np
import tensorflow as tf
from collections import deque

actions = World.actions
# intialize state to all ones, will get updated later
state = []
# tt = rr + discount * max(a') Q(ss',aa') or rr if terminal state
# intialize the tt value and target
tt = 0
target = 0

# The variables below hold all the trainable weights for our CNN. For each, the
# parameter defines how the variables will be initialized.

# The random seed that defines initialization
GAMMA = 0.8 # decay rate of past observations
OBSERVE = 2000 # timesteps to observe before training
EXPLORE = 3000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
BATCH = 32 # size of minibatch
REPLAY_MEMORY = 500000 # the size of the replay memory
# store the previous observations in replay memory
D = deque()

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
  tf.truncated_normal([5, 5, 1, 16],  # 5x5 filter, depth 8.
                      stddev=0.1))
conv1_biases = tf.Variable(tf.zeros([16]))
conv2_weights = tf.Variable(
  tf.truncated_normal([3, 3, 16, 32], # 3x3 filter, depth 16
                      stddev=0.1))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[32]))

conv3_weights = tf.Variable(
  tf.truncated_normal([2, 2, 32, 64], # 3x3 filter, depth 16
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
#	     |------target-------|  |prediction|

# Do a feedforward pass for the current state s to get predicted Q-values for all actions.
action_array_1 = network(state_input_1)
# Do a feedforward pass for the next state s' and calculate maximum overall network outputs max a' Q(s', a').
# Set Q-value target for action to r + discount * max a' Q(s', a') (use the max calculated in step 2). 
# For all other actions, set the Q-value target to the same as originally returned from step 1, making the error 0 for those outputs.
tt = reward_input + terminal_input * (GAMMA * max_val_input) 
tt = tf.reshape(tt,(BATCH,1))
target_prep = tf.tile(tt,[1,4])
target = tf.select(action_input, target_prep, action_array_1)

# loss is .5(tt - Q(ss,aa))^2
Qerror = tf.sub(target, action_array_1)
loss = .5*tf.reduce_sum(tf.mul(Qerror, Qerror))

regularizers = (tf.nn.l2_loss(fc1_weights)+tf.nn.l2_loss(fc1_biases)+tf.nn.l2_loss(fc2_weights)+tf.nn.l2_loss(fc2_biases))
				
# Add the regularization term to the loss.
loss += 5e-5 * regularizers     

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
#batch = tf.Variable(0)
#learning_rate = tf.train.exponential_decay(
#  1e-3,                # Base learning rate.
#  batch * BATCH,  # Current index into the dataset.
#  5000,            # Decay step.
#  0.95,                # Decay rate.
#  staircase=True)

# Update the weights using backpropagation.
optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

# saving and loading networks
saver = tf.train.Saver()
tf.initialize_all_variables().run()

#checkpoint = tf.train.get_checkpoint_state("saved_networks")
#if checkpoint and checkpoint.model_checkpoint_path:
#	saver.restore(sess, checkpoint.model_checkpoint_path)
#	print("Successfully loaded:", checkpoint.model_checkpoint_path)
#else:
#	print("Could not find old network weights")

def see_action(action):
	
	if action == actions[0]:
		reward, s2, t = World.see_move(0, -1)
	elif action == actions[1]:
		reward, s2, t= World.see_move(1, 0)
	elif action == actions[2]:
		reward, s2, t = World.see_move(0, 1)
	elif action == actions[3]:
		reward, s2, t = World.see_move(-1, 0)
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
	for i in range(World.x):
		for j in range(World.y):
			state = World.get_state((i,j))
			state = np.reshape(state,(1, 9, 9, 1)).astype(np.float32)
			feed_dict = {state_input_1: state}
			values = sess.run(action_array_1, feed_dict=feed_dict)
			for action in actions:
				World.set_cell_score(i,j,action,values)

def run():
    time.sleep(1.0)
    trials = 0 
    epsilon = INITIAL_EPSILON  # the exploration value, starts high and becomes smaller
    t = 0
    moves = 0
    while trials < 500:

    	# run transitions multiple times to get collection of <s,a,r,s'> data thats equal to BATCH_SIZE

    	# update current state
    	state_1 = World.get_state(World.player)

    	state_peek = np.reshape(state_1,(1, 9, 9, 1)).astype(np.float32)
    	state_1    = np.reshape(state_peek,(9, 9, 1)).astype(np.float32)
    	feed_dict = {state_input_1: state_peek}
    	net_out_1 = sess.run(action_array_1, feed_dict=feed_dict)
    	
    	#World.get_pos_from_state(state_peek)
    	#print(net_out_1[0])
    	
    	# help exploration early in the game
    	choice = np.random.choice(2,1,p=[1-epsilon,epsilon])
    	if choice==1:
    		random_index = np.random.choice(4,1)
    		max_index = random_index[0]
    	else:
    		max_index = np.argmax(net_out_1[0])

    	max_act = actions[max_index]

    	max_act_prep = np.reshape([False, False, False, False],(4)).astype(np.bool)
    	max_act_prep[max_index] = True

    	reward, s2, terminal = see_action(max_act)
    		
    	do_action(max_act)

    	state_peek_2 = np.reshape(s2,(1, 9, 9, 1)).astype(np.float32)
    	feed_dict = {state_input_1: state_peek_2}
    	net_out_2 = sess.run(action_array_1, feed_dict=feed_dict)

    	max_val_data = np.amax(net_out_2)

    	# store the transition in D
        D.append((state_1, max_act_prep, reward, max_val_data, terminal))
        if len(D) > REPLAY_MEMORY:
        	D.popleft()

    	# Check if the game has restarted
    	if World.has_restarted():
    		print('completed trial {}'.format(trials))
    		print('it took {} moves'.format(moves))
    		print('epsilon was {}'.format(epsilon))
    		print('time was {}'.format(t))

    		
    		trials+=1
    		moves = 0
    		World.restart_game()

    	# expiration for the reset, the agent is stuck
    	if(moves > 500 and t > OBSERVE):
    		moves = 0
    		World.restart_game()

    	# only train if done observing
    	# update weights and minimize loss function for BATCH_SIZE amount of data points

    	if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        elif epsilon < FINAL_EPSILON:
        	epsilon = 0.0

    	if t > OBSERVE:
    		
    		# sample a minibatch to train on
    		minibatch = random.sample(D, BATCH)

    		s1_update = [d[0] for d in minibatch]
    		a_update  = [d[1] for d in minibatch]
    		r_update  = [d[2] for d in minibatch]
    		mv_update = [d[3] for d in minibatch]
    		term      = [d[4] for d in minibatch]

    		feed_dict = {state_input_1: s1_update, action_input: a_update, reward_input: r_update, max_val_input: mv_update, terminal_input: term}

    		_, my_loss, start, _end_, my_tt = sess.run([optimizer, loss, action_array_1, target, tt], feed_dict=feed_dict)

    		#show visually the updated cells
    		#state_prep = np.reshape(s1_update,(BATCH,9,9))
    		#for i in range(BATCH):
    		#	xu, yu = World.get_pos_from_state(state_prep[i])
    		#	World.update_show(xu, yu)
    		#print(World.get_pos_from_state(state_peek))
    		#run visual analysis of network state space
    		network_triangles()
    		#print('a update {}'.format(a_update))
    		#print('start: {}'.format(start))
    		#print('_end_: {}'.format(_end_))
    		#print('___s2: {}'.format(s_prime))
    		#print('___tt: {}'.format(my_tt))
    		#print('loss_: {}'.format(my_loss))

    		#print("new result")
    		#feed_dict = {state_input_1: s1_update, action_input: a_update, reward_input: r_update, state_input_2: s2_update, terminal_input: term } 
    		#post_u, new_loss = s.run([action_array_1, loss], feed_dict=feed_dict)
    		#print('_new_: {}'.format(post_u))
    		#print('nloss: {}'.format(new_loss))

    	# MODIFY THIS SLEEP IF THE GAME IS GOING TOO FAST.
    	#time.sleep(0.01)
    	moves += 1
    	t += 1
    	# save progress every 5000 iterations
        if t % 5000 == 0:
            saver.save(sess, 'saved_networks/' + 'i_maze' + '-dqn', global_step = t)

    #log = open(".\optimal_policy.txt", "w")
    #print(get_policy(), file = log)


t = threading.Thread(target=run)
t.daemon = True
t.start()
World.start_game()
