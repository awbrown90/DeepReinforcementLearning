from __future__ import print_function
__author__ = 'Aaron Brown'
import World
import threading
import time
import random
import numpy as np
import tensorflow as tf

discount = 0.3
learning_rate = .05
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
SEED = 42
BATCH_SIZE = 16

#<s,a,r,s'>
state_input_1 = tf.placeholder(
	tf.float32,
	shape=(BATCH_SIZE,9,9,1))

action_input = tf.placeholder(
	tf.bool,
	shape=(BATCH_SIZE,4))

reward_input = tf.placeholder(
	tf.float32,
	shape=(BATCH_SIZE))

state_input_2 = tf.placeholder(
	tf.float32,
	shape=(BATCH_SIZE,9,9,1))

tf_discount = tf.constant(discount)

conv1_weights = tf.Variable(
  tf.truncated_normal([5, 5, 1, 8],  # 5x5 filter, depth 8.
                      stddev=0.1,
                      seed=SEED))
conv1_biases = tf.Variable(tf.zeros([8]))
conv2_weights = tf.Variable(
  tf.truncated_normal([3, 3, 8, 16], # 3x3 filter, depth 16
                      stddev=0.1,
                      seed=SEED))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[16]))
fc1_weights = tf.Variable(  # fully connected, depth 128.
  tf.truncated_normal([3 * 3 * 16, 128],
                      stddev=0.1,
                      seed=SEED))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[128]))
fc2_weights = tf.Variable(
  tf.truncated_normal([128, 4],
                      stddev=0.1,
                      seed=SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[4]))

def network(data, train=False):
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

  	# fully connected layers.
  	relu_shape = relu.get_shape().as_list()
  	reshape = tf.reshape(
      relu,
      [relu_shape[0], relu_shape[1] * relu_shape[2] * relu_shape[3]])
  
  	# Fully connected layer. Note that the '+' operation automatically
  	# broadcasts the biases.
  	hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
  
  	# Add a 50% dropout during training only. Dropout also scales
  	# activations such that no rescaling is needed at evaluation time.
  	if train:
  		hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
  	return tf.matmul(hidden, fc2_weights) + fc2_biases

s = tf.InteractiveSession()
s.as_default()

# L = .5[r + discount * max a' Q(s', a') - Q(s, a)]^2
#	     |------target-------|  |prediction|

# Do a feedforward pass for the current state s to get predicted Q-values for all actions.
action_array_1 = network(state_input_1)
# Do a feedforward pass for the next state s' and calculate maximum overall network outputs max a' Q(s', a').
action_array_2 = network(state_input_2)
max_val = tf.reduce_max(action_array_2)
# Set Q-value target for action to r + discount * max a' Q(s', a') (use the max calculated in step 2). 
# For all other actions, set the Q-value target to the same as originally returned from step 1, making the error 0 for those outputs.
tt = reward_input + tf_discount * max_val
tt = tf.reshape(tt,(BATCH_SIZE,1))
target_prep = tf.tile(tt,[1,4])
target = tf.select(action_input, target_prep, action_array_1)

# loss is .5(tt - Q(ss,aa))^2
Qerror = tf.sub(target, action_array_1)
loss = .5*tf.reduce_sum(tf.mul(Qerror, Qerror))

# Update the weights using backpropagation.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

tf.initialize_all_variables().run()

def do_action(action):
	
	reward = -World.score
	if action == actions[0]:
		World.try_move(0, -1)
	elif action == actions[1]:
		World.try_move(1, 0)
	elif action == actions[2]:
		World.try_move(0, 1)
	elif action == actions[3]:
		World.try_move(-1, 0)
	else:
		return

	#update new state after moving
	s2 = World.get_state()

	reward += World.score
	return reward, s2

def run():
    time.sleep(1.0)
    trials = 0 

    # intialize replay memory that stores experience <s,a,r,s'>
    state_data_1 = []
    max_act_data = []
    reward_data  = []
    state_data_2 = []
    e = 0.9  # the exploration value, starts high and becomes smaller
    N = 1000 # the size of the replay memory

    while trials < 100:

    	# run transitions multiple times to get collection of <s,a,r,s'> data thats equal to BATCH_SIZE

    	for i in range(BATCH_SIZE):

    		# free up replay memory if needed
    		if(len(state_data_1) > N):
    			pop_index = random.randrange(len(state_data_1))
    			state_data_1.pop(pop_index)
    			max_act_data.pop(pop_index)
    			reward_data.pop(pop_index)
    			state_data_2.pop(pop_index)
    			N -= 1

    		# update current state
    		state = World.get_state()

    		state = np.reshape(state,(9, 9, 1)).astype(np.float32)
    		state_data_1.append(state)

    		state_prep = []
    		for i in range(BATCH_SIZE):
    			state_prep.append(state)
    		feed_dict = {state_input_1: state_prep}

    		# run the CNN and get outputed max action and value based on current state
    		net_out_1 = s.run(action_array_1, feed_dict=feed_dict)
    	
    		max_index = np.argmax(net_out_1[0])
    		max_act = actions[max_index]

    		max_act_prep = np.reshape([False, False, False, False],(4)).astype(np.bool)
    		max_act_prep[max_index] = True
    		max_act_data.append(max_act_prep)

    		# help exploration early in game
    		choice = np.random.choice(2,1,p=[1-e,e])
    		if choice==1:
    			max_act = actions[np.random.choice(4,1)]
    			#print('random max action {}'.format(max_act))
    		#else:
    			#print(max_act)
    		reward, s2 = do_action(max_act)
    		reward_data.append(reward)

    		state = s2
    		state = np.reshape(state,(9, 9, 1)).astype(np.float32)
    		state_data_2.append(state)

    		# Check if the game has restarted
    		if World.has_restarted():
    			print('completed trial {}'.format(trials))
    			e *= .95 # make the exploration smaller
    			trials+=1
    			World.restart_game()

    		# MODIFY THIS SLEEP IF THE GAME IS GOING TOO FAST.
    		N += 1
    		time.sleep(0.0)

    	# update weights and minimize loss function for BATCH_SIZE amount of data points

    	update = np.arange(len(state_data_1))
    	update = random.sample(update, BATCH_SIZE)

    	s1_update = [state_data_1[index] for index in update]
    	a_update  = [max_act_data[index] for index in update]
    	r_update  = [reward_data[index] for index in update]
    	s2_update = [state_data_2[index] for index in update]

    	feed_dict = {state_input_1: s1_update, action_input: a_update, reward_input: r_update, state_input_2: s2_update }

    	_, my_loss = s.run([optimizer, loss], feed_dict=feed_dict)

    	#print('start: {}'.format(start))
    	#print('_end_: {}'.format(my_target))

    #log = open(".\optimal_policy.txt", "w")
    #print(get_policy(), file = log)


t = threading.Thread(target=run)
t.daemon = True
t.start()
World.start_game()
