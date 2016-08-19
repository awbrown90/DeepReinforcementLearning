from __future__ import print_function
__author__ = 'Aaron Brown'
import World
import threading
import time
import random
import numpy as np
import tensorflow as tf

discount = 0.3
learning_rate = .01
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

state_input_1 = tf.placeholder(
	tf.float32,
	shape=(1,9,9,1))

state_input_2 = tf.placeholder(
	tf.float32,
	shape=(1,9,9,1))

reward_input = tf.placeholder(
	tf.float32,
	shape=())

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

# Run state through CNN to get list of action values
action_array_1 = network(state_input_1)
action_array_2 = network(state_input_2)
# loss is (tt - Q(ss,aa))^2

target = tf.reduce_max(action_array_1)
tt = tf.add(tf.mul(tf.reduce_max(action_array_2),tf_discount),reward_input) 
Qerror = tf.sub(tt, target)
loss = tf.mul(Qerror, Qerror)

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
	return state, action, reward, s2

def run():
    time.sleep(1)
    trials = 0

    while trials < 100:

    	# update current state
    	state = World.get_state()

    	state_data_1 = np.reshape(state,(1, 9, 9, 1)).astype(np.float32)
    	feed_dict = {state_input_1: state_data_1 }

    	# run the CNN and get outputed max action and value based on current state
    	net_out = s.run(action_array_1, feed_dict=feed_dict)
    	max_act = actions[np.argmax(net_out)]
    	print(max_act)
    	(s1, action, reward, s2) = do_action(max_act)

    	state = s2
    	state_data_2 = np.reshape(state,(1, 9, 9, 1)).astype(np.float32)

    	feed_dict = {state_input_1: state_data_1, state_input_2: state_data_2, reward_input: reward }

    	_, my_loss = s.run([optimizer, loss], feed_dict=feed_dict)

    	
    	#print(my_loss)

    	# Check if the game has restarted
    	
    	if World.has_restarted():
    		print('completed trial {}'.format(trials))
    		trials+=1
    		World.restart_game()
    		time.sleep(0.1)
    	time.sleep(1.0)

   		# Update the learning rate
   		#
   		# MODIFY THIS SLEEP IF THE GAME IS GOING TOO FAST.

    #log = open(".\optimal_policy.txt", "w")
    #print(get_policy(), file = log)


t = threading.Thread(target=run)
t.daemon = True
t.start()
World.start_game()
