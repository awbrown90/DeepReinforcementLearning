from __future__ import print_function
__author__ = 'Aaron Brown'
import World
import threading
import time
import random

discount = 0.7
actions = World.actions
states = []
gui_en = True

if not gui_en:
	World.no_gui()
	sleep_value = 0.0


#store states for positions and surrounding walls and previous state for a total of 2000 combinations if 5x5
Q = {}
for i in range(World.x):
    for j in range(World.y):
    	for up in [0,1]:
    		for right in [0,1]:
    			for down in [0,1]:
    				for left in [0,1]:
    					states.append((i, j,up,right,down,left))

for state in states:
    temp = {}
    for action in actions:
        temp[action] = random.uniform(-.1, .1) 
    Q[state] = temp

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
	x, y = World.player
	u,r,d,l = World.sense_walls()
	s2 = (x,y,u,r,d,l)
	reward += World.score
	return reward, s2

def max_Q(s):
    val = None
    act = None
    items = Q[s]
    print(World.player)
    print(Q[s])
    
    random_actions = random.sample(items,len(items))
    
    for a in random_actions:
    	q =items[a]
    	
        if val is None or (q > val):
            val = q
            act = a
    
    return act, val


def inc_Q(s, a, alpha, inc):
    Q[s][a] *= 1 - alpha
    Q[s][a] += alpha * inc

def get_policy():
	return Q

def run():
    global discount
    time.sleep(1)
    alpha = 1
    t = 1
    trials = 0
    while trials < 50:
        # Pick the right action
        x, y = World.player
        u,r,d,l = World.sense_walls()
        #print(previous)

    	s = (x,y,u,r,d,l)
        max_act, dummy = max_Q(s)
        
        (reward, s2) = do_action(max_act)
        
        #print(max_act)
        #print(reward)

        # Update Q
        dummy, max_val = max_Q(s2)
        inc_Q(s, max_act, alpha, reward + discount * max_val)

        # Check if the game has restarted
        t += 1.0
        if World.has_restarted():
        	print('completed trial {}'.format(trials))
        	trials+=1
            	World.restart_game()
            	t = 1.0

        # Update the learning rate
        #alpha = pow(t, -0.1)

        # MODIFY THIS SLEEP IF THE GAME IS GOING TOO FAST.
        time.sleep(0.5)

    log = open("./optimal_policy.txt", "w")
    print(get_policy(), file = log)


t = threading.Thread(target=run)
t.daemon = True
t.start()
World.start_game()
