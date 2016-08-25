__author__ = 'Aaron Brown'
from Tkinter import *
import maze_gen
master = Tk()

wall_width = 90
pip_width = 6
(x, y) = (5, 5)
actions = ["up", "right", "down", "left"]
gui_en = True

board = Canvas(master, width=(x+1)*pip_width+x*wall_width, height=(y+1)*pip_width+y*wall_width)
player = (0, y-1)
score = 0
restart = False
walk_reward = -0.1
wall_reward = -1.0
goal_reward = 10
me = 0

#wall for rows and columns
rows, columns = maze_gen.generate(5)
goal = (2, 2)

def render_grid():
	global walls, Width, x, y, player
	#creat the white base board
	board.create_rectangle(0, 0, (x+1)*pip_width+x*wall_width, (y+1)*pip_width+y*wall_width, fill="white", width=1)
	for i in range(x+1):
		for j in range(y+1):
			#create the red pips
			board.create_rectangle(i*pip_width+i*wall_width, j*pip_width+j*wall_width, (i+1)*pip_width+i*wall_width, (j+1)*pip_width+j*wall_width, fill="red", width=1)
	#create the blue row walls
	for n in range(len(rows)):
		for i in range(len(rows[n])):
			if rows[n][i] is 1:
				board.create_rectangle((i+1)*pip_width+i*wall_width, n*pip_width+n*wall_width, (i+1)*pip_width+(i+1)*wall_width, (n+1)*pip_width+n*wall_width, fill="blue", width=1)
	#create the blue column walls
	for n in range(len(columns)):
		for i in range(len(columns[n])):
			if columns[n][i] is 1:
				board.create_rectangle(i*pip_width+i*wall_width, (n+1)*pip_width+n*wall_width, (i+1)*pip_width+i*wall_width, (n+1)*pip_width+(n+1)*wall_width, fill="blue", width=1)

	board.grid(row=0, column=0)

def render_player():
	global me
	me = board.create_rectangle((player[0]+1)*pip_width+player[0]*wall_width+wall_width*1/3, (player[1]+1)*pip_width+player[1]*wall_width+wall_width*1/3,
			(player[0]+1)*pip_width+player[0]*wall_width+wall_width*2/3, (player[1]+1)*pip_width+player[1]*wall_width+wall_width*2/3, fill="black", width=1, tag="me")	

if(gui_en):
	render_grid()
	render_player()

def try_move(dx, dy):

	global player, x, y, score, walk_reward, goal_reward, me, restart
	if restart == True:
		restart_game()
	new_x = player[0] + dx
	new_y = player[1] + dy
	reward_record = 0
	

	if (new_x >= 0) and (new_x < x) and (new_y >= 0) and (new_y < y) and wall_check( player[0], player[1], dx, dy):
		if(gui_en):
			board.coords(me, (new_x+1)*pip_width+new_x*wall_width+wall_width*1/3, (new_y+1)*pip_width+new_y*wall_width+wall_width*1/3, 
						 (new_x+1)*pip_width+new_x*wall_width+wall_width*2/3, (new_y+1)*pip_width+new_y*wall_width+wall_width*2/3)
		
		
		
		score += walk_reward
		reward_record += walk_reward

		player = (new_x, new_y)
		
		if new_x == goal[0] and new_y == goal[0]:
			score -= reward_record
			score += goal_reward
			if score > 0:
				print "Success! score: ", score
			else:
				print "Fail! score: ", score
			restart = True
			return
	else:
		score += wall_reward
	
def sense_walls():
	curr_x = player[0]
	curr_y = player[1]
	up = 1 if rows[curr_y][curr_x] == 1 else 0
	right = 1 if columns[curr_y][curr_x+1] == 1 else 0
	down = 1 if rows[curr_y+1][curr_x] == 1 else 0
	left = 1 if columns[curr_y][curr_x] == 1 else 0
	return (up, right, down, left)

def wall_check(curr_x, curr_y, dx, dy):
	#if going right
	if(dx > 0):
		if columns[curr_y][curr_x+1] is not 1:
			return True
	#if going left
	elif(dx < 0):
		if columns[curr_y][curr_x] is not 1:
			return True
	#if going up
	elif(dy < 0):
		if rows[curr_y][curr_x] is not 1:
			return True
	#if going down
	else:
		if rows[curr_y+1][curr_x] is not 1:
			return True
	return False

def no_gui():
	gui_en = False

def call_up(event):
	try_move(0, -1)


def call_down(event):
	try_move(0, 1)


def call_left(event):
	try_move(-1, 0)


def call_right(event):
	try_move(1, 0)


def restart_game():
	#print "lets restart"
	global player, score, me, restart, rows, columns

	rows, columns = rows, columns = maze_gen.generate(5)
	if(gui_en):	
		render_grid()
		render_player()

	player = (0, y-1)
	score = 0
	restart = False

def has_restarted():
	return restart

master.bind("<Up>", call_up)
master.bind("<Down>", call_down)
master.bind("<Right>", call_right)
master.bind("<Left>", call_left)


def start_game():
	master.mainloop()



