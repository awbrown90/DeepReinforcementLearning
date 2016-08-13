__author__ = 'Aaron Brown'
#Class to generate random mazes of some nxn dimensions
import numpy as np
import random

rows = []
columns = []
cells = []
walls = 0

class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)

class MazeCell:
	def __init__(self,x,y):
		self.x = x
    		self.y = y
        	self.visited = False

#generate a random maze
def generate(maze_size):
	
	global walls
	global rows
	global columns 
	global cells 

	walls = maze_size
	rows = [[1 for i in range(walls)] for j in range(walls+1)] 
	columns = [[1 for i in range(walls+1)] for j in range(walls)] 

	# If maze_size is 5
	#rows = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
	#columns = [[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]]

	# Create the cells that shows visited or not
	for y in range(walls):
		for x in range(walls):
			cells.append(MazeCell(x,y))

	cell_stack = Stack()
	unvistedCells = len(cells)
	currentCell = 0
	cells[currentCell].visited = True
	unvistedCells -= 1
	
	#While there are unvisited cells
	while (unvistedCells > 0):
		nextCell = chooseUnvisitedNeighbor(currentCell)
		if(nextCell != -1):
			cell_stack.push(currentCell)
			#remove the wall in between currentCell and nextCell
			removeWall(currentCell,nextCell)
			currentCell = nextCell
			cells[currentCell].visited = True	
			unvistedCells -= 1
		elif(cell_stack.size() > 0):
			currentCell = cell_stack.pop()
	
	cells = [] #reset cells for when method is called again
	return rows, columns

def chooseUnvisitedNeighbor(currentCell):
	x = cells[currentCell].x
	y = cells[currentCell].y
	
	candidates = []

	# left
	if(x > 0 and cells[currentCell-1].visited is False):
		candidates.append(currentCell-1)
	# right
	if(x < (walls-1) and cells[currentCell+1].visited is False):
		candidates.append(currentCell+1)
	# up
	if(y > 0 and cells[currentCell-walls].visited is False):
		candidates.append(currentCell-walls)	
	# down
	if(y < (walls-1) and cells[currentCell+walls].visited is False):
		candidates.append(currentCell+walls)

	if(len(candidates) == 0):
		#print "no choice"
		return -1

	#choose a random candidate
	random_choice = random.sample(candidates,len(candidates))
	#print random_choice[0]
	return random_choice[0]

def removeWall(currentCell,nextCell):

	global columns
	global rows

	#remove column to the right of currentCell
	if(nextCell-currentCell == 1):
		columns[currentCell/walls][currentCell%walls+1] = 0
		#print "right"
	#remove column to the left of currentCell
	elif(currentCell - nextCell == 1):
		columns[currentCell/walls][currentCell%walls] = 0
		#print "left"
	#remove row above currentCell
	elif(currentCell - nextCell == walls):
		rows[currentCell/walls][currentCell%walls] = 0
		#print "up"
	#remove row below currentCell
	elif(nextCell - currentCell == walls):
		rows[currentCell/walls+1][currentCell%walls] = 0
		#print "down"
