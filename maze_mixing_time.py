import numpy as np
# 1= Wall, 0 = Open; actions are 0 = up, 1= right, 2= down, 3 = left
def maze_to_transitions(maze):
	transitions = { a:{} for a in range(4)}
	num_states = 0
	alias = {}
	for i in range(maze.shape[0]):
		for j in range(maze.shape[1]):
			if maze[i,j] == 0:
				alias[(i,j)] = num_states
				num_states += 1
	for pair in alias:
		(i,j) = pair
		transitions[0][alias[pair]] = alias[(i-1,j)] if  (i-1,j) in alias else alias[pair]
		transitions[1][alias[pair]] = alias[(i,j+1)] if  (i,j+1) in alias else alias[pair]
		transitions[2][alias[pair]] = alias[(i+1,j)] if  (i+1,j) in alias else alias[pair]
		transitions[3][alias[pair]] = alias[(i,j-1)] if  (i,j-1) in alias else alias[pair]
	return transitions, num_states
def transitions_to_random_action_matrix(transitions, num_states):
	mat = np.zeros((num_states,num_states))
	for act in range(4):
		for s in range(num_states):
			mat[transitions[act][s]][s]+= .25
	return mat

four_room = np.array([
	[1,1,1,1,1,1,1,1,1,1,1],
	[1,0,0,0,0,1,0,0,0,0,1],
	[1,0,0,0,0,0,0,0,0,0,1],
	[1,0,0,0,0,1,0,0,0,0,1],
	[1,0,0,0,0,1,0,0,0,0,1],
	[1,1,0,1,1,1,1,1,0,1,1],
	[1,0,0,0,0,1,0,0,0,0,1],
	[1,0,0,0,0,1,0,0,0,0,1],
	[1,0,0,0,0,0,0,0,0,0,1],
	[1,0,0,0,0,1,0,0,0,0,1],
	[1,1,1,1,1,1,1,1,1,1,1]])

transitions, num_states = maze_to_transitions(four_room)
mat = transitions_to_random_action_matrix(transitions, num_states)
# Transition matrix is symmetric 
assert ((mat.transpose() == mat).all() )

# Find stationary distribution:
eigenvalues, eigenvectors = np.linalg.eig(mat)
stationary = eigenvectors[:,eigenvalues.argmax()] /eigenvectors[:,eigenvalues.argmax()].sum()
# For a random walk on this kind of maze, stationary distribution is uniform
# print(stationary)
# print(1/num_states)
assert(all(np.isclose(stationary, 1/num_states)))


# Compute t_mix(1/32) for this chain:

dist= np.identity(num_states)
diff = 1
i = 0
while diff > 1/32:
	dist = mat@dist
	i += 1
	diff = (np.absolute(dist - 1/num_states)).sum(axis=1).max()/2
	print("At t = " +str(i) + " TV from stationary is "+ str(diff))


print("t_mix(1/32): " + str(i))





