from abc import ABC, abstractmethod
from scipy.sparse.csgraph import dijkstra
import numpy as np
import math
from PIL import Image
from filelock import FileLock
import argparse
import os

def save_binary_array_as_image(array, filename):
	array = array.astype(np.uint8)
	array *= 255
	img = Image.fromarray(array, mode='L')
	img.save(filename)

class MDP(ABC):
	@abstractmethod
	def get_num_actions(self):
		pass
	@abstractmethod
	def get_current_observation(self):
		pass
	@abstractmethod
	def save_current_observation_as_image(self, path):
		pass
	@abstractmethod
	def step(self,action):
		pass
	@abstractmethod
	def get_current_step_count(self):
		pass
	# Returns dict of actions to dict of params of s_0 classifier to params of s_1 classifier
	# Assumes that optimal state encoders are 1-to-1 with parameters, and parameters hashable
	@abstractmethod
	def get_correct_dynamics(self, training_oracle): 
		pass


class TrainingOracle(ABC):
	@abstractmethod
	def get_trained_classifier_params_and_acc(self, class_0, class_1):
		pass
	@abstractmethod
	def apply_params(self, params, raw_samples):
		pass

	@abstractmethod
	def get_param_space_size(self): 
		pass

	# Default implementation; just uses a flat np.ndarray to store data. Some hypothesis classes may allow more efficient storage.
	class PreprocessedDataset():
		def __init__(self,samples, init_storage_size = None):

			if (isinstance(samples, np.ndarray) ):
				self.data = samples
				self.num_samples = samples.shape[0]
				self.internal_storage_size =samples.shape[0]
			elif (isinstance(samples, list) ):

				self.num_samples = len(samples)
				self.internal_storage_size =max(self.num_samples,init_storage_size if init_storage_size is not None else 0)
				if len(samples) == 0:
					self.data = None
				else:
					self.data = np.zeros([max(self.num_samples,init_storage_size)] + list(samples[0].shape),dtype=samples[0].dtype)
					for s in range(len(samples)):
						self.data[s] = samples[s]

			else:
				raise NotImplementedError()

		def get_size(self):
			return self.num_samples
		def insert_sample(self,sample):
			if (self.data is None):
				self.internal_storage_size = max(self.internal_storage_size, 1)
				self.data = np.zeros([self.internal_storage_size] + list(sample.shape),dtype=sample.dtype)
			elif (self.num_samples >= self.internal_storage_size):
				self.internal_storage_size *= 2
				new_data = np.zeros([self.internal_storage_size] + list(self.data[0].shape),dtype=self.data[0].dtype)
				new_data[:self.data.shape[0]] = self.data
				self.data = new_data
			self.data[self.num_samples] = sample
			self.num_samples  += 1
		def increase_internal_size(self, new_internal_size):
			if (self.internal_storage_size > new_internal_size):
				self.internal_storage_size = new_internal_size
				new_data = np.zeros([self.internal_storage_size] + list(self.data[0].shape),dtype=self.data[0].dtype)
				new_data[:self.data.shape[0]] = self.data
				self.data = new_data
		def merge_datasets(datasets):
			datasets_data = []
			tot_samples = 0
			tot_storage = 0
			for ds in datasets:
				if (ds.data is not None):
					datasets_data.append(ds.data[:ds.get_size()])
			out =  TrainingOracle.PreprocessedDataset([])
			if len(datasets_data) != 0:
				merged_data = np.concatenate(datasets_data)

				out.data = merged_data
				out.num_samples = merged_data.shape[0]
				out.internal_storage_size =  merged_data.shape[0]
			return out


class SparseFeatureDetector(TrainingOracle):
	def __init__(self,num_features):
		self.num_features = num_features
	def get_trained_classifier_params_and_acc(self, class_0, class_1):
		num_samples = class_0.get_size() + class_1.get_size()
		accs_by_feature = (class_0.data[:class_0.get_size()] == 0).sum(axis=0) + (class_1.data[:class_1.get_size()] == 1).sum(axis=0)
		param = np.argmax(accs_by_feature)
		acc = accs_by_feature[param]
		return param, acc/num_samples

	def apply_params(self, params, raw_samples):
		return raw_samples[...,params]

	def get_param_space_size(self): 
		return self.num_features




class EfficientSparseFeatureDetector(SparseFeatureDetector):
	def __init__(self,num_features):
		self.num_features = num_features
	def get_trained_classifier_params_and_acc(self, class_0, class_1):
		num_samples = class_0.get_size() + class_1.get_size()
		accs_by_feature = (class_0.get_size()-class_0.feature_counts) +class_1.feature_counts
		param = np.argmax(accs_by_feature)
		acc = accs_by_feature[param]
		return param, acc/num_samples

	def apply_params(self, params, raw_samples):
		return raw_samples[...,params]

	def get_param_space_size(self): 
		return self.num_features



	class PreprocessedDataset():
		def __init__(self,samples, init_storage_size = None):

			if (isinstance(samples, np.ndarray) ):
				self.feature_counts = samples.sum(axis = 0)
				self.num_samples = samples.shape[0]
			elif (isinstance(samples, list) ):

				self.num_samples = len(samples)
				if len(samples) == 0:
					self.feature_counts = None
				else:
					self.feature_counts = np.copy(samples[0])
					for s in range(1,len(samples)):
						self.feature_counts += samples[s]

			else:
				raise NotImplementedError()

		def get_size(self):
			return self.num_samples
		def insert_sample(self,sample):
			if (self.feature_counts is None):
				self.feature_counts = np.copy(sample)
			else:
				self.feature_counts += sample
			self.num_samples  += 1
		def increase_internal_size(self, new_internal_size):
			pass
		def merge_datasets(datasets):
			new_fcs = None
			new_num_samps = 0

			for ds in datasets:
				if (ds.feature_counts is not None):
					if (new_fcs is None):
						new_fcs = np.copy(ds.feature_counts)
						new_num_samps = ds.get_size()
					else:
						new_fcs += ds.feature_counts
						new_num_samps += ds.get_size()
					
			out =  EfficientSparseFeatureDetector.PreprocessedDataset([])
			out.feature_counts = new_fcs
			out.num_samples = new_num_samps
			return out



class MultiMaze(MDP):
	def __init__(self, num_distractor_mazes, noise_generator, env_generator):
		self.num_distractor_mazes = num_distractor_mazes
		self.maze = np.array([
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
			[1,1,1,1,1,1,1,1,1,1,1]], dtype=np.uint8)

		self.offs_y = {0: -1, 1:0, 2:1,3:0}
		self.offs_x = {0: 0, 1:1, 2:0,3:-1}
		self.i_poses = np.ones(num_distractor_mazes+1, dtype=np.int32)*-1
		self.j_poses = np.ones(num_distractor_mazes+1, dtype=np.int32)*-1
		self.rand_gen = noise_generator
		self.controllable_maze = env_generator.integers(num_distractor_mazes +1)
		first = True
		self.step_count = 0
		self.obs = np.concatenate([self.maze] * (1 + num_distractor_mazes))
		for idx in range(num_distractor_mazes + 1):
			while (first or self.maze[self.i_poses[idx], self.j_poses[idx]] == 1):
				gen = env_generator if idx == self.controllable_maze else self.rand_gen 
				self.i_poses[idx] =  gen.integers(self.maze.shape[0])
				self.j_poses[idx] =  gen.integers(self.maze.shape[1])
				first = False
			self.obs[idx*self.maze.shape[0] + self.i_poses[idx], self.j_poses[idx]] = 1

	def get_num_actions(self):
		return 4
	def get_current_step_count(self):
		return self.step_count

	def step(self,action):
		self.step_count += 1
		actions = self.rand_gen.integers(4,size=self.num_distractor_mazes+1)
		actions[self.controllable_maze] = action
		for m in range(self.num_distractor_mazes+1):
			self.obs[m*self.maze.shape[0] + self.i_poses[m], self.j_poses[m]] = 0
			if (self.maze[self.i_poses[m] +self.offs_y[actions[m]],self.j_poses[m] +self.offs_x[actions[m]] ] == 0):
				self.i_poses[m] += self.offs_y[actions[m]]
				self.j_poses[m] += self.offs_x[actions[m]]
			self.obs[m*self.maze.shape[0] + self.i_poses[m], self.j_poses[m]] = 1
		return self.obs.reshape(-1)

	def get_current_observation(self):
		return self.obs.reshape(-1)

	def save_current_observation_as_image(self, path):
		n = self.num_distractor_mazes+1
		sz = 1
		for i in range(math.floor(math.sqrt(n)),0,-1):
			if n % i == 0:
				sz = i
				break
		save_binary_array_as_image(self.obs.reshape( sz,n//sz ,self.maze.shape[0],self.maze.shape[1]).transpose(0,2,1,3).reshape(self.maze.shape[0]*sz, self.maze.shape[1]*(n//sz) ), path)

	def get_correct_dynamics(self, training_oracle): 
		if not isinstance(training_oracle, SparseFeatureDetector):
			raise NotImplementedError()
		else:
			out = {0:{},1:{},2:{},3:{}}
			shift = self.controllable_maze * self.maze.shape[1] *  self.maze.shape[0]
			for i in range(self.maze.shape[0]):
				for j in range(self.maze.shape[1]):
					if self.maze[i,j] == 0:
						for a in range(4):
							out[a][shift + i*self.maze.shape[1] + j] = shift + ((i+ self.offs_y[a])*self.maze.shape[1] + (j+  self.offs_x[a]) if self.maze[i+ self.offs_y[a],j+  self.offs_x[a]] == 0 else i*self.maze.shape[1] + j)
			return out

class ComboLock(MDP):
	def __init__(self, num_latents, obs_size, min_eps, noise_generator, env_generator):
		self.rand_gen = noise_generator
		self.eps_0 = env_generator.uniform(low=min_eps, high=1.-min_eps,size=obs_size)
		self.eps_1 = env_generator.uniform(low=min_eps, high=1.-min_eps,size=obs_size)
		self.latent_projection = env_generator.choice(obs_size, num_latents, replace=False)
		self.unlock_sequence =  (env_generator.random(num_latents) >= 0.5)
		self.latent = env_generator.integers(num_latents)

		init_no_mask = (self.rand_gen.random(obs_size) >= 0.5)
		init_no_mask[self.latent_projection] = 0
		init_no_mask[self.latent_projection[self.latent]] = 1
		self.obs = init_no_mask
		self.step_count = 0
	def get_current_observation(self):
		return self.obs
	def get_num_actions(self):
		return 2
	def step(self,action):
		self.step_count += 1
		switch_0_to_1 = (self.rand_gen.random(self.obs.shape[0]) < self.eps_0)
		switch_1_to_0 = (self.rand_gen.random(self.obs.shape[0]) < self.eps_1)
		self.obs = self.obs * (1-switch_1_to_0) + (1-self.obs)*switch_0_to_1
		if (self.unlock_sequence[self.latent] == action):
			self.latent = (self.latent + 1)%self.latent_projection.shape[0]
		else:
			self.latent = 0
		self.obs[self.latent_projection] = 0
		self.obs[self.latent_projection[self.latent]] = 1
		return self.obs
	def save_current_observation_as_image(self, path):
		n = self.obs.shape[0]
		sz = 1
		for i in range(math.floor(math.sqrt(n)),0,-1):
			if n % i == 0:
				sz = i
				break
		save_binary_array_as_image(self.obs.reshape( sz,n//sz), path)

	def get_current_step_count(self):
		return self.step_count

	def get_correct_dynamics(self, training_oracle): 
		if not isinstance(training_oracle, SparseFeatureDetector):
			raise NotImplementedError()
		else:
			out = {0:{},1:{}}
			for ind in range(self.latent_projection.shape[0]):
				out[self.unlock_sequence[ind]][self.latent_projection[ind]] = self.latent_projection[(ind+1)%(self.latent_projection.shape[0])] 
				out[(1-self.unlock_sequence[ind])][self.latent_projection[ind]] = self.latent_projection[0] 
			return out

def cycle_find(mdp, training_oracle, actions, size_s_pr, datasets, transition_dynamics, N, D,t_mix, delta, epsilon):
	n_samp_init = math.ceil(-math.log((training_oracle.get_param_space_size()* 4*mdp.get_num_actions()*N*(N-1))/delta)/math.log(9/16))

	n_samp = math.ceil(-math.log((training_oracle.get_param_space_size()* 4*mdp.get_num_actions()*N*N*N*N*(D+1))/delta)/math.log(9/16))
	# print("nsamp init: " + str(n_samp_init))

	# print("nsamp: " + str(n_samp))
	c_init = n_samp_init*(3*N*len(actions)-2 + 2*t_mix) + max((N-1)*len(actions),t_mix)
	# print("c_init: " + str(c_init))
	x_0 = mdp.get_current_observation()
	obs = np.zeros([c_init+1]+list(x_0.shape))
	obs[0] = x_0
	for i in range(c_init):
		obs[i + 1] = mdp.step(actions[i % len(actions)])
	def cyc_ind(i):
		return (i*len(actions) + max((N-1)*len(actions),t_mix))
	n_cyc = 1
	for n_cyc_prime in range(N,1,-1):
		q = math.ceil(t_mix/(n_cyc_prime*len(actions)))
		r = n_cyc_prime * q
		k = math.floor((c_init+r*len(actions)- max((N-1)*len(actions),t_mix))/(2*r*len(actions)+ n_cyc_prime*len(actions)))
		d_0 = training_oracle.PreprocessedDataset(obs[[cyc_ind(r + (2*r+n_cyc_prime)*i+j) for i in range(k) for j in range(1,n_cyc_prime)]])
		d_1 = training_oracle.PreprocessedDataset(obs[[cyc_ind((2*r+n_cyc_prime)*i) for i in range(k)]])
		_,acc = training_oracle.get_trained_classifier_params_and_acc( d_0, d_1)
		if (acc == 1.0):
			n_cyc =  n_cyc_prime
			break

	c= 2*n_cyc*len(actions)*((n_samp-1)*math.ceil(t_mix/(n_cyc*len(actions)))+1)  + t_mix  + max((N-n_cyc)*len(actions),t_mix)
	# print("c: " + str(c))
	c = max(c,c_init)
	obs_new = np.zeros([c+1]+list(x_0.shape))
	obs_new[:c_init+1] = obs
	obs = obs_new

	for i in range(c_init,c):
		obs[i + 1] = mdp.step(actions[i % len(actions)])


	d_prime = list([training_oracle.PreprocessedDataset([]) for i in range(n_cyc*len(actions))])
	n_0 = max((N-n_cyc)*len(actions),t_mix)
	n_0_prime = n_0 + n_cyc*len(actions)*((n_samp-1)*math.ceil(t_mix/(n_cyc*len(actions))) + 1) + t_mix 
	for k in range(n_samp):
		for i in range(n_cyc*len(actions)):
			for offset in [n_0,n_0_prime]:
				index =  n_cyc*len(actions)*math.ceil(t_mix/(n_cyc*len(actions)))*k + offset + (i - offset )% ( n_cyc*len(actions))
				d_prime[i].insert_sample(obs[index])

	s_cyc = []

	for i in range(n_cyc*len(actions) ):
		d_1 = d_prime[i]
		state_already_found = False
		for s in range(size_s_pr):
			d_0 = datasets[s]
			_,acc = training_oracle.get_trained_classifier_params_and_acc( d_0, d_1)
			if (not (acc == 1.0)):
				s_cyc.append(s)
				datasets[s] = training_oracle.PreprocessedDataset.merge_datasets([d_0,d_1])
				state_already_found = True
				break
		if (not state_already_found):
			s_prime = size_s_pr
			size_s_pr += 1
			datasets[s_prime] = d_1
			s_cyc.append(s_prime)
	for i in range(n_cyc*len(actions)):
		transition_dynamics[actions[i%len(actions)]][s_cyc[i]] = s_cyc[(i+1)%(len(actions)*n_cyc)]
	return size_s_pr, datasets, transition_dynamics, s_cyc[(c)%(len(actions)*n_cyc)]
def find_covering_cycle_at_t_mix(start_state, to_visit, transition_dynamics, dist_matrix, adj_matrix, predecessors, t_mix, mdp):
	to_visit_fixed = set(to_visit) - set([start_state])
	uncleared = set(to_visit_fixed)
	action_seq = []
	curr = start_state
	curr_idx = 0
	self_loop_dists = (adj_matrix.transpose() + dist_matrix).min(axis = 1)
	self_loop_preds = (adj_matrix.transpose() + dist_matrix).argmin(axis = 1)
	# print("adj matrix")
	# print(adj_matrix)
	# print("dist_matrix")
	# print(dist_matrix)
	# print("self_loop_dists")
	# print(self_loop_dists)
	# print("self_loop_preds")
	# print(self_loop_preds)
	shortest_self_loop_len = self_loop_dists[start_state]
	shortest_self_loop_index = 0
	shortest_self_loop_state = start_state
	while (len(uncleared) != 0):
		closest_node = min(uncleared, key=lambda i: dist_matrix[curr,i])
		assert dist_matrix[curr,closest_node] < np.inf
		backtrack_actions = []
		bt_pos = closest_node
		while (bt_pos != curr):
			pred_pos = predecessors[curr,bt_pos]
			action = None
			for ac in range(mdp.get_num_actions()):
				if ((pred_pos in transition_dynamics[ac] and transition_dynamics[ac][pred_pos] == bt_pos)) :
					action = ac
					break
			assert action is not None
			backtrack_actions = [action] + backtrack_actions
			bt_pos = pred_pos
		for a in backtrack_actions:
			curr = transition_dynamics[a][curr]
			curr_idx += 1
			if (self_loop_dists[curr]  < shortest_self_loop_len):
				shortest_self_loop_index = curr_idx
				shortest_self_loop_len = self_loop_dists[curr]
				shortest_self_loop_state = curr
			uncleared.discard(curr)
		action_seq  = action_seq + backtrack_actions

	backtrack_actions = []
	bt_pos = start_state
	while (bt_pos != curr):
		pred_pos = predecessors[curr,bt_pos]
		action = None
		for ac in range(mdp.get_num_actions()):
			if ((pred_pos in transition_dynamics[ac] and transition_dynamics[ac][pred_pos] == bt_pos)) :
				action = ac
				break
		assert action is not None
		backtrack_actions = [action] + backtrack_actions
		bt_pos = pred_pos
	for a in backtrack_actions:
		curr = transition_dynamics[a][curr]
		curr_idx += 1
		if (self_loop_dists[curr]  < shortest_self_loop_len):
			shortest_self_loop_index = curr_idx
			shortest_self_loop_len = self_loop_dists[curr]
			shortest_self_loop_state = curr
	action_seq  = action_seq + backtrack_actions
	if (len(action_seq) < t_mix):
		shortest_loop_actions = []
		first = True
		bt_pos = shortest_self_loop_state
		while (first or bt_pos != shortest_self_loop_state):
			if (first):
				pred_pos = self_loop_preds[shortest_self_loop_state]
			else:
				pred_pos = predecessors[shortest_self_loop_state,bt_pos]
			first = False
			action = None
			for ac in range(mdp.get_num_actions()):
				if ((pred_pos in transition_dynamics[ac] and transition_dynamics[ac][pred_pos] == bt_pos)) :
					action = ac
					break
			assert action is not None
			shortest_loop_actions = [action] + shortest_loop_actions
			bt_pos = pred_pos
		while (len(action_seq) < t_mix):
			action_seq = action_seq[:shortest_self_loop_index] + shortest_loop_actions + action_seq[shortest_self_loop_index:]
	return action_seq

def main_alg(mdp, training_oracle, N, D,t_mix, delta, epsilon, generator):
	size_s_pr = 0
	transition_dynamics = { a:{} for a in range(mdp.get_num_actions())}
	datasets = {}
	action_seq = [generator.integers(mdp.get_num_actions())]
	size_s_pr, datasets, transition_dynamics,s_curr = cycle_find(mdp, training_oracle, action_seq, size_s_pr, datasets, transition_dynamics, N,D, t_mix, delta, epsilon)

	while(any([  any([state not in transition_dynamics[act] for state in range(size_s_pr)] ) for act in range(mdp.get_num_actions())])):
		adj_matrix = np.ones((size_s_pr+1, size_s_pr+1))*np.inf
		for act in range(mdp.get_num_actions()):
			for s0 in range(size_s_pr):
				if s0 in transition_dynamics[act]:
					adj_matrix[s0,transition_dynamics[act][s0]] = 1
				else:
					adj_matrix[s0,size_s_pr] = 1

		dist_matrix, predecessors = dijkstra(adj_matrix, return_predecessors = True)
		action_seq = []
		uncleared_indices = set(range(size_s_pr))
		while (len(uncleared_indices) != 0):
			closest_node = min(uncleared_indices, key=lambda i: dist_matrix[i,size_s_pr])
			# print(transition_dynamics)
			# print(size_s_pr)
			# print(closest_node)
			# print(adj_matrix)
			assert dist_matrix[closest_node,size_s_pr] < np.inf
			backtrack_actions = []
			cur_pos = size_s_pr
			while (cur_pos != closest_node):
				pred_pos = predecessors[closest_node,cur_pos]
				action = None
				for ac in range(mdp.get_num_actions()):
					if ((pred_pos in transition_dynamics[ac] and transition_dynamics[ac][pred_pos] == cur_pos) or (size_s_pr == cur_pos and pred_pos not in transition_dynamics[ac])) :
						action = ac
						break
				# print(size_s_pr)
				# print(cur_pos)
				# print(pred_pos)
				# print(closest_node)
				# print(transition_dynamics)
				# print(predecessors)
				assert action is not None
				backtrack_actions = [action] + backtrack_actions
				cur_pos = pred_pos
			for act in backtrack_actions:
				new_indices = set()
				for node in uncleared_indices:
					if node in transition_dynamics[act]:
						new_indices.add(transition_dynamics[act][node])
				uncleared_indices = new_indices
			action_seq = action_seq + backtrack_actions

		size_s_pr, datasets, transition_dynamics,s_curr =  cycle_find(mdp, training_oracle, action_seq, size_s_pr, datasets, transition_dynamics, N, D,t_mix, delta, epsilon)
		print("Current T':")
		print(transition_dynamics)
		print("Current |S'|:")
		print(size_s_pr)
	# print("Cutoff before re-sample")
	# print(mdp.get_current_step_count())
	# exit()
	adj_matrix = np.ones((size_s_pr, size_s_pr))*np.inf
	for act in range(mdp.get_num_actions()):
		for s0 in range(size_s_pr):
			adj_matrix[s0,transition_dynamics[act][s0]] = 1
	dist_matrix, predecessors = dijkstra(adj_matrix, return_predecessors = True)

	d = math.ceil(3*size_s_pr*math.log(16*size_s_pr*size_s_pr*training_oracle.get_param_space_size()/delta)/epsilon)
	# print("d: " + str(d))

	for s in range(size_s_pr):
		datasets[s].increase_internal_size(d)

	while (True):
		goal_dests = []
		for s in range(size_s_pr):
			if (datasets[s].get_size() < d ):
				goal_dests.append(s)
		# print(goal_dests)
		if (len(goal_dests) == 0):
			break
		
		actions = find_covering_cycle_at_t_mix(s_curr, goal_dests, transition_dynamics, dist_matrix, adj_matrix, predecessors, t_mix, mdp)
		should_recalc = False
		first_it = True
		visited_yet  = np.zeros(size_s_pr)
		while (not should_recalc):
			visited_yet[:] = 0
			for a in actions:
				mdp.step(a)
				s_curr = transition_dynamics[a][s_curr]
				if (not first_it):
					if (datasets[s_curr].get_size()< d and visited_yet[s_curr] == 0 ):
						datasets[s_curr].insert_sample(mdp.get_current_observation())
						visited_yet[s_curr] = 1
						# if (datasets[s_curr].get_size() % 1000 == 0):
						# 	print(datasets[s_curr].get_size())
					if (datasets[s_curr].get_size() >= d and s_curr in goal_dests):
						should_recalc = True
			first_it = False
		
		

		#np.append(datasets[s_curr], mdp.get_current_observation(), axis = 0)

	state_to_classifier = {}
	for state in range(size_s_pr):
		other_datasets = list([datasets[i] for i in range(size_s_pr) if i != state])
		other_datasets = training_oracle.PreprocessedDataset.merge_datasets(other_datasets)
		param, _ = training_oracle.get_trained_classifier_params_and_acc( other_datasets, datasets[state])
		state_to_classifier[state] = param

	# Maybe split into separate method
	classifier_dynamics = {}
	for a in range(mdp.get_num_actions()):
		classifier_dynamics[a] = {}
		for state_0 in range(size_s_pr):
			classifier_0 = state_to_classifier[state_0]
			classifier_dynamics[a][classifier_0] = state_to_classifier[transition_dynamics[a][state_0]]

	return size_s_pr, state_to_classifier, transition_dynamics, classifier_dynamics




parser = argparse.ArgumentParser(description="STEEL Simulations")

# Add arguments
parser.add_argument('-E', '--env', type=str, choices=['combo_lock', "multi_maze"],default='combo_lock',  help="Environment to run")
parser.add_argument('-t', '--tmix', type=int, help='Upper bound on mixing time')
parser.add_argument('-N', '--state_N', type=int, help='Upper bound on number of states')
parser.add_argument('-D', '--d_hat', type=int, help='Upper bound on diameter of dynamics (optional)')
parser.add_argument('-d', '--delta', type=float, default = 0.05, help='Acceptable failure prob.')
parser.add_argument('-e', '--epsilon', type=float, default = 0.05, help='Acceptable encoder failure prob.')
parser.add_argument('-K', '--combo_lock_K', type=int, default = 30, help='Ground truth number of latent states in combo lock environment')
parser.add_argument('-L', '--combo_lock_L', type=int, default = 512, help='Ground truth observation size in combo lock environment')
parser.add_argument('-P', '--combo_lock_P', type=float, default = 0.1, help='Ground truth minimal transition probability of noise in combo lock environment')
parser.add_argument('-M', '--multi_maze_M', type=int, default = 8, help='Ground truth number of distractor mazes in multi maze environment')
parser.add_argument('-s', '--seed_noise', type=int, default = 0, help='Random seed for noise')
parser.add_argument('-S', '--seed_env', type=int, help='Random seed for environment setup')
parser.add_argument('-l', '--log_file', type=str, help='Log file path')


# Parse the arguments
args = parser.parse_args()
var_lf_name = False
if (args.seed_env is None):
	args.seed_env = args.seed_noise
	var_lf_name = True

if (args.log_file is None):
	args.log_file = args.env + (('_' + str(args.combo_lock_K) ) if args.env == 'combo_lock' else '')+ ('_var_env' if var_lf_name else '_fixed_env' ) + '.txt'
env_gen = np.random.default_rng(args.seed_env)
noise_gen = np.random.default_rng(args.seed_noise)



# _,_,_,learned_classifier_dynamics = main_alg(env, oracle, 40,40, 40, 0.05, 0.05,env_gen)


# env.save_current_observation_as_image("combo_lock_0.png")
# env.step(0)
# env.save_current_observation_as_image("combo_lock_1.png")

if args.env == 'multi_maze':
	env = MultiMaze(args.multi_maze_M,noise_gen,env_gen)
	oracle = EfficientSparseFeatureDetector(11*11*(args.multi_maze_M+1))
	if (args.tmix is None):
		args.tmix = 300
	if (args.state_N is None):
		args.state_N = 80

elif args.env == 'combo_lock':
	env = ComboLock(args.combo_lock_K,args.combo_lock_L,args.combo_lock_P,noise_gen,env_gen)
	oracle = EfficientSparseFeatureDetector(args.combo_lock_L)
	if (args.tmix is None):
		args.tmix = 40
	if (args.state_N is None):
		args.state_N = args.combo_lock_K + 10

if args.d_hat is None:
	args.d_hat = args.state_N

_,_,_,learned_classifier_dynamics = main_alg(env, oracle, args.state_N,args.d_hat, args.tmix, args.delta, args.epsilon, env_gen)



# env.save_current_observation_as_image("multimaze_0.png")
# env.step(0)
# env.save_current_observation_as_image("multimaze_1.png")

gt_classifier_dynamics = env.get_correct_dynamics(oracle)
# print(learned_classifier_dynamics)
# print(gt_classifier_dynamics)
print("Total steps: "+str(env.get_current_step_count()))
if (learned_classifier_dynamics == gt_classifier_dynamics):
	print("Success!")
else:
	print("Failure!")




with FileLock(args.log_file + '.lock'):
	new_file = not os.path.exists(args.log_file)
	with open(args.log_file, "a") as f:
		if new_file:
			f.write("seed,success,num_steps\n")
		f.write(str(args.seed_noise) +  (',1,'if  learned_classifier_dynamics == gt_classifier_dynamics else ',0,') + str(env.get_current_step_count()) + "\n")

