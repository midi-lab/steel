# Code for the paper "Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory"

The implementation of the STEEL algorithm, and the simulation experiments in the paper, are in loop_exbmdp.py. The default parameters are set to the ones used in the paper for each experiment, so experiments can be run simply:

For example:

```
python3 loop_exbmdp.py --env combo_lock --combo_lock_K 20 --seed_noise ${SEED}
```

This will run the Combination Lock experiment, with K = 20 ground truth states and hyperparameters otherwise as set in the paper. This will append a line to the file "combo_lock_20_var_env.txt" (or create it if it doesn't already exist) that records the random seed, whether or not STEEL successfuly learned the correct gound truth dynamics and encoder (up to permutation), and the number of environment steps taken. The equivalent command for the multi-maze environment is:

```
python3 loop_exbmdp.py --env multi_maze  --seed_noise ${SEED}
```

Note that if only one random seed is specified, as above, this seed is used both for "parameters" of the experiment (such as the ground-truth latent transitions in the combination lock experiment; and the starting controllable latent state), as well as the "random" parts of the Ex-BMDP dynamics, such as the noise in the observations and exogenous latent state.  However, these seeds can be specified separately: 

```
python3 loop_exbmdp.py --env multi_maze  --seed_noise ${SEED}  --seed_env 0
```

In this case, the seed for the experimental parameters will be set to 0, while the seed for the Ex-BMDP dynamics noise will be set to ${SEED}. If the experiment parameters seed "seed_env" is specified explicitly, then results will be saved by default to "multi_maze_fixed_env.txt,"  (rather than "multi_maze_var_env.txt") and the "seed" column of the output refers to seed_noise. 

The results of all of the experiments in the paper are also provided, and can be recreated by running (on a Unix machine):

```
for i in {0..19}
do
   nohup python3 loop_exbmdp.py --env combo_lock --combo_lock_K 20 --seed_noise $i --seed_env 0 > /dev/null &
   nohup python3 loop_exbmdp.py --env combo_lock --combo_lock_K 30 --seed_noise $i --seed_env 0 > /dev/null &
   nohup python3 loop_exbmdp.py --env combo_lock --combo_lock_K 40 --seed_noise $i --seed_env 0 > /dev/null &
   nohup python3 loop_exbmdp.py --env multi_maze --seed_noise $i --seed_env 0 > /dev/null &
done
for i in {20..39}
do
   nohup python3 loop_exbmdp.py --env combo_lock --combo_lock_K 20 --seed_noise $i > /dev/null &
   nohup python3 loop_exbmdp.py --env combo_lock --combo_lock_K 30 --seed_noise $i > /dev/null &
   nohup python3 loop_exbmdp.py --env combo_lock --combo_lock_K 40 --seed_noise $i > /dev/null &
   nohup python3 loop_exbmdp.py --env multi_maze --seed_noise $i > /dev/null &
done
```

Note that this will try to run all experiments simultaniously; if running locally on a desktop, one may need to run the experiments sequentially to conserve system memory.

Other, optional command line arguments can also be provided:

STEEL Arguments:
```
--tmix # Sets upper bound on exogenous noise mixing time (default = 300 for multi_maze; 40 for combo_lock)
--state_N # Sets upper bound N on |S|, the number of controllable latent states (default = 80 for multi_maze; 10 + combo_lock_K for combo_lock)
--d_hat # Sets upper bound \hat{D} on the diameter of the controllable latent dynamics (default = state_N)
--delta # Sets upper bound on acceptable failure rate \delta for STEEL (default = .05)
--epsilon # 1-epsilon is lower bound on the accuracy of the encoder that is learned (default = .05)
```
Environment arguments:

combo_lock:
```
--combo_lock_K # Sets number of latent states (default = 30)
--combo_lock_L # Sets number of bits in each observation x (default = 512)
--combo_lock P # Sets the minimum transition probability for any of the two-state markov chains used as observation distractor noise. (default = 0.1)
```
multi_maze:
```
--multi_maze_M # Sets the number of "distractor" mazes in the observation, other than the controllable maze (default = 8)
```
Other:
```
--log_file # Override results logging file name
```

----------------------------

There is another program provided, maze_mixing_time.py, which computes the mixing time t_mix(1/32) for a single copy of the four-room maze under uniformly random actions. This is used in Appendix C in the paper to compute an upper bound on t_mix for the multi_maze environment. You can run it as:
```
python3 maze_mixing_time.py
```
