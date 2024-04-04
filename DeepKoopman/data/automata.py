
import cellpylib as cpl
import numpy as np

# This function is a slightly modified version of init_random from cellpylib, but with a seeded rng, to allow for 
# better reproducability
def init_random(size, seed, k=2, n_randomized=None, empty_value=0, dtype=np.int32):
    """
    Returns a randomly initialized array with values consisting of numbers in {0,...,k - 1}, where k = 2 by default.
    If dtype is not an integer type, then values will be uniformly distributed over the half-open interval [0, k - 1).

    :param size: the size of the array to be created

    :param k: the number of states in the cellular automaton (2, by default)

    :param n_randomized: the number of randomized sites in the array; this value must be >= 0 and <= size, if specified;
                         if this value is not specified, all sites in the array will be randomized; the randomized sites
                         will be centered in the array, while all others will have an empty value

    :param empty_value: the value to use for non-randomized sites (0, by default)

    :param dtype: the data type

    :return: a vector with shape (1, size), randomly initialized with numbers in {0,...,k - 1}
    """
    rng = np.random.default_rng(seed=seed)
    if n_randomized is None:
        n_randomized = size
    if n_randomized > size or n_randomized < 0:
        raise ValueError("the number of randomized sites, if specified, must be >= 0 and <= size")
    pad_left = (size - n_randomized) // 2
    pad_right = (size - n_randomized) - pad_left
    if np.issubdtype(dtype, np.integer):
        rand_nums = rng.integers(k, size=n_randomized, dtype=dtype)
    else:
        rand_nums = rng.uniform(0, k - 1, size=n_randomized).astype(dtype)
    return np.array([np.pad(np.array(rand_nums), (pad_left, pad_right), 'constant', constant_values=empty_value)])



'''
function: simulate_automata

purpose: create a trajectory of states where each one is evolved from the next 
according to a fixed cellular automata rule.

params: 
- state_size: int, how large the state is of our discrete dynamical system
- traj_len: int, the length of the trajectory that you want to simulate with automata
- rule: int, the cellular automata rule that you want to simulate 
a trajectory with (these have standard numerical names, look up on wikipedia)

returns:
- cellular_automaton: a list of lists, where each list is a state in the trajectory

'''
def simulate_automata(state_size, traj_len, rule, seed):
    # cellular_automaton = cpl.init_random(state_size)
    cellular_automaton = init_random(state_size, seed)

    cellular_automaton = cpl.evolve(cellular_automaton, timesteps=traj_len, memoize=True,
                                    apply_rule=lambda n, c, t: cpl.nks_rule(n, rule))
    return cellular_automaton
    

'''
function: gather_rule_trajectories

purpose: create a collection of trajectories. Each trajectory is a sequence of 
states where each one is evolved from the next 
according to a fixed cellular automata rule.

params: 
- state_size: int, how large the state is of our discrete dynamical system
- traj_len: int, the length of the trajectory that you want to simulate with automata
- rule: int, the cellular automata rule that you want to simulate 
a trajectory with (these have standard numerical names, look up on wikipedia)
- seed_list: list of ints, a collection of seeds, where each is used to create the random init for
the initial state of a different trajectory.

returns:
- trajs: a list of trajectories (list of lists), each seeded with a seed from seed_list

'''

def gather_rule_trajectories(state_size, traj_len, rule, seed_list):
    trajs = []
    for seed in seed_list:
        trajs.append(simulate_automata(state_size=state_size,
                                       traj_len=traj_len, rule=rule, seed=seed))
    return trajs

def plot_traj(traj):
    cpl.plot(traj)  
        

# this is just a temporary testing function to try things out quickly. Will remove later.
def test():
    trajs = gather_rule_trajectories(state_size=100, traj_len=200, rule=30, seed_list=[65, 857])
    print(trajs[0])
    print(trajs[1])
    cpl.plot(trajs[0])
    cpl.plot(trajs[1])

# uncomment this if you would like to see a plot as an example of what a trajectory simulation of cellular autmata looks like.
# test()