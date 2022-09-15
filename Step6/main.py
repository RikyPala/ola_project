from Non_Stationary_Environment import Non_Stationary_Environment
from Optimizer import Optimizer
from UCB import UCB

env = Non_Stationary_Environment()
optimizer = Optimizer(env,UCB((env.n_arms,)*env.n_products), 100)
T = 500
n_phase = 4
phase_len = len(T/n_phase)
n_experiments = 100
ts_rewards_per_experiment = []
swts_rewards_per_experiment = []
window_size = int(T**0.5)