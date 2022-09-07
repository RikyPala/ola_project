import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from Environment import Environment
from UCB import UCB

env = Environment()
T = 100
n_experiments = 50
rewards_per_experiment = np.empty((n_experiments, env.n_products, T))

for e in tqdm(range(n_experiments)):

    ucb = UCB(env)
    for _ in range(T):
        pulled_arms = ucb.pull_arms()
        rewards, conversion_rates = env.round(pulled_arms)
        ucb.update(rewards, conversion_rates, pulled_arms)

    rewards_per_experiment[e, :, :] = np.array(ucb.collected_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Cumulative Rewards")
colors = ['g', 'b', 'r', 'y', 'm']
for i in range(env.n_products):
    # plt.plot(np.cumsum(np.mean(optimals[i] - rewards_per_experiment[:, i], axis=0)), colors[i])
    plt.plot(np.cumsum(np.mean(rewards_per_experiment[:, i], axis=0)), colors[i])
plt.show()
