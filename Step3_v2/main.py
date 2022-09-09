import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from Environment import Environment
from UCB import UCB

env = Environment()
T = 100
n_experiments = 3
rewards_per_exp = np.zeros((env.n_products, T, n_experiments))
optimal_rewards_per_exp = np.zeros((env.n_products, n_experiments))

for e in tqdm(range(n_experiments)):

    ucb = UCB(env)
    for _ in range(T):
        pulled_arms = ucb.pull_arms()
        rewards, conversion_rates = env.round(pulled_arms)
        ucb.update(rewards, conversion_rates, pulled_arms)

    rewards_per_exp[:, :, e] = np.array(ucb.collected_rewards)

    optimal_configuration = ucb.last_configuration
    optimal_rewards_per_exp[:, e], _ = env.round(optimal_configuration)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Cumulative Regrets")
plt.figure(2)
plt.xlabel("t")
plt.ylabel("Cumulative Rewards")

colors = ['g', 'b', 'r', 'y', 'm']

for i in range(env.n_products):
    plt.figure(0)
    ax, = plt.plot(np.cumsum(np.mean(optimal_rewards_per_exp[i, :] - rewards_per_exp[i, :, :], axis=1)), colors[i])
    ax.set_label('Product ' + str(i + 1))
    plt.figure(1)
    ax, = plt.plot(np.cumsum(np.mean(rewards_per_exp[i, :, :], axis=1)), colors[i])
    ax.set_label('Product ' + str(i + 1))

plt.figure(0)
plt.legend()
plt.figure(1)
plt.legend()
plt.show()
