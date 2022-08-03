import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from Environment import Environment
from Learner import Learner

n_experiments = 1000

rewards_per_experiment = []
env = Environment()

for e in tqdm(range(n_experiments)):

    learner = Learner(env.n_products, env.n_arms, env.n_user_types)

    while True:
        configurations = learner.pull_arms()
        for pulled_arms in configurations:
            rewards = env.round(pulled_arms)
            stop = learner.update(pulled_arms, rewards)
            if stop:
                break

    rewards_per_experiment.append(learner.collected_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Rewards")
plt.figure(1)
plt.xlabel("t")
plt.ylabel("Cumulative Rewards")
plt.figure(2)
plt.xlabel("t")
plt.ylabel("Cumulative Regrets")

colors = ['g', 'b', 'r', 'y', 'm']

for i in range(env.n_products):
    product_experiments = [elem[i] for elem in rewards_per_experiment]
    max_horizon = max([len(elem) for elem in product_experiments])
    horizon = max_horizon
    mean_rewards_per_horizon = []
    mean_regrets_per_horizon = []
    for t in range(max_horizon):
        mean_rewards_per_horizon.append(np.mean([elem[t] for elem in product_experiments if len(elem) > t]))
        mean_regrets_per_horizon.append(np.mean([env.optimals[i] - elem[t] for elem in product_experiments if len(elem) > t]))
    plt.figure(0)
    ax, = plt.plot(mean_rewards_per_horizon, colors[i])
    ax.set_label('Product ' + str(i+1))
    plt.figure(1)
    ax, = plt.plot(np.cumsum(mean_rewards_per_horizon), colors[i])
    ax.set_label('Product ' + str(i+1))
    plt.figure(2)
    ax, = plt.plot(np.cumsum(mean_regrets_per_horizon), colors[i])
    ax.set_label('Product ' + str(i+1))
plt.figure(0)
plt.legend()
plt.figure(1)
plt.legend()
plt.figure(2)
plt.legend()
plt.show()
