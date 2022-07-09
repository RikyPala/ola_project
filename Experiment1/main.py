import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from Environment import Environment
from Learner import Learner

n_experiments = 1000

rewards_per_experiment = []
best_arms_per_experiment = []
env = Environment()

for e in tqdm(range(n_experiments)):

    learner = Learner(env.n_products, env.n_arms, env.n_user_types)

    best_arms = learner.pull_arms()
    rewards = env.round(best_arms)
    learner.update(best_arms, rewards)

    while True:
        pulled_arms = learner.pull_arms()
        rewards = env.round(pulled_arms)
        stop = learner.update(pulled_arms, rewards)
        if stop == 1:
            break
        best_arms = pulled_arms
        if stop == 2:
            break

    best_arms_per_experiment.append(best_arms)
    rewards_per_experiment.append(learner.collected_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Rewards")
plt.figure(1)
plt.xlabel("t")
plt.ylabel("Cumulative Rewards")

colors = ['g', 'b', 'r', 'y', 'm']

for i in range(env.n_products):
    product_experiments = [elem[i] for elem in rewards_per_experiment]
    max_horizon = max([len(elem) for elem in product_experiments])
    horizon = max_horizon
    mean_rewards_per_horizon = []
    #mean_regrets_per_horizon = []
    for t in range(max_horizon):
        mean_rewards_per_horizon.append(np.mean([elem[t] for elem in product_experiments if len(elem) > t]))
        #mean_regrets_per_horizon.append(np.mean([optimals[i] - elem[t] for elem in product_experiments if len(elem) > t]))
    plt.figure(0)
    plt.plot(mean_rewards_per_horizon, colors[i])
    plt.figure(1)
    plt.plot(np.cumsum(mean_rewards_per_horizon), colors[i])
plt.show()
