import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from Environment import Environment
from Learner import Learner

n_products = 5
n_arms = 4

rewards_per_experiment = []

probabilities = np.array([[0.15, 0.10, 0.10, 0.35],
                          [0.10, 0.15, 0.35, 0.10],
                          [0.35, 0.10, 0.15, 0.10],
                          [0.15, 0.35, 0.10, 0.10],
                          [0.35, 0.10, 0.10, 0.15]])

optimals = np.amax(probabilities, axis=1)

for e in tqdm(range(n_experiments)):

    env = Environment()
    learner = Learner(n_products, n_arms)

    for t in range(0, T):
        pulled_arms = learner.pull_arm()
        rewards = env.round(pulled_arms)
        stop = learner.update(pulled_arms, rewards)
        if stop:
            break

    rewards_per_experiment.append(learner.collected_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
colors = ['g', 'b', 'r', 'y', 'm']
for i in range(n_products):
    product_experiments = [elem[i] for elem in rewards_per_experiment]
    max_horizon = max([len(elem) for elem in product_experiments])
    horizon = max_horizon
    mean_regrets_per_horizon = []
    for t in range(max_horizon):
        mean_regrets_per_horizon.append(np.mean([optimals[i] - elem[t] for elem in product_experiments if len(elem) > t]))
    plt.plot(np.cumsum(mean_regrets_per_horizon), colors[i])
plt.show()
