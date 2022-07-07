import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from Environment import Environment
from GreedyLearner import GreedyLearner

n_products = 5
n_arms = 4
T = 300
n_experiments = 1000
rewards_per_experiment = []

probabilities = np.array([[0.15, 0.10, 0.10, 0.35],
                          [0.10, 0.15, 0.35, 0.10],
                          [0.35, 0.10, 0.15, 0.10],
                          [0.15, 0.35, 0.10, 0.10],
                          [0.35, 0.10, 0.10, 0.15]])

optimals = np.amax(probabilities, axis=1)

for e in tqdm(range(n_experiments)):

    env = Environment(n_products, n_arms, probabilities)
    gr_learner = GreedyLearner(n_products, n_arms)

    for t in range(0, T):
        pulled_arms = gr_learner.pull_arm()
        rewards = env.round(pulled_arms)
        gr_learner.update(pulled_arms, rewards)

    rewards_per_experiment.append(gr_learner.collected_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
colors = ['g', 'b', 'r', 'y', 'm']
for i in range(n_products):
    plt.plot(np.cumsum(np.mean(optimals[i] - rewards_per_experiment[i], axis=0)), colors[i])
plt.show()
