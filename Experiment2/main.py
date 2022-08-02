import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from Environment import Environment
from Experiment2.TS import TS
from Experiment2.UCB import UCB

n_experiments = 100
T = 20

env = Environment()

ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []

for e in tqdm(range(n_experiments)):

    ts_learner = TS(env.n_products, env.n_arms, env.n_user_types)
    # ucb_learner = UCB(env.n_products, env.n_arms, env.n_user_types)

    for t in range(T):

        ts_pulled_arms = ts_learner.pull_arms()
        rewards = env.round(ts_pulled_arms)
        ts_learner.update(ts_pulled_arms, rewards)

        # ucb_pulled_arms = ucb_learner.pull_arms()
        # ucb_rewards = env.round(ucb_pulled_arms)
        # ucb_learner.update(ucb_pulled_arms, rewards)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    # ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)

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

# TODO: fix plots
for i in range(env.n_products):
    ts_product_experiments = [elem[i] for elem in ts_rewards_per_experiment]
    print(np.shape(ts_product_experiments))
    print(np.shape(np.mean(ts_product_experiments, axis=0)))
    print(np.shape(np.cumsum(np.mean(ts_product_experiments, axis=0))))
    plt.figure(0)
    ax, = plt.plot(np.mean(ts_product_experiments, axis=0), colors[i])
    ax.set_label('Product ' + str(i+1))
    plt.figure(1)
    ax, = plt.plot(np.cumsum(np.mean(ts_product_experiments, axis=0)), colors[i])
    ax.set_label('Product ' + str(i+1))
    plt.figure(2)
    ax, = plt.plot(np.cumsum(np.mean(env.optimals[i] - ts_product_experiments, axis=0)), colors[i])
    ax.set_label('Product ' + str(i+1))
plt.figure(0)
plt.legend()
plt.figure(1)
plt.legend()
plt.figure(2)
plt.legend()
plt.show()
