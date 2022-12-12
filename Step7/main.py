import numpy as np
import matplotlib.pyplot as plt

from Environment import Environment, ContextConfig
from ContextNode import ContextNode
from TS import TS
from UCB import UCB
from Solver import Solver

env = Environment()
solver = Solver(env)
optimal_configuration, optimal_reward = solver.find_optimal()
print(solver.conversion_rates)

print("OPTIMAL CONFIGURATION")
print(optimal_configuration)
print("OPTIMAL A-PRIORI REWARD")
print(optimal_reward)

T = 100
split_step = 14

ucb_learner = UCB(env)
ucb_rounds = []

ts_learner = TS(env)
ts_rounds = []

optimal_rounds = []
learners = []

ucb_root = ContextNode(env, UCB, feature_1=None, feature_2=None, delta=0.1)
ts_root = ContextNode(env, TS, feature_1=None, feature_2=None, delta=0.1)

ucb_splits = []
ts_splits = []

for i in range(T):
    seed = np.random.randint(1, 2 ** 30)

    ucb_leaves = ucb_root.get_leaves()
    if i % split_step:
        for leaf in ucb_leaves:
            if leaf.split():
                ucb_splits.append(i)
        ucb_leaves = ucb_root.get_leaves()
    ucb_learners = [leaf.learner for leaf in ucb_leaves]
    ucb_ctx_configs = [ContextConfig(ucb.pull(), ucb.agg_classes) for ucb in ucb_learners]
    ucb_round_data = env.round(ucb_ctx_configs, seed)
    ucb_learner.update(ucb_round_data)
    ucb_rounds.append(ucb_round_data)

    ts_leaves = ts_root.get_leaves()
    if i % split_step:
        for leaf in ts_leaves:
            if leaf.split():
                ts_splits.append(i)
        ts_leaves = ts_root.get_leaves()
    ts_learners = [leaf.learner for leaf in ts_leaves]
    ts_ctx_configs = [ContextConfig(ts.pull(), ts.agg_classes) for ts in ts_learners]
    ts_round_data = env.round(ts_ctx_configs, seed)
    ts_learner.update(ts_round_data)
    ts_rounds.append(ts_round_data)

    optimal_round_data = env.round(optimal_configuration, seed)
    optimal_rounds.append(optimal_round_data)

    print("\nROUND: " + str(i + 1))
    print("--------------------UCB---------------------")
    print("PLAYED: " + str(ucb_ctx_configs))
    print("REWARDS: " + str(ucb_round_data.rewards))
    print("--------------------TS----------------------")
    print("PLAYED: " + str(ts_ctx_configs))
    print("REWARDS: " + str(ts_round_data.rewards))
    print("------------------OPTIMAL-------------------")
    print("REWARD: " + str(optimal_round_data.rewards))

print("\n###################################################")

ucb_rewards = []
ts_rewards = []
optimal_rewards = []

for i in range(T):
    ucb_rewards.append(ucb_rounds[i].rewards)
    ts_rewards.append(ts_rounds[i].rewards)
    optimal_rewards.append(optimal_rounds[i].rewards)

ucb_rewards = np.array(ucb_rewards)
ts_rewards = np.array(ts_rewards)
optimal_rewards = np.array(optimal_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Rewards")
ax, = plt.plot(ucb_rewards, 'g')
ax.set_label("UCB")
ax, = plt.plot(ts_rewards, 'r')
ax.set_label("TS")
ax, = plt.plot(optimal_rewards, 'b--')
ax.set_label("Optimal")
plt.legend()

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Cumulative Rewards")
ax, = plt.plot(np.cumsum(ucb_rewards), 'g')
ax.set_label("UCB")
ax, = plt.plot(np.cumsum(ts_rewards), 'r')
ax.set_label("TS")
ax, = plt.plot(np.cumsum(optimal_rewards), 'b--')
ax.set_label("Optimal")
plt.legend()

plt.figure(2)
plt.xlabel("t")
plt.ylabel("Cumulative Regrets")
ax, = plt.plot(np.cumsum(optimal_rewards - ucb_rewards), 'g')
ax.set_label("UCB")
ax, = plt.plot(np.cumsum(optimal_rewards - ts_rewards), 'r')
ax.set_label("TS")
plt.legend()

plt.show()
