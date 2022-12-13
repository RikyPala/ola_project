import numpy as np
import matplotlib.pyplot as plt

from Environment import Environment, ContextConfig
from ContextNode import ContextNode
from TS import TS
from UCB import UCB
from Solver import Solver

env = Environment()
solver = Solver(env)
optimal_configurations, optimal_rewards = solver.find_optimal()

print("OPTIMAL CONFIGURATION")
print(optimal_configurations)
print("OPTIMAL A-PRIORI REWARD")
print(optimal_rewards)

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

optimal_ctx_configs = [ContextConfig(config, [user_type, user_type + 1]) if user_type == 0
                       else ContextConfig(config, [user_type + 1])
                       for user_type, config in enumerate(optimal_configurations)]

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

    optimal_round_data = env.round(optimal_configurations, seed)
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

ucb_rewards = [[] for _ in range(env.n_user_types)]
ts_rewards = [[] for _ in range(env.n_user_types)]
optimal_rewards = [[] for _ in range(env.n_user_types)]

for user_type in range(env.n_user_types):
    for i in range(T):
        if user_type == 0:
            ucb_rewards[user_type].extend([ucb_rounds[i].rewards[user_type], ucb_rounds[i].rewards[user_type + 1]])
            ts_rewards[user_type].extend([ts_rounds[i].rewards[user_type], ts_rounds[i].rewards[user_type + 1]])
            optimal_rewards[user_type].extend([optimal_rounds[i].rewards[user_type],
                                               optimal_rounds[i].rewards[user_type + 1]])
        else:
            ucb_rewards[user_type].append(ucb_rounds[i].rewards[user_type])
            ts_rewards[user_type].append(ts_rounds[i].rewards[user_type])
            optimal_rewards[user_type].append(optimal_rounds[i].rewards[user_type])

ucb_rewards = np.array(ucb_rewards)
ts_rewards = np.array(ts_rewards)
optimal_rewards = np.array(optimal_rewards)

for user_type in env.n_user_types:
    fig = plt.figure(0)
    fig.add_subplot(1, env.n_user_types, user_type)
    plt.title(f'User Type: {user_type}')
    plt.xlabel("t")
    plt.ylabel("Rewards")
    ax, = plt.plot(ucb_rewards[user_type], 'g')
    ax.set_label("UCB")
    ax, = plt.plot(ts_rewards[user_type], 'r')
    ax.set_label("TS")
    ax, = plt.plot(optimal_rewards[user_type], 'b--')
    ax.set_label("Optimal")
    plt.legend()

    plt.figure(1)
    fig.add_subplot(1, env.n_user_types, user_type)
    plt.title(f'User Type: {user_type}')
    plt.xlabel("t")
    plt.ylabel("Cumulative Rewards")
    ax, = plt.plot(np.cumsum(ucb_rewards[user_type]), 'g')
    ax.set_label("UCB")
    ax, = plt.plot(np.cumsum(ts_rewards[user_type]), 'r')
    ax.set_label("TS")
    ax, = plt.plot(np.cumsum(optimal_rewards[user_type]), 'b--')
    ax.set_label("Optimal")
    plt.legend()

    plt.figure(2)
    fig.add_subplot(1, env.n_user_types, user_type)
    plt.title(f'User Type: {user_type}')
    plt.xlabel("t")
    plt.ylabel("Cumulative Regrets")
    ax, = plt.plot(np.cumsum(optimal_rewards[user_type] - ucb_rewards[user_type]), 'g')
    ax.set_label("UCB")
    ax, = plt.plot(np.cumsum(optimal_rewards[user_type] - ts_rewards[user_type]), 'r')
    ax.set_label("TS")
    plt.legend()

plt.show()
