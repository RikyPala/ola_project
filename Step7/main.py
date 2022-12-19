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

"""
Opt. config should be:
UserType 0 [FF (class: 0) + FT (class: 1)]:     1, 0, 2, 3, 1
UserType 1 [TF (class: 2)]:                     1, 2, 3, 0, 0
UserType 2 [TT (class: 3)]:                     3, 0, 1, 1, 2

"""

print("OPTIMAL CONFIGURATION")
print(optimal_configurations)
print("OPTIMAL A-PRIORI REWARD")
print(optimal_rewards)

T = 100
split_step = 14

ucb_rounds = []
ts_rounds = []
optimal_rounds = []
learners = []

ucb_root = ContextNode(env, UCB, feature_1=None, feature_2=None)
ts_root = ContextNode(env, TS, feature_1=None, feature_2=None)

ucb_splits = []
ts_splits = []

optimal_ctx_configs = [ContextConfig(config, [user_type, user_type + 1]) if user_type == 0
                       else ContextConfig(config, [user_type + 1])
                       for user_type, config in enumerate(optimal_configurations)]

for i in range(T):
    seed = np.random.randint(1, 2 ** 30)

    ucb_leaves = ucb_root.get_leaves()
    if i > 0 and i % split_step == 0:
        for leaf in ucb_leaves:
            if leaf.split():
                ucb_splits.append(i)
        ucb_leaves = ucb_root.get_leaves()
    ucb_learners = [leaf.learner for leaf in ucb_leaves]
    ucb_ctx_configs = [ContextConfig(ucb.pull(), ucb.agg_classes) for ucb in ucb_learners]
    ucb_round_data = env.round(ucb_ctx_configs, learner_class=UCB, seed=seed)
    ucb_rounds.append(ucb_round_data)
    for ucb_learner in ucb_learners:
        ucb_learner.update(ucb_round_data)

    ts_leaves = ts_root.get_leaves()
    if i > 0 and i % split_step == 0:
        for leaf in ts_leaves:
            if leaf.split():
                ts_splits.append(i)
        ts_leaves = ts_root.get_leaves()
    ts_learners = [leaf.learner for leaf in ts_leaves]
    ts_ctx_configs = [ContextConfig(ts.pull(), ts.agg_classes) for ts in ts_learners]
    ts_round_data = env.round(ts_ctx_configs, learner_class=TS, seed=seed)
    ts_rounds.append(ts_round_data)
    for ts_learner in ts_learners:
        ts_learner.update(ts_round_data)

    optimal_round_data = env.round(optimal_ctx_configs, seed=seed)
    optimal_rounds.append(optimal_round_data)

    print("\nROUND: " + str(i + 1))
    print("--------------------UCB---------------------")
    print("PLAYED:")
    for ctx_config in ucb_ctx_configs:
        print(f"Aggregated classes: {ctx_config.agg_classes}\tConfiguration: {ctx_config.configuration}")
    print("REWARDS:")
    for user_type in range(env.n_user_types):
        print(f"UserType {user_type}: {ucb_round_data.rewards[user_type]}")
    print("--------------------TS----------------------")
    print("PLAYED:")
    for ctx_config in ts_ctx_configs:
        print(f"Aggregated classes: {ctx_config.agg_classes}\tConfiguration: {ctx_config.configuration}")
    print("REWARDS:")
    for user_type in range(env.n_user_types):
        print(f"UserType {user_type}: {ts_round_data.rewards[user_type]}")
    print("------------------OPTIMAL-------------------")
    print("PLAYED:")
    for ctx_config in optimal_ctx_configs:
        print(f"Aggregated classes: {ctx_config.agg_classes}\tConfiguration: {ctx_config.configuration}")
    print("REWARDS:")
    for user_type in range(env.n_user_types):
        print(f"UserType {user_type}: {optimal_round_data.rewards[user_type]}")

print("\n###################################################")

ucb_rewards = [np.zeros(T) for _ in range(env.n_user_types)]
ts_rewards = [np.zeros(T) for _ in range(env.n_user_types)]
optimal_rewards = [np.zeros(T) for _ in range(env.n_user_types)]

for user_type in range(env.n_user_types):
    for i in range(T):
        ucb_rewards[user_type][i] = ucb_rounds[i].rewards[user_type]
        ts_rewards[user_type][i] = ts_rounds[i].rewards[user_type]
        optimal_rewards[user_type][i] = optimal_rounds[i].rewards[user_type]

for user_type in range(env.n_user_types):

    fig = plt.figure(0, figsize=(13, 12))
    fig.add_subplot(env.n_user_types, 1, user_type + 1)
    plt.title(f'User Type: {user_type}')
    plt.xlabel("t")
    plt.ylabel("Rewards")

    ucb_y = ucb_rewards[user_type]
    ts_y = ts_rewards[user_type]
    optimal_y = optimal_rewards[user_type]
    max_y = max([*ucb_y, *ts_y, *optimal_y])

    ax, = plt.plot(ucb_y, 'g')
    ax.set_label("UCB")
    ax, = plt.plot(ts_y, 'r')
    ax.set_label("TS")
    ax, = plt.plot(optimal_y, 'b--')
    ax.set_label("Optimal")

    plt.vlines(x=ucb_splits, ymin=0, ymax=max_y, color='springgreen', label='UCB Splits')
    plt.vlines(x=ts_splits, ymin=0, ymax=max_y, color='fuchsia', label='TS Splits', linestyles='dashdot')
    plt.legend()

    fig = plt.figure(1, figsize=(13, 12))
    fig.add_subplot(env.n_user_types, 1, user_type + 1)
    plt.title(f'User Type: {user_type}')
    plt.xlabel("t")
    plt.ylabel("Cumulative Rewards")

    ucb_y = np.cumsum(ucb_rewards[user_type])
    ts_y = np.cumsum(ts_rewards[user_type])
    optimal_y = np.cumsum(optimal_rewards[user_type])
    max_y = max([*ucb_y, *ts_y, *optimal_y])

    ax, = plt.plot(ucb_y, 'g')
    ax.set_label("UCB")
    ax, = plt.plot(ts_y, 'r')
    ax.set_label("TS")
    ax, = plt.plot(optimal_y, 'b--')
    ax.set_label("Optimal")

    plt.vlines(x=ucb_splits, ymin=0, ymax=max_y, color='springgreen', label='UCB Splits')
    plt.vlines(x=ts_splits, ymin=0, ymax=max_y, color='fuchsia', label='TS Splits', linestyles='dashdot')
    plt.legend()

    fig = plt.figure(2, figsize=(13, 12))
    fig.add_subplot(env.n_user_types, 1, user_type + 1)
    plt.title(f'User Type: {user_type}')
    plt.xlabel("t")
    plt.ylabel("Cumulative Regrets")

    ucb_y = np.cumsum(optimal_rewards[user_type] - ucb_rewards[user_type])
    ts_y = np.cumsum(optimal_rewards[user_type] - ts_rewards[user_type])
    max_y = max([*ucb_y, *ts_y])

    ax, = plt.plot(ucb_y, 'g')
    ax.set_label("UCB")
    ax, = plt.plot(ts_y, 'r')
    ax.set_label("TS")

    plt.vlines(x=ucb_splits, ymin=0, ymax=max_y, color='springgreen', label='UCB Splits')
    plt.vlines(x=ts_splits, ymin=0, ymax=max_y, color='fuchsia', label='TS Splits', linestyles='dashdot')
    plt.legend()

plt.show()
