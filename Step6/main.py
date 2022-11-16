import numpy as np
import matplotlib.pyplot as plt

from NonStationaryEnvironment import Environment
from CUMSUM_UCB import CUMSUM_UCB
from SW_UCB import SW_UCB
from Solver import Solver

T = 100
env = Environment(T)
solver = Solver(env)
optimal_configuration, optimal_reward = solver.find_optimal()
print(solver.conversion_rates)

print("OPTIMAL CONFIGURATION")
print(optimal_configuration)
print("OPTIMAL A-PRIORI REWARD")
print(optimal_reward)



ucb_cum_sum = CUMSUM_UCB(env)
ucb_sw = SW_UCB(env)

ucb_cum_sum_rounds = []
ucb_sw_rounds = []

optimal_rounds = []

for i in range(T):
    seed = np.random.randint(1, 2 ** 30)

    ucb_configuration_cum_sum = ucb_cum_sum.pull()
    ucb_round_data = env.round(ucb_configuration_cum_sum, seed)
    ucb_cum_sum.update(ucb_round_data)
    ucb_cum_sum_rounds.append(ucb_round_data)

    ucb_configuration_sw = ucb_sw.pull()
    ts_round_data = env.round(ucb_configuration_sw, seed)
    ucb_sw.update(ts_round_data)
    ucb_sw_rounds.append(ts_round_data)

    optimal_round_data = env.round(optimal_configuration, seed)
    optimal_rounds.append(optimal_round_data)

    print("\nROUND: " + str(i + 1))
    print("--------------------UCB---------------------")
    print("PLAYED: " + str(ucb_configuration_cum_sum))
    print("REWARD: " + str(ucb_round_data.reward))
    print("--------------------TS----------------------")
    print("PLAYED: " + str(ucb_configuration_sw))
    print("REWARD: " + str(ts_round_data.reward))
    print("------------------OPTIMAL-------------------")
    print("REWARD: " + str(optimal_round_data.reward))

print("\n###################################################")

ucb_cum_sum_rewards = []
ucb_sw_rewards = []
optimal_rewards = []

for i in range(T):
    ucb_cum_sum_rewards.append(ucb_cum_sum_rounds[i].reward)
    ucb_sw_rewards.append(ucb_sw_rounds[i].reward)
    optimal_rewards.append(optimal_rounds[i].reward)

ucb_cum_sum_rewards = np.array(ucb_cum_sum_rewards)
ucb_sw_rewards = np.array(ucb_sw_rewards)
optimal_rewards = np.array(optimal_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Rewards")
ax, = plt.plot(ucb_cum_sum_rewards, 'g')
ax.set_label("UCB")
ax, = plt.plot(ucb_sw_rewards, 'r')
ax.set_label("TS")
ax, = plt.plot(optimal_rewards, 'b--')
ax.set_label("Optimal")
plt.legend()

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Cumulative Rewards")
ax, = plt.plot(np.cumsum(ucb_cum_sum_rewards), 'g')
ax.set_label("UCB")
ax, = plt.plot(np.cumsum(ucb_sw_rewards), 'r')
ax.set_label("TS")
ax, = plt.plot(np.cumsum(optimal_rewards), 'b--')
ax.set_label("Optimal")
plt.legend()

plt.figure(2)
plt.xlabel("t")
plt.ylabel("Cumulative Regrets")
ax, = plt.plot(np.cumsum(optimal_rewards - ucb_cum_sum_rewards), 'g')
ax.set_label("UCB")
ax, = plt.plot(np.cumsum(optimal_rewards - ucb_sw_rewards), 'r')
ax.set_label("TS")
plt.legend()

plt.show()
