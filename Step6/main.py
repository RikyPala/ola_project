import numpy as np
import matplotlib.pyplot as plt

from NonStationaryEnvironment import NonStationaryEnvironment
from CUMSUM_UCB import CUMSUM_UCB
from SW_UCB import SW_UCB
from Solver import Solver

n_experiments = 10
T = 200
env = NonStationaryEnvironment(T)
solver = Solver(env)
optimal_configurations, optimal_rewards = solver.find_optimal()

for i in range(env.n_phases):

    print("OPTIMAL CONFIGURATION Phase "+str(i))
    print(optimal_configurations[i])
    print("OPTIMAL A-PRIORI REWARD "+str(i))
    print(optimal_rewards[i])


ucb_cum_sum = CUMSUM_UCB(env)
ucb_sw = SW_UCB(env)

ucb_cum_sum_rounds = np.zeros((T+1, n_experiments))
ucb_sw_rounds = np.zeros((T+1, n_experiments))
optimal_rounds = np.zeros((T+1, n_experiments))

for t in range(n_experiments):
    for i in range(T):
        seed = np.random.randint(1, 2 ** 25)

        ucb_cum_sum_configuration = ucb_cum_sum.pull()
        ucb_cum_sum_round_data = env.round(ucb_cum_sum_configuration, seed)
        ucb_cum_sum.update(ucb_cum_sum_round_data)
        ucb_cum_sum_rounds[T, n_experiments] = ucb_cum_sum_round_data.reward

        ucb_sw_configuration = ucb_sw.pull()
        ucb_sw_round_data = env.round(ucb_sw_configuration, seed)
        ucb_sw.update(ucb_sw_round_data)
        ucb_sw_rounds[T, n_experiments] = ucb_sw_round_data.reward

        current_phase = int(env.t / env.phases_size)
        optimal_round_data = env.round(optimal_configurations[current_phase], seed, new_round=True)
        optimal_rounds[T, n_experiments] = optimal_round_data.reward

        print("\nROUND: " + str(i + 1))
        print("--------------------CUMSUM UCB---------------------")
        print("PLAYED: " + str(ucb_cum_sum_configuration))
        print("REWARD: " + str(ucb_cum_sum_round_data.reward))
        print("--------------------SW UCB----------------------")
        print("PLAYED: " + str(ucb_sw_configuration))
        print("REWARD: " + str(ucb_sw_round_data.reward))
        print("------------------OPTIMAL-------------------")
        print("REWARD: " + str(optimal_round_data.reward))

print("\n###################################################")

ucb_cum_sum_rewards_avg = []
ucb_sw_rewards_avg = []
optimal_rewards_avg = []

for i in range(T):
    q = np.mean(ucb_cum_sum_rounds[i, :])
    print("SHAPE MEAN")
    print(q.shape)
    ucb_cum_sum_rewards_avg.append(np.mean(ucb_cum_sum_rounds[i, :]))
    ucb_sw_rewards_avg.append(np.mean(ucb_sw_rounds[i, :]))
    optimal_rewards_avg.append(np.mean(optimal_rounds[i, :]))

ucb_cum_sum_rewards = np.array(ucb_cum_sum_rewards_avg)
ucb_sw_rewards = np.array(ucb_sw_rewards_avg)
optimal_rewards = np.array(optimal_rewards_avg)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Rewards")
ax, = plt.plot(ucb_cum_sum_rewards, 'g')
ax.set_label("CUMSUM UCB")
ax, = plt.plot(ucb_sw_rewards, 'r')
ax.set_label("SW UCB")
ax, = plt.plot(optimal_rewards, 'b--')
ax.set_label("Optimal")
plt.legend()

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Cumulative Rewards")
ax, = plt.plot(np.cumsum(ucb_cum_sum_rewards), 'g')
ax.set_label("CUMSUM UCB")
ax, = plt.plot(np.cumsum(ucb_sw_rewards), 'r')
ax.set_label("SW UCB")
ax, = plt.plot(np.cumsum(optimal_rewards), 'b--')
ax.set_label("Optimal")
plt.legend()

plt.figure(2)
plt.xlabel("t")
plt.ylabel("Cumulative Regrets")
ax, = plt.plot(np.cumsum(optimal_rewards - ucb_cum_sum_rewards), 'g')
ax.set_label("CUMSUM UCB")
ax, = plt.plot(np.cumsum(optimal_rewards - ucb_sw_rewards), 'r')
ax.set_label("SW UCB")
plt.legend()

plt.show()
