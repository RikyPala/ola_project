import numpy as np
import matplotlib.pyplot as plt
from CUMSUM_UCB import CUMSUM_UCB
from NonStationaryEnvironment import Environment

from Solver import Solver


T = 1000
env = Environment(T)
solver = Solver(env)
optimal_configuration, optimal_reward = solver.find_optimal()
print(solver.conversion_rates)


print("OPTIMAL CONFIGURATION")
print(optimal_configuration)
print("OPTIMAL REWARD")
print(optimal_reward)


ucb_learner = CUMSUM_UCB(env)
ucb_rounds = []

optimal_rounds = []

for i in range(T):
    seed = np.random.randint(1, 2 ** 30)

    ucb_configuration = ucb_learner.pull()
    ucb_round_data = env.round(ucb_configuration, seed)
    ucb_learner.update(ucb_round_data)
    ucb_rounds.append(ucb_round_data)

    optimal_round_data = env.round(optimal_configuration, seed)
    optimal_rounds.append(optimal_round_data)

    print("\nROUND: " + str(i + 1))
    print("--------------------UCB---------------------")
    print("PLAYED: " + str(ucb_configuration))
    print("REWARD: " + str(ucb_round_data.reward))
    print("------------------OPTIMAL-------------------")
    print("REWARD: " + str(optimal_round_data.reward))

print("\n###################################################")

ucb_rewards = []
ts_rewards = []
optimal_rewards = []

for i in range(T):
    ucb_rewards.append(ucb_rounds[i].reward)
    #optimal_rewards.append(optimal_rounds[i].reward)

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


plt.figure(2)
plt.xlabel("t")
plt.ylabel("Cumulative Regrets")
ax, = plt.plot(np.cumsum(optimal_rewards - ucb_rewards), 'g')
ax.set_label("UCB")
ax, = plt.plot(np.cumsum(optimal_rewards - ts_rewards), 'r')
ax.set_label("TS")
plt.legend()

plt.show()
