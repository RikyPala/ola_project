import numpy as np
import matplotlib.pyplot as plt

from Environment import Environment, RoundData
from Optimizer import Optimizer
from TS import TS
from UCB import UCB
from Learner import Learner
from Solver import Solver
from Step3.Solver import Solver as Solv



env = Environment()
solv = Solv(env)
solver = Solver(env)
optimal_configuration, optimal_reward = solver.find_optimal()
print(solver.conversion_rates)

print("OPTIMAL CONFIGURATION")
print(optimal_configuration)
print("OPTIMAL REWARD")
print(optimal_reward)

T = 5000


ucb_learner = UCB(env)
rounds = []
optimal_rounds = []


"""

[[0.726575 0.625275 0.4214   0.3106  ]
 [0.754775 0.595075 0.488025 0.1755  ]
 [0.669275 0.491125 0.2106   0.1989  ]
 [0.712075 0.62765  0.53295  0.089775]
 [0.555175 0.4472   0.063    0.      ]]
 
 
configuration = learner.initialize_configuration()

seed = np.random.randint(1, 2**30)
round_data = env.round(configuration, seed)
optimal_round_data = env.round(optimal_configuration, seed)

learner.update(round_data)
rounds.append(round_data)
optimal_rounds.append(optimal_round_data)

print("ROUND: Initial")
print("PLAYED: " + str(configuration))
print("REWARD: " + str(round_data.reward))
print("OPTIMAL REWARD: " + str(optimal_round_data.reward))
"""

for i in range(T):
    pulled_arms = ucb_learner.pull()
    data = env.round(pulled_arms)
    ucb_learner.update(data)



print("OPTIMAL CONFIGURATION")
print(optimal_configuration)
print("OPTIMAL REWARD")
print(optimal_reward)

"""
    seed = np.random.randint(1, 2 ** 30)
    round_data = env.round(configuration, seed)
    optimal_round_data = env.round(optimal_configuration, seed)

    learner.update(round_data)
    rounds.append(round_data)
    optimal_rounds.append(optimal_round_data)

    print("\n-----------------------------------------")
    print("ROUND: " + str(i+1))
    print("PLAYED: " + str(configuration))
    print("REWARD: " + str(round_data.reward))
    print("OPTIMAL REWARD: " + str(optimal_round_data.reward))

print("\n###################################################")

print("\nESTIMATED CONVERSION RATES: ")
print(learner.conversion_rates_est)
print("\nAGGREGATED ESTIMATED CONVERSION RATES: ")
print(np.sum(env.conversion_rates * np.expand_dims(env.user_probabilities, axis=(1, 2)), axis=0))

rewards = []
optimal_rewards = []
for i in range(T):
    rewards.append(rounds[i].reward)
    optimal_rewards.append(optimal_rounds[i].reward)

rewards = np.array(rewards)
optimal_rewards = np.array(optimal_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Rewards")
ax, = plt.plot(rewards, 'g')
ax.set_label("Played")
ax, = plt.plot(optimal_rewards, 'b')
ax.set_label("Optimal")
plt.legend()

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Cumulative Rewards")
ax, = plt.plot(np.cumsum(rewards), 'g')
ax.set_label("Played")
ax, = plt.plot(np.cumsum(optimal_rewards), 'b')
ax.set_label("Optimal")
plt.legend()

plt.figure(2)
plt.xlabel("t")
plt.ylabel("Cumulative Regrets")
plt.plot(np.cumsum(optimal_rewards - rewards))

plt.show()
"""
dict = solv.find_optimal_arm()
Keymax1 = max(zip(dict.values(), dict.keys()))[1]
print(Keymax1)
print(dict)