import numpy as np
import matplotlib.pyplot as plt

from Environment import Environment, RoundData
from Optimizer import Optimizer
from TS import TS
from Step3.Solver import Solver
from Solver import Solver as Solver1


env = Environment()
solver = Solver(env)
solver1 = Solver1(env)
dict = solver.find_optimal_arm()
Keymax = max(zip(dict.values(), dict.keys()))[1]

conf = solver1.optimize()
print(conf)
print(Keymax)
print(dict)


arms_shape = (env.n_arms,) * env.n_products
T = 0
best_configuration = (1, 2, 0, 2, 1)

learner = Optimizer(env, TS(arms_shape, gamma_rate=50000., prior_mean=500.))
rounds = []
optimal_rounds = []

configuration = learner.initialize_configuration()

seed = np.random.randint(1, 2**30)
round_data = env.round(configuration, seed)
optimal_round_data = env.round(best_configuration, seed)

learner.update(round_data)
rounds.append(round_data)
optimal_rounds.append(optimal_round_data)

print("ROUND: Initial")
print("PLAYED: " + str(configuration))
print("REWARD: " + str(round_data.reward))
print("OPTIMAL REWARD: " + str(optimal_round_data.reward))

for i in range(T):

    configuration = learner.optimize_round()

    seed = np.random.randint(1, 2 ** 30)
    round_data = env.round(configuration, seed)
    optimal_round_data = env.round(best_configuration, seed)

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
