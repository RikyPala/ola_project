import numpy as np
import matplotlib.pyplot as plt

from Environment import Environment, RoundData
from OptimizerTS import OptimizerTS


env = Environment()
learner = OptimizerTS(env)

best_configuration = (1, 2, 0, 2, 1)
rounds = []
best = []

print("===========")
print("===========")
configuration = learner.initialize_configuration()
print("PLAY: " + str(configuration))
seed = np.random.randint(1, 2**30)
round_data = env.round(configuration, seed)
best_data = env.round(best_configuration, seed)
print("REWARD: " + str(round_data.reward))
learner.update(round_data)
rounds.append(round_data)
best.append(best_data)

for i in range(200):
    print("===========")
    print("===========")
    configuration = learner.optimize_round()
    print("-")
    print("ROUND: " + str(i))
    print("PLAY: " + str(configuration))
    seed = np.random.randint(1, 2 ** 30)
    round_data = env.round(configuration, seed)
    best_data = env.round(best_configuration, seed)
    print(round_data.visits / round_data.users)
    print("REWARD: " + str(round_data.reward))
    learner.update(round_data)
    rounds.append(round_data)
    best.append(best_data)

print("===========")
print("===========")
print(learner.conversion_rates_est)
print(np.sum(env.conversion_rates * np.expand_dims(env.user_probabilities, axis=(1, 2)), axis=0))

rewards = []
best_rewards = []
for i in range(len(rounds)):
    rewards.append(rounds[i].reward)
    best_rewards.append(best[i].reward)

plt.plot(rewards)
plt.plot(best_rewards)
plt.ylabel('round reward')
plt.legend(['reward', 'best reward'], loc='upper left')
plt.show()
