import numpy as np

from Environment import Environment, RoundData
from OptimizerTS import OptimizerTS


env = Environment()
learner = OptimizerTS(env)

print("===========")
print("===========")
configuration = learner.initialize_configuration()
print("PLAY: " + str(configuration))
round_data = env.round(configuration)
print("REWARD: " + str(round_data.reward))
learner.update(round_data)

for _ in range(100):
    print("===========")
    print("===========")
    configuration = learner.optimize_round()
    print("-")
    print("PLAY: " + str(configuration))
    round_data = env.round(configuration)
    print("REWARD: " + str(round_data.reward))
    learner.update(round_data)

print("===========")
print("===========")
print(learner.conversion_rates_est)
print(np.sum(env.conversion_rates * np.expand_dims(env.user_probabilities, axis=(1, 2)), axis=0))
