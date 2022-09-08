from UCB import UCB
from Enviroment_UCB import Environment_UCB
import numpy as np
from scipy.stats import dirichlet

ucb = UCB(3, 2)
env = Environment_UCB()
T=10000
alpha_ratios = env.draw_alpha_ratios()

while ucb.t < T:

    pulled_arms = ucb.pull_arm(env.prices, env.product_sold)
    print("PULLED")
    print(pulled_arms)
    conversion = env.round(pulled_arms)
    print("CONVERSIONNNNNNNNNNNNNNNNNNNNNNN")
    print(conversion)
    ucb.update(pulled_arms, conversion, env.graph_probabilities, env.secondaries, env.lambda_p)


