from UCB import UCB
from Enviroment_UCB import Environment_UCB
import numpy as np
from scipy.stats import dirichlet

ucb = UCB(5, 4)
env = Environment_UCB()
T=100
alpha_ratios = env.draw_alpha_ratios()

while ucb.t<10000:

    pulled_arms = ucb.pull_arm(env.prices, env.products_sold)
    print("PULLED")
    print(pulled_arms)
    conversion, alpha_ratios = env.round(pulled_arms)
    print("CONVERSIONNNNNNNNNNNNNNNNNNNNNNN")
    print(conversion)
    ucb.update(pulled_arms, conversion, alpha_ratios, env.graph_probabilities, env.secondaries, pulled_arms, env.lambda_p)


print(env.mean_dirichlet)