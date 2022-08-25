from UCB import UCB
from Enviroment_UCB import Environment_UCB
import numpy as np
from scipy.stats import dirichlet

ucb = UCB(5, 4)
env = Environment_UCB()
T=100
while ucb.t<10000:

    pulled_arms = ucb.pull_arm(env.prices, env.products_sold, env.mean_dirichlet)
    print("PULLED")
    print(pulled_arms)
    conversion = env.round(pulled_arms)
    print("CONVERSIONNNNNNNNNNNNNNNNNNNNNNN")
    print(conversion)
    ucb.update(pulled_arms, conversion)


print(env.mean_dirichlet)