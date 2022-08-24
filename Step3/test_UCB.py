from UCB import UCB
from Enviroment_UCB import Environment_UCB

ucb = UCB(5, 4)
env = Environment_UCB()
T=100
while(ucb.t<1000):

    pulled_arms = ucb.pull_arm(env.prices)
    print("PULLED")
    print(pulled_arms)
    rewards, conversion, number = env.round(pulled_arms)
    print("CONVERSIONNNNNNNNNNNNNNNNNNNNNNN")
    print(conversion)
    ucb.update(pulled_arms, conversion)


