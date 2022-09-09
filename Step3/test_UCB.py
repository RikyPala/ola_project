from UCB import UCB
from Enviroment_UCB import Environment_UCB
import itertools
import numpy as np
from scipy.stats import dirichlet




ucb = UCB(3, 2)
env = Environment_UCB()
T = 10000
alpha_ratios = env.draw_alpha_ratios()

def find_optimal_arm(env):
    # print(''.join(map(str,configuration)))
    dict = {}
    for configuration in itertools.product([0,1], [0,1], [0,1]):

        configuration = np.asarray(configuration)
        print(configuration)
        print(configuration[2])
        reward = np.zeros(3)
        for i in range(len(configuration)): # number of product
            common_term = env.conversion_rates[configuration[i]][i] * env.prices[configuration[i]] * env.product_sold[i][configuration[i]]
            print(common_term)
            reward[i] += env.expected_alpha_ratios[i]*common_term
            parents_lv2 = find_parents_lv1(env, i, configuration, reward, common_term)
            compute_parents_lv2(parents_lv2, env, i, configuration, reward, common_term)
        dict[''.join(map(str, configuration))] = np.sum(reward)

    return dict


def find_parents_lv1(env, i, configuration, reward, common_term):
    parents_lv2 = []
    for k in range(len(configuration)):
        print(k)
        if env.secondaries[k][0] == i:
            reward[i] += env.expected_alpha_ratios[k]*\
                         env.conversion_rates[configuration[k]][k]*\
                         env.graph_probabilities[k][i]*common_term
            parents_lv2.append([k, 0])
        elif env.secondaries[k][1] == i:
            reward[i] += env.expected_alpha_ratios[k]*env.conversion_rates[configuration[k]][k]* \
                         env.lambda_p*env.graph_probabilities[k][i]*common_term
            parents_lv2.append([k, 1])
    return parents_lv2

def compute_parents_lv2(parents_lv2, env, i, configuration, reward, common_term):

    for parent in parents_lv2:
        for k in range(len(configuration)):

            if env.secondaries[k][0] == parent[0] and parent[1] == 0:

                reward += env.expected_alpha_ratios[k]*\
                          env.conversion_rates[configuration[k]][k]\
                          *env.graph_probabilities[k][parent[0]]*\
                          env.conversion_rates[configuration[parent[0]]][parent[0]]*\
                          env.graph_probabilities[parent[0]][i]*\
                          common_term
            elif env.secondaries[k][1] == parent[0] and parent[1] == 0:
                reward += env.expected_alpha_ratios[k]\
                          *env.conversion_rates[configuration[k]][k]\
                          *env.graph_probabilities[k][parent[0]]\
                          *env.lambda_p*\
                          env.conversion_rates[configuration[parent[0]]][parent[0]]*\
                          env.graph_probabilities[parent[0]][i]*\
                          common_term
            elif env.secondaries[k][0] == parent[0] and parent[1] == 1:
                reward += env.expected_alpha_ratios[k]*\
                          env.conversion_rates[configuration[k]][k]*\
                          env.graph_probabilities[k][parent[0]]*\
                          env.conversion_rates[configuration[parent[0]]][parent[0]]*\
                          env.graph_probabilities[parent[0]][i]*\
                          env.lambda_p*\
                          common_term
            elif env.secondaries[k][1] == parent[0] and parent[1] == 1:
                reward += env.expected_alpha_ratios[k]*\
                          env.conversion_rates[configuration[k]][k]*\
                          env.graph_probabilities[k][parent[0]]*\
                          env.lambda_p*\
                          env.conversion_rates[configuration[parent[0]]][parent[0]]*\
                          env.graph_probabilities[parent[0]][i]*\
                          env.lambda_p*common_term

dict = find_optimal_arm(env)
print(dict)




while ucb.t < T:

    pulled_arms = ucb.pull_arm(env.prices, env.product_sold)
    print("PULLED")
    print(pulled_arms)
    conversion = env.round(pulled_arms)
    print("CONVERSIONNNNNNNNNNNNNNNNNNNNNNN")
    print(conversion)
    ucb.update(pulled_arms, conversion, env.graph_probabilities, env.secondaries, env.lambda_p)


Keymax = max(zip(dict.values(), dict.keys()))[1]
print(Keymax)