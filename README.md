# eCommerce Website | OLA-Project

## Project Specifications

Imagine an ecommerce website which can sell an unlimited number of units of 5 different items without any storage cost.

In every webpage, a single product, called *primary*, is displayed together with its price. The user can add a number of units of this product to the cart. After the product has been added to the cart, two products, called *secondary*, are recommended. When displaying the secondary products, the price is hidden. Furthermore, the products are recommended in two slots, one above the other, thus providing more importance to the product displayed in the slot above. If the user clicks on a secondary product, a new tab on the browser is opened and, in the loaded webpage, the clicked product is displayed as primary together with its price. At the end of the visit over the ecommerce website, the user buys the products added to the cart.

We assume that the user has the following behavior:
- she/he buys a number of units of the primary product if the price of a single unit is under the user’ reservation price; in other words, the users’ reservation price is not over the cumulative price of the multiple units, but only over the single unit;
- once the primary product has been bought, the user clicks on a secondary product with a probability depending on the primary product except when the secondary product has been already displayed as primary in the past, in this case the click probability is zero (thus, in practice, excluding the case in which a product is displayed as primary more than once --- as a result the number of webpages visited by the user is finite);
- when observing the secondary products, the user initially observes the first slot and, after having observed that slot, observes the second slot. Assume that the probability with which the first slot is observed is 1, while the probability with which the second slot is observed is lambda < 1. The value of lambda is assumed to be known in all the three project proposals.

## Example

<img src='https://i.postimg.cc/kM8DV8Nc/a.png' border='0' alt='a'/>

We are given five products P1, P2, P3, P4, P5, each corresponding to a node of the graph above. Every directed edge between two products is associated with a weight. Given a primary product P and a secondary product P’, the weight of the edge starting from P and ending in P’ provides the probability with which the user clicks on P’ while displayed in the first slot with product P as the main product. For instance, if the primary product is P2 and the secondary products are P1, in the first slot, and P4, in the second slot, the probabilities that the user clicks on P1 and P4 are 0.1 and 0.3 * lambda. The absence of an edge means that the corresponding probability is zero. Consider now another case. When the primary product is P2 and the secondary products are P1 and P4, the user can buy a number of units of P2 and then click on one of both secondary products. Assume, for instance, that the user clicks on P4 and therefore P4 is displayed as the primary product, along with, e.g., P1 and P3 displayed as secondary products. Notice that, from now on, if P2 is displayed as a secondary product, the probability that the user clicks on it is zero independently of the slot in which it is displayed.

Assumptions:
- the clicks on the secondary products are independent of each other and the user can click on multiple secondary products, so as to activate multiple parallel paths over the graph depicted above;
- the number of items a user will buy is a random variable independent of any other variable; that is, the user decides first whether to buy or not the products and, subsequently, in the case of a purchase, the number of units to buy;
- the actions performed by the users are perfectly observable by the ecommerce website.

Every day, there is a random number of potential new customers (returning customers are not considered here). In particular, every single customer can land on the webpage in which one of the 5 products is primary or on the webpage of a product sold by a (non-strategic) competitor. Call 𝛼_i the ratio of customers landing on the webpage in which product Pi is primary, and call 𝛼_0 the ratio of customers landing on the webpage of a competitor. In practice, you can only consider the 𝛼 ratios and disregard the total number of users. However, the 𝛼 ratios will be subject to noise. That is, every day, the value of the 𝛼 ratios will be realizations of independent Dirichlet random variables.

The following picture summarizes the overall scenario.

<img src="https://i.ibb.co/YBrQrvK/Picture2.png" alt="Picture2" border="0">

## Assignment

Consider the scenario in which:
- for every primary product, the pair and the order of the secondary products to display is fixed by the business unit and cannot be controlled,
- the price of every primary product is a variable to optimize,
- the expected values of the 𝛼 ratios are known.

For simplicity, assume that there are four values of price for every product and that the price can be changed once a day. For every product, order the prices in increasing levels. Every price is associated with a known margin. On the other hand, for every product, the conversion probability associated with each price value is a random variable whose mean is unknown. 

- **Step 1**: *Environment*. Develop the simulator by Python. In doing that, imagine a motivating application and specify an opportune choice of the probability distributions associated with every random variable. Moreover, assume that there are 2 binary features that define 3 different user classes. The users’ classes potentially differ for the demand curves of the 5 products, number of daily users, 𝛼 ratios, number of products sold, and graph probabilities. That is, for every random variable, you need to provide three different distributions, each one corresponding to a different users’ class.
- **[Step 2](https://github.com/riccardo-pala/OLA-Project/tree/master/Step2)**: *Optimization algorithm*. Formally state the optimization problem where the objective function is the maximization of the cumulative expected margin over all the products. Design a greedy algorithm to optimize the objective function when all the parameters are known. The algorithm works as follows. At the beginning, every item is associated with the corresponding lowest price. Then, evaluate the marginal increase obtained when the price of a single product is increased by a single level, thus considering 5 potential different price configurations at every iteration, and choose the price configuration providing the best marginal increase (a price configuration specifies the price of every product). The algorithm stops when no new configuration among the 5 evaluated is better than the previous one. For instance, at the beginning, evaluate the 5 price configurations in which all the products are priced with the lowest price except for one product which is priced with the second lowest price. If all these price configurations are worse than the configuration in which all the products are priced with the lowest price, stop the algorithm and return the configuration with the lowest price for all the products. Otherwise, choose the best price configuration and re-iterate the algorithm. Notice that the algorithm monotonically increases the prices as well as the cumulative expected margin. Therefore, the algorithms cannot cycle. However, there is not guarantee that the algorithm will return the optimal price configuration. Develop the algorithm by Python.
- **[Step 3](https://github.com/riccardo-pala/OLA-Project/tree/master/Step3)**: *Optimization with uncertain conversion rates*. Focus on the situation in which the binary features cannot be observed and therefore data are aggregated. Design bandit algorithms (based on UCB and TS) to face the case in which the conversion rates are unknown. Develop the algorithms by Python and evaluate their performance when applied to your simulator.
- **[Step 4](https://github.com/riccardo-pala/OLA-Project/tree/master/Step4)**: *Optimization with uncertain conversion rates, 𝛼 ratios, and number of items sold per product*. Do the same of Step 3 when also the alpha ratios and the number of items sold per product are uncertain. Develop the algorithms by Python and evaluate their performance when applied to your simulator.
- **[Step 5](https://github.com/riccardo-pala/OLA-Project/tree/master/Step5)**: *Optimization with uncertain graph weights*. Do the same as Step 3 when the uncertain parameters are the graph weights. Develop the algorithms by Python and evaluate their performance when applied to your simulator.
- **[Step 6](https://github.com/riccardo-pala/OLA-Project/tree/master/Step6)**: *Non-stationary demand curve*. Now assume that the demand curves could be subjected to abrupt changes. Use a UCB-like approach with a change detection algorithm to face this situation and show whether it works better or worse than using a sliding-window UCB-like algorithm. Develop the algorithms by Python and evaluate their performance when applied to your simulator.
- **[Step 7](https://github.com/riccardo-pala/OLA-Project/tree/master/Step7)**: *Context generation*. Do the same of Step 4 when the features can be observed by the ecommerce website. For simplicity, run the context-generation algorithms only every 2 weeks. When we have multiple contexts, the prices of each single context can be chosen and thus optimized independently of the others. Develop the algorithms by Python and evaluate their performance when applied to your simulator.

For the Steps 3-7, in the algorithm evaluation, report:
- the average regret and reward computed over a significant number of runs,
- their standard deviation, 
- when theoretical bounds are available, also report the ratio between the empiric regret and the upper bound.
