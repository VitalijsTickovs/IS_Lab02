# IS_Lab02
# GenAlgo Lab Pairs 7 

## Participants:
- Vitaly Tičkovs
- Yahor Dziomin

## Description
The project consists of 3 scripts: a genetic algorithm, 
an abstract implementation of the algorithm's logic, and 2 concrete realizations of the 
0/1 Knapsack and Travelling Salesman problems. 

You can run either Knapsack or TSP scripts. For each you can specify parameters you want. 
A solve method in each of the classes will run a specified number of genetic algorithm and will
return the best found individual and stats per population over all the generations.

EvolutionaryAlg implements common selection functions, also mutation and crossover for 
binary representations. The class receives a population and a custom fitness function. 
You can also provide the algorithm with your custom crossover and mutation functions as 
was done by us for the TSP.

Knapsack and TSP contain additional methods to perform various tests with parameters/functions.

## Sources
### ChatGPT
We used ChatGPT to generate a skeleton of the generic evolutionary 
algorithm without implementation of crossover, mutation, 
selection, and fitness functions. Also, we used ChatGPT to create a function which solves
0/1 Knapsack problem to optimality using integer linear programming. The rest of usage included 
consulting with the LLM about ideas for the most suitable crossover and selection functions
to solve TSP with an evolutionary algorithm.

### Papers and Articles
- Inspiration for TSP representation, crossovers: Hussain, A., Muhammad, Y. S., Nauman Sajid, M., Hussain, I., Mohamd Shoukry, A., & Gani, S. (2017). Genetic Algorithm for Traveling Salesman Problem with Modified Cycle Crossover Operator. Computational Intelligence and Neuroscience, 2017, 7430125. doi:10.1155/2017/7430125
- Inspiration for selection functions: https://www.linkedin.com/pulse/selections-genetic-algorithms-ali-karazmoodeh-g9yyf/


