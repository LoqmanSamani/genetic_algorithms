import random


# source: https://www.youtube.com/watch?v=CRtZ-APJEKI

"""
Problem:

Imagine you have a collection of items, each represented by a number,
and you want to find the smallest combination of groups that collectively
cover all the items in the collection.

For example,you have a set of numbers:

     {1, 2, 3, 4, 5, 12, 6, 8, 9, 10, ...}

and several groups of numbers:

    {1, 2}, {3, 5, 6}, {4, 1}, {12, 14}, etc.


How can you determine the minimum number of groups needed to include all the items in the collection?

Solution: Genetic Algorithm!

By employing a genetic algorithm, we can iteratively evolve sets of groups,
optimizing them to cover the entire collection efficiently. This process
mimics the principles of natural selection, where the fittest combinations
of groups are selected and recombined to produce even better solutions over
successive generations. Eventually, the genetic algorithm converges to a
near-optimal solution, providing us with the minimum number of subsets
required to cover the entire collection.



Certainly! Here's an improved explanation:

How Genetic Algorithm Works:

1. Representation of the Problem:

   Initially, the problem is represented using a suitable encoding method. 
   In our case, we represent solutions as combinations of groups of numbers, 
   where each group represents a potential subset to cover the entire collection.

2. Reproduction:

   During the reproduction phase, solutions are combined or modified to generate new candidate solutions.
   This is typically achieved through processes like crossover, where components of two parent solutions 
   are exchanged to create offspring, or by introducing random variations to existing solutions.

3. Mutation:

   Mutation introduces diversity into the population by randomly altering individual solutions.
   This helps in exploring different regions of the solution space and prevents premature convergence 
   to suboptimal solutions. In our example, mutation could involve changing values within groups or 
   introducing entirely new groups.

4. Selection:

   Selection involves choosing the most promising solutions from the current population to proceed to the next
    generation. Solutions are evaluated using a fitness function, which measures how well they perform with 
    respect to the problem's objective. In our case, the fitness function could be defined as the number 
    of groups of numbers in each solution, aiming to minimize this value.

Fitness Function:

The fitness function serves as a measure of the quality of a solution. 
It evaluates how well a particular solution addresses the problem at hand. 
In our example, the fitness function calculates the number of groups of 
numbers in each solution. Solutions with fewer groups are considered fitter, 
as they represent a more efficient way to cover the entire collection.

By iteratively applying these steps, Genetic Algorithms explore the solution space,
gradually improving the quality of solutions until an optimal or near-optimal solution is found.
"""


population_size = 100  # number of solutions
genome_length = 20  # length of the individual solution
mutation_rate = 0.01  # the probability of occurring a mutation
crossover_rate = 0.6  # cross over rate
generations = 200  # maximum number of generation


def random_genome(length):
    bits = [random.randint(a=0, b=1) for _ in range(length)]
    return bits


def init_population(population_size, genome_length):

    populations = [random_genome(length=genome_length) for _ in range(population_size)]

    return populations


def fitness(genome):

    qualities = sum(genome)

    return qualities


def select_parent(population, fitness_values):

    total_fitness = sum(fitness_values)
    pick = random.uniform(0, total_fitness)
    current = 0

    for individual, fitness_value in zip(population, fitness_values):
        current += fitness_value
        if current > pick:
            return individual


def crossover(parent1, parent2):

    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)

        return parent1[:crossover_point] + parent2[crossover_point:], parent2[:crossover_point] + parent1[crossover_point:]

    else:
        return parent1, parent2


def mutate(genome):

    for i in range(len(genome)):
        if random.random() < mutation_rate:
            genome[i] = abs(genome[i] - 1)
    return genome


def genetic_algorithm():

    population = init_population(population_size, genome_length)
    for generation in range(generations):
        fitness_values = [fitness(genome) for genome in population]

        new_population = []
        for _ in range(population_size // 2):

            parent1 = select_parent(population, fitness_values)
            parent2 = select_parent(population, fitness_values)

            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1), mutate(offspring2)])

        population = new_population

        fitness_values = [fitness(genome) for genome in population]
        best_fitness = max(fitness_values)
        print(f"Generation: {generation}; Best Fitness: {best_fitness}")

    best_index = fitness_values.index(max(fitness_values))
    best_solution = population[best_index]
    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {fitness(best_solution)}")


if __name__ == "__main__":
    genetic_algorithm()

"""
Generation: 0; Best Fitness: 18
Generation: 1; Best Fitness: 17
Generation: 2; Best Fitness: 17
...

Generation: 197; Best Fitness: 20
Generation: 198; Best Fitness: 20
Generation: 199; Best Fitness: 20
Best Solution: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Best Fitness: 20

"""

"""
Generation: 0; Best Fitness: 15
Generation: 1; Best Fitness: 15
Generation: 2; Best Fitness: 16
...
Generation: 197; Best Fitness: 19
Generation: 198; Best Fitness: 19
Generation: 199; Best Fitness: 19
Best Solution: [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Best Fitness: 19
"""

