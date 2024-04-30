import random


population_size = 100  # number of solutions
genome_length = 20  # length of the individual solution
mutation_rate = 0.01  # the probability of occurring a mutation
crossover_rate = 0.6  # cross over rate
generations = 200  # maximum number of generations


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

