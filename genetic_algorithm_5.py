import random
import numpy as np



def decimal_to_binary(array, precision_bits):
    """
    Convert a  NumPy array to a binary string.

    Args:
        - array (numpy.ndarray): an arrays of float or int values.
        - precision_bits (tuple): tuple containing (min_val, max_val, bits) to define the precision for array.
            - min_val (float): The minimum possible value of the original range for the element.
            - max_val (float): The maximum possible value of the original range for the element.
            - bits (int): The number of bits used to represent the element in the binary string.

    Returns:
        - binary_string (str): a binary string representing the array.
    """

    min_val, max_val, bits = precision_bits

    if max_val == min_val:
        # All values will be the same in this case, handle gracefully by converting to '0' * bits
        binary_string = ''.join(
            '0' * bits
            for _ in array
        )
    else:
        binary_string = ''.join(
            f"{int((val - min_val) / (max_val - min_val) * ((1 << bits) - 1)):0{bits}b}"
            if not np.isnan(val) else '0' * bits  # Handle NaN by converting to '0' * bits
            for val in array
        )

    return binary_string


"""
arr = np.array([1,2,3,4,5,6,7])
b_arr = decimal_to_binary(arr, (0, 10, 8))
print(arr)
print(b_arr)
print(len(b_arr))

# [1 2 3 4 5 6 7]
# 00011001001100110100110001100110011111111001100110110010
# 56
"""


def extract_based_on_max_index(list1, list2):
    """
    Extract an object from list1 based on the index of the maximum value in list2.

    Args:
        list1 (list): The list from which to extract the object.
        list2 (list): The list used to determine the index of the maximum value.

    Returns:
        object: The object from list1 corresponding to the index of the maximum value in list2.
    """
    max_index = list2.index(max(list2))

    return list1[max_index]




def mutate(chromosome, mutation_rate):

    """
    Mutate a chromosome  based on its mutation rate.

    Args:
        - chromosome (str): a binary strings representing the chromosome,
        - mutation_rate (float): The probability of mutating each bit in chromosome.

    Returns:
        - mutated_chromosome (str): The mutated chromosome

    Example:
        chromosome = '100001110010101010110'
        mutation_rate = 0.1
        mutated_chromosome = mutate(chromosome, mutation_rates)
    """
    mutated_chromosome = ''.join(
        '1' if bit == '0' and random.random() <= mutation_rate else
        '0' if bit == '1' and random.random() <= mutation_rate else
        bit
        for bit in chromosome
    )

    return mutated_chromosome

"""
chr = "111100000010101001111111"
m_chr = mutate(chr, 0.1)
print(chr)
print(m_chr)
# 111100000010101001111111
# 111110100000100001111111
"""



def crossover(parent1, parent2, crossover_rate, num_crossover_points):
    """
    Perform multi-point crossover between two parents to generate two offspring.

    Args:
        parent1 (str): A binary string representing the first parent.
        parent2 (str): A binary string representing the second parent.
        crossover_rate (float): Probability of performing crossover for the chromosome.
        num_crossover_points (int): An integer representing the crossover points for the chromosome.

    Returns:
        tuple: Two binary strings representing the offspring.
    """
    if random.random() < crossover_rate:
        crossover_points = sorted(random.sample(range(1, len(parent1)), num_crossover_points))
        child1, child2 = "", ""
        start = 0

        for i, point in enumerate(crossover_points + [len(parent1)]):
            if i % 2 == 0:
                child1 += parent1[start:point]
                child2 += parent2[start:point]
            else:
                child1 += parent2[start:point]
                child2 += parent1[start:point]
            start = point

        offspring1, offspring2 = child1, child2
    else:
        offspring1, offspring2 = parent1, parent2

    return offspring1, offspring2

"""
p1 = "1111111110000000000000000001111111"
p2 = "0000000000000000000000000000000111"

o1, o2 = crossover(p1, p2, 0.80, 2)
print(o1)
print(o2)
# 1111100000000000000000000000000111
# 0000011110000000000000000001111111
"""


def initialize_population(pop_size, bit_length):

    """
    Initialize a population for a genetic algorithm.

    Args:
        - pop_size (int): The number of individuals in the population.
        - bit_length (int): The length of the binary string representing each individual.

    Returns:
        - population (list): A list of binary strings, each representing an individual in the population.
    """
    population = [
        ''.join(random.choice('01') for _ in range(bit_length)) for _ in range(pop_size)]

    return population

"""
p = initialize_population(20, 30)
print(p)
# ['001001101011001100001000100011', '010000010100111101101101011000',
# '110111110011100011010110111001', '001011100010000110011010011101', 
# '010101000100010010011001011001', '001011101101100111010010010010',
# '100101010000010100000010001111', '100110111011001111100110010000', 
# '001000011100111011000010111000', '110001101101001110111101100011', 
# '010000111110010000000101100001', '110001101101011100000010110101', 
# '111100110010100110101110110111', '001010110010011111110000101000', 
# '110001100111100000110111000100', '011100100100101010010011101000', 
# '110000101110001001110110111010', '111000000001110101111101101001', 
# '011111110101110001100001111110', '100111100011001000110010100101']
"""


def select_parents_roulette(population, fitness_scores, population_size):
    """
    Select parents for the next generation based on their fitness scores
    using the Fitness Proportionate Selection (Roulette Wheel Selection) method.

    Args:
        population (list of str): The current population of individuals, each represented as a binary string.
        fitness_scores (list of int): The fitness scores of the individuals in the population.
        population_size (int): number of chromosomes in the original population

    Returns:
        selected (list): The selected parents for the next generation.
    """

    total_fitness = np.sum(fitness_scores)

    if total_fitness == 0:
        probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
    else:
        probabilities = np.array(fitness_scores) / total_fitness

    selected_indices = np.random.choice(len(population), size=population_size, p=probabilities)
    selected = [population[i] for i in selected_indices]

    return selected


"""
p = initialize_population(10, 10)
f = [20, 12, 2, 3, 4, 5, 6, 7, 12, 70]
sp = select_parents_roulette(p, f, 10)
print(p)
print(sp)
# ['1100010110', '1000000111', '0011001100', '0001000111', '1000111000',
# '0110011111', '1000011010', '0001000110', '0011100101', '1111111001']

# ['1100010110', '1111111001', '1000011010', '1111111001', '1111111001',
# '1111111001', '1111111001', '1111111001', '1111111001', '1111111001']
"""



def compute_fitness(population, target):
    """
    Compute the fitness value for each individual in the population.

    Args:
        population (list of str): The population of individuals, each represented as a binary string.
        target (str): The target binary string to compare against.

    Returns:
        fitness_values (list of int): A list of fitness values, one for each individual in the population.
    """
    fitness_values = []

    for individual in population:
        fitness = sum(1 for i, j in zip(individual, target) if i == j)
        fitness_values.append(fitness)

    return fitness_values

"""
p = initialize_population(10, 10)
t = "0000111001"
fv = compute_fitness(p, t)
print(fv)
# [1, 0, 1, 0, 1, 0, 0, 0, 0, 1]
"""


def genetic_algorithm(population_size, mutation_rate, crossover_rate, num_crossover_points,
                      target, precision_bits, max_generation=1000, fitness_trigger=None):

    """
    Execute a genetic algorithm to optimize a population of binary-encoded chromosomes.

    Args:
        population_size (int): Number of individuals in the population.

        component of the chromosomes. (min_val, max_val, bits)
        max_generation (int): maximum Number of generations to run the algorithm.
        mutation_rate (float): Probability of mutating each bit in each chromosome.
        crossover_rate (float): Probability of performing crossover for each chromosome.
        num_crossover_points (int): number of crossover points in chromosome.
        target (numpy.ndarray): Target array to compare simulation results against.
        precision_bits (tuple): Tuple containing (min_val, max_val, bits) for encoding the target.
            - min_val (float): Minimum possible value of the target range.
            - max_val (float): Maximum possible value of the target range.
            - bits (int): Number of bits used to represent the target values in the binary string.

        fitness_trigger (int or float): fitness threshold to break the algorithm

    Returns:
        population (list): Final population of binary-encoded chromosomes.
    """

    elite_chromosomes = []  # list to store the best binary chromosome of each generation
    best_fitness = []  # list to store the best fitness of each generation

    print(f"{'-' * 40}")
    print("      *** Genetic Algorithm *** ")
    print(f"{'-' * 40}")

    binary_target = decimal_to_binary(array=target, precision_bits=precision_bits)
    population = initialize_population(
        pop_size=population_size,
        bit_length=len(binary_target)
    )


    if fitness_trigger:
        max_fitness = fitness_trigger
    else:
        max_fitness = len(binary_target)

    max_generation_fitness = 0
    generation = 1

    while generation <= max_generation and max_generation_fitness < max_fitness:

        generation_fitness = compute_fitness(
            population=population,
            target=binary_target
        )

        elite_chromosomes.append(extract_based_on_max_index(list1=population, list2=generation_fitness))
        max_generation_fitness = max(generation_fitness)
        best_fitness.append(max_generation_fitness)

        if max_generation_fitness == max_fitness:
            print()
            print("The Algorithm Found The Best Solution (max fitness == max generation fitness)")
            break

        new_population = []
        parents = select_parents_roulette(
            population=population,
            fitness_scores=generation_fitness,
            population_size=population_size
        )

        for _ in range(len(parents) // 2):

            while len(parents) >= 2:
                parent1 = random.choice(parents)
                parents.remove(parent1)

                if len(parents) > 0:
                    parent2 = random.choice(parents)
                    parents.remove(parent2)
                else:
                    parent2 = parent1
                break  # Exit the while loop after selecting parent1 and parent2

            offspring1, offspring2 = crossover(
                parent1=parent1,
                parent2=parent2,
                crossover_rate=crossover_rate,
                num_crossover_points=num_crossover_points
            )
            new_population.extend([
                mutate(
                    chromosome=offspring1,
                    mutation_rate=mutation_rate
                ),
                mutate(
                    chromosome=offspring2,
                    mutation_rate=mutation_rate
                )
            ])

        print(f"Generation {generation}; Best/Max Fitness: {max_generation_fitness}/{max_fitness}")
        population = new_population
        generation += 1

    average_fitness = sum(best_fitness) / len(best_fitness)

    print(f"{'------------------------------------------'}")
    print(f"      Simulation Complete!")
    print(f"      The best found fitness: {max(best_fitness)}")
    print(f"      Total Generations: {len(best_fitness)}")
    print(f"      Average Fitness: {average_fitness:.2f}")
    print(f"{'------------------------------------------'}")

    return population, elite_chromosomes, best_fitness


pop, elite_chr, bf = genetic_algorithm(
    population_size=300,
    mutation_rate=0.01,
    crossover_rate=0.85,
    num_crossover_points=2,
    target=np.array([1, 3, 1, 6, 9, 1]),
    precision_bits=(0, 10, 8),
    max_generation=10000,
)

import matplotlib.pyplot as plt
plt.plot(bf)
plt.show()
