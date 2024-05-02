import random

"""
In this genetic algorithm, the improvement over generations depends on the selection, 
crossover, and mutation operations. Initially, the population consists of random genomes. 
As the algorithm progresses through generations, the population evolves through these operations, 
ideally leading to genomes with better fitness values.

The selection process ensures that genomes with higher fitness values are more likely to be chosen as 
parents for producing offspring. The crossover operation exchanges genetic information between parents 
to generate potentially better solutions. Mutation introduces random changes to the genomes, 
allowing exploration of new regions of the search space.

The algorithm aims to find the best solution by iteratively improving the population over generations. 
However, there is an element of randomness involved due to the stochastic nature of selection, 
crossover, and mutation. Thus, while the algorithm strives to improve over generations, 
finding the best genome is not guaranteed and may rely on chance to some extent.

The termination condition based on reaching a maximum fitness value provides a stopping criterion 
for the algorithm. If a solution with a fitness value equal to or exceeding the maximum fitness 
value is found, the algorithm terminates early, assuming that the desired solution has been achieved.
"""


class GeneticAlgorithm:
    def __init__(self, population_size=100, genome_length=50, crossover_rate=0.5,  mutation_rate=0.01, max_generation=100, population=None, max_fitness_value=50):

        self.population_size = population_size
        self.genome_length = genome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generation = max_generation
        self.population = population
        self.max_fitness_value = max_fitness_value

    def genome_generator(self, genome_length):

        genome = [random.randint(a=0, b=1) for _ in range(genome_length)]
        return genome

    def init_population(self, population_size, genome_length):

        populations = [self.genome_generator(genome_length=genome_length) for _ in range(population_size)]

        return populations

    def fitness(self, genome):

        qualities = sum(genome)

        return qualities

    def select_parent(self, population, fitness_values):

        total_fitness = sum(fitness_values)
        pick = random.uniform(0, total_fitness)
        current = 0

        for individual, fitness_value in zip(population, fitness_values):

            current += fitness_value
            if current > pick:

                return individual

    def crossover(self, parent1, parent2, crossover_rate):

        if random.random() < crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)

            new_parent1 = parent1[:crossover_point] + parent2[crossover_point:]
            new_parent2 = parent2[:crossover_point] + parent1[crossover_point:]

            return new_parent1, new_parent2

        else:
            return parent1, parent2

    def mutation(self, genome, mutation_rate):

        for i in range(len(genome)):

            if random.random() < mutation_rate:

                genome[i] = abs(genome[i] - 1)

        return genome

    def run(self, pop=None):

        if pop:
            population = pop
        else:
            population = self.init_population(
                population_size=self.population_size,
                genome_length=self.genome_length
            )

        for generation in range(self.max_generation):
            fitness_values = [self.fitness(genome=genome) for genome in population]

            new_population = []
            for _ in range(self.population_size // 2):

                parent1 = self.select_parent(
                    population=population,
                    fitness_values=fitness_values
                )
                parent2 = self.select_parent(
                    population=population,
                    fitness_values=fitness_values
                )

                offspring1, offspring2 = self.crossover(
                    parent1=parent1,
                    parent2=parent2,
                    crossover_rate=self.crossover_rate
                )
                new_population.extend([self.mutation(
                    genome=offspring1,
                    mutation_rate=self.mutation_rate
                ), self.mutation(
                    genome=offspring2,
                    mutation_rate=self.mutation_rate
                )])

            population = new_population

            fitness_values = [self.fitness(genome=genome) for genome in population]
            best_fitness = max(fitness_values)
            print(f"Generation: {generation}; Best Fitness: {best_fitness}")

            if best_fitness >= self.max_fitness_value:
                break

        best_index = fitness_values.index(max(fitness_values))
        best_solution = population[best_index]
        print(f"Best Solution: {best_solution}")
        print(f"Best Fitness: {self.fitness(genome=best_solution)}")



model = GeneticAlgorithm(

    population_size=2000,
    genome_length=1000,
    crossover_rate=0.4,
    mutation_rate=0.01,
    max_generation=200,
    population=None,
    max_fitness_value=800
)

model.run()

"""
Generation: 0; Best Fitness: 553
Generation: 1; Best Fitness: 562
Generation: 2; Best Fitness: 564
Generation: 3; Best Fitness: 566
Generation: 4; Best Fitness: 556
Generation: 5; Best Fitness: 557
Generation: 6; Best Fitness: 556
Generation: 7; Best Fitness: 552

    ...
    
Generation: 197; Best Fitness: 569
Generation: 198; Best Fitness: 574
Generation: 199; Best Fitness: 568
Best Solution: [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
                1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                
                    ... 
                    
                1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
                
Best Fitness: 568
"""