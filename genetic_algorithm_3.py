import random

# source: https://www.youtube.com/watch?v=zumC_C0C25c&list=WL&index=1


"""
1. Population: The `Population` class represents a group of potential solutions (chromosomes) to a problem. 
   Each chromosome is a potential solution to the problem. The population size determines how many potential
   solutions are generated initially.

2. Chromosome: A `Chromosome` is an individual potential solution to the problem. In this implementation, 
   each chromosome consists of a sequence of genes. Each gene represents a part of the solution. 
   In the provided code, the genes are randomly initialized with binary values (0 or 1).

3. Fitness Function: The `get_fitness` method of the `Chromosome` class calculates the fitness of a chromosome. 
   The fitness function evaluates how well a chromosome solves the problem. In this case, the fitness is 
   determined by how many genes in the chromosome match the corresponding genes in the target chromosome 
   (`TARGET_CHROMOSOME`). The higher the fitness value, the better the chromosome.

4. Crossover: Crossover is a genetic operator used to combine genetic information from two parent chromosomes 
   to create new offspring chromosomes. In the `GeneticAlgorithm` class, the `crossover_population` method
   performs crossover between pairs of parent chromosomes. It selects parent chromosomes using tournament 
   selection (explained later), and then performs crossover by randomly selecting genes from each parent 
   to create new offspring chromosomes.

5. Mutation: Mutation is another genetic operator that introduces random changes in the genes of a chromosome. 
   It helps introduce diversity into the population and prevents premature convergence to suboptimal solutions. 
   In the `GeneticAlgorithm` class, the `mutate_population` method applies mutation to the offspring chromosomes.
   It randomly flips the value of each gene with a probability defined by the mutation rate (`MUTATION_RATE`).

6. Tournament Selection: Tournament selection is a method used to select parent chromosomes for crossover. 
   In tournament selection, a small number of chromosomes are randomly selected from the population, 
   and the one with the highest fitness is chosen as a parent. This process is repeated to select multiple parents. 
   The `select_tournament_population` method in the `GeneticAlgorithm` class implements tournament selection.

7. Evolution: The `evolve` method in the `GeneticAlgorithm` class is responsible for evolving the population to 
   the next generation. It first performs crossover to create new offspring chromosomes, then applies mutation 
   to the offspring chromosomes, and finally returns the updated population for the next generation.
"""

POPULATION_SIZE = 8
# number of chromosomes which in each generation are not affected by mutation or crossover
NUMBER_OF_ELITE_CHROMOSOMES = 1
TOURNAMENT_SELECTION_SIZE = 4
MUTATION_RATE = 0.25
TARGET_CHROMOSOME = [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]


class Chromosome:
    def __init__(self):
        self.genes = []
        self.fitness = 0
        i = 0
        while i < TARGET_CHROMOSOME.__len__():
            if random.random() >= 0.5:
                self.genes.append(1)
            else:
                self.genes.append(0)
            i += 1

    def get_genes(self):
        return self.genes

    def get_fitness(self):

        self.fitness = 0

        for i in range(self.genes.__len__()):

            if self.genes[i] == TARGET_CHROMOSOME[i]:

                self.fitness += 1

        return self.fitness

    def __str__(self):
        return self.genes.__str__()


class Population:
    def __init__(self, size):
        self.chromosomes = []

        i = 0
        while i < size:
            self.chromosomes.append(Chromosome())
            i += 1

    def get_chromosomes(self):

        return self.chromosomes


class GeneticAlgorithm:
    @staticmethod
    def evolve(pop):
        return GeneticAlgorithm.mutate_population(GeneticAlgorithm.crossover_population(pop))

    @staticmethod
    def crossover_population(pop):
        crossover_pop = Population(0)
        for i in range(NUMBER_OF_ELITE_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i])
        i = NUMBER_OF_ELITE_CHROMOSOMES
        while i < POPULATION_SIZE:
            chromosome1 = GeneticAlgorithm.select_tournament_population(pop).get_chromosomes()[0]
            chromosome2 = GeneticAlgorithm.select_tournament_population(pop).get_chromosomes()[0]
            crossover_pop.get_chromosomes().append(GeneticAlgorithm.crossover_chromosomes(chromosome1, chromosome2))
            i += 1




        return crossover_pop

    @staticmethod
    def mutate_population(pop):
        for i in range(NUMBER_OF_ELITE_CHROMOSOMES, POPULATION_SIZE):
            GeneticAlgorithm.mutate_chromosome(pop.get_chromosomes()[i])


        return pop

    @staticmethod
    def crossover_chromosomes(chromosome1, chromosome2):
        crossover_chrom = Chromosome()
        for i in range(TARGET_CHROMOSOME.__len__()):
            if random.random() >= 0.5:
                crossover_chrom.get_genes()[i] = chromosome1.get_genes()[i]
            else:
                crossover_chrom.get_genes()[i] = chromosome2.get_genes()[i]

        return crossover_chrom

    @staticmethod
    def mutate_chromosome(chromosome):
        for i in range(TARGET_CHROMOSOME.__len__()):
            if random.random() < MUTATION_RATE:
                if random.random() < 0.5:
                    chromosome.get_genes()[i] = 1
                else:
                    chromosome.get_genes()[i] = 0

    @staticmethod
    def select_tournament_population(pop):

        tournament_pop = Population(0)

        i = 0
        while i < TOURNAMENT_SELECTION_SIZE:

            tournament_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0, POPULATION_SIZE)])
            i += 1

        tournament_pop.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)

        return tournament_pop



def print_population(pop, gen_number):
    print("\n________________________________________________")
    print(f"Generation # {gen_number} | Fittest chromosome fitness: {pop.get_chromosomes()[0].get_fitness()}")
    print(f"Target Chromosome: {TARGET_CHROMOSOME}")
    print("________________________________________________")
    i = 0
    for x in pop.get_chromosomes():
        print(f"Chromosome # {i}: {x} | Fitness: {x.get_fitness()}")
        i += 1


population = Population(POPULATION_SIZE)
population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)

print_population(population, 0)
generation_number = 1

while population.get_chromosomes()[0].get_fitness() < TARGET_CHROMOSOME.__len__():
    population = GeneticAlgorithm.evolve(population)
    population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
    print_population(population, generation_number)
    generation_number += 1



"""
________________________________________________
Generation # 0 | Fittest chromosome fitness: 6
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 1, 1, 1, 0, 1, 0, 0, 1] | Fitness: 6
Chromosome # 1: [1, 0, 1, 0, 0, 1, 1, 1, 1, 1] | Fitness: 6
Chromosome # 2: [1, 1, 1, 0, 1, 0, 1, 1, 1, 0] | Fitness: 6
Chromosome # 3: [1, 0, 1, 1, 0, 0, 1, 0, 1, 1] | Fitness: 5
Chromosome # 4: [0, 0, 1, 0, 1, 1, 0, 0, 0, 1] | Fitness: 5
Chromosome # 5: [1, 1, 1, 0, 0, 0, 0, 0, 0, 1] | Fitness: 5
Chromosome # 6: [0, 0, 1, 1, 0, 0, 1, 0, 0, 1] | Fitness: 3
Chromosome # 7: [0, 0, 0, 1, 1, 0, 1, 1, 0, 0] | Fitness: 3

________________________________________________
Generation # 1 | Fittest chromosome fitness: 8
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 1, 1, 1, 1, 1, 1, 1] | Fitness: 8
Chromosome # 1: [1, 1, 1, 0, 1, 0, 1, 1, 1, 1] | Fitness: 7
Chromosome # 2: [1, 1, 1, 1, 1, 0, 1, 0, 1, 1] | Fitness: 7
Chromosome # 3: [1, 1, 1, 1, 1, 0, 1, 0, 0, 1] | Fitness: 6
Chromosome # 4: [1, 0, 0, 0, 0, 0, 1, 1, 1, 1] | Fitness: 6
Chromosome # 5: [1, 1, 1, 1, 1, 0, 1, 0, 0, 1] | Fitness: 6
Chromosome # 6: [1, 1, 0, 0, 1, 0, 0, 1, 0, 1] | Fitness: 6
Chromosome # 7: [0, 1, 1, 0, 1, 1, 0, 1, 0, 1] | Fitness: 5

________________________________________________
Generation # 2 | Fittest chromosome fitness: 8
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 1, 1, 1, 1, 1, 1, 1] | Fitness: 8
Chromosome # 1: [1, 1, 1, 0, 1, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 2: [1, 1, 0, 1, 1, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 3: [1, 1, 1, 0, 1, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 4: [1, 1, 1, 1, 1, 1, 1, 0, 0, 1] | Fitness: 7
Chromosome # 5: [0, 1, 1, 0, 1, 1, 1, 1, 0, 0] | Fitness: 5
Chromosome # 6: [1, 1, 1, 1, 0, 0, 1, 1, 1, 1] | Fitness: 5
Chromosome # 7: [1, 1, 0, 1, 0, 0, 1, 1, 0, 1] | Fitness: 5

________________________________________________
Generation # 3 | Fittest chromosome fitness: 8
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 1, 1, 1, 1, 1, 1, 1] | Fitness: 8
Chromosome # 1: [1, 1, 1, 0, 1, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 2: [1, 1, 0, 1, 1, 1, 1, 1, 1, 0] | Fitness: 7
Chromosome # 3: [1, 1, 1, 0, 1, 0, 0, 0, 1, 1] | Fitness: 7
Chromosome # 4: [1, 1, 1, 0, 1, 0, 1, 0, 1, 0] | Fitness: 7
Chromosome # 5: [1, 1, 0, 1, 0, 0, 1, 0, 1, 1] | Fitness: 7
Chromosome # 6: [1, 1, 0, 1, 1, 0, 1, 1, 1, 1] | Fitness: 7
Chromosome # 7: [1, 1, 1, 0, 1, 0, 1, 0, 1, 0] | Fitness: 7

________________________________________________
Generation # 4 | Fittest chromosome fitness: 9
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 1: [1, 1, 0, 1, 1, 1, 1, 1, 1, 1] | Fitness: 8
Chromosome # 2: [1, 1, 0, 0, 1, 0, 1, 0, 1, 0] | Fitness: 8
Chromosome # 3: [1, 1, 1, 0, 1, 0, 1, 0, 1, 0] | Fitness: 7
Chromosome # 4: [1, 1, 0, 1, 0, 1, 1, 1, 1, 1] | Fitness: 7
Chromosome # 5: [1, 1, 1, 1, 1, 1, 0, 1, 1, 1] | Fitness: 6
Chromosome # 6: [0, 1, 1, 0, 1, 0, 1, 1, 1, 1] | Fitness: 6
Chromosome # 7: [0, 1, 1, 0, 1, 1, 0, 1, 1, 0] | Fitness: 5

________________________________________________
Generation # 5 | Fittest chromosome fitness: 9
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 1: [1, 1, 0, 1, 1, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 2: [1, 1, 0, 0, 1, 0, 1, 0, 1, 0] | Fitness: 8
Chromosome # 3: [1, 1, 0, 0, 1, 0, 1, 0, 1, 0] | Fitness: 8
Chromosome # 4: [1, 1, 1, 0, 1, 0, 1, 0, 1, 0] | Fitness: 7
Chromosome # 5: [0, 1, 1, 0, 1, 0, 1, 0, 1, 1] | Fitness: 7
Chromosome # 6: [1, 0, 0, 1, 1, 1, 1, 1, 1, 0] | Fitness: 6
Chromosome # 7: [1, 1, 1, 1, 0, 1, 0, 0, 1, 1] | Fitness: 6

________________________________________________
Generation # 6 | Fittest chromosome fitness: 9
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 1: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 2: [1, 0, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 3: [1, 1, 0, 0, 1, 0, 1, 1, 1, 1] | Fitness: 8
Chromosome # 4: [1, 1, 0, 0, 1, 0, 1, 1, 1, 0] | Fitness: 7
Chromosome # 5: [1, 1, 1, 0, 1, 0, 1, 0, 1, 0] | Fitness: 7
Chromosome # 6: [1, 1, 0, 0, 0, 0, 1, 1, 1, 1] | Fitness: 7
Chromosome # 7: [1, 0, 0, 1, 1, 0, 1, 1, 1, 0] | Fitness: 5

________________________________________________
Generation # 7 | Fittest chromosome fitness: 9
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 1: [1, 0, 0, 0, 1, 1, 1, 0, 1, 1] | Fitness: 9
Chromosome # 2: [1, 1, 0, 0, 0, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 3: [1, 1, 0, 0, 1, 0, 0, 0, 1, 1] | Fitness: 8
Chromosome # 4: [1, 1, 0, 1, 1, 0, 1, 1, 1, 1] | Fitness: 7
Chromosome # 5: [0, 1, 1, 0, 1, 0, 1, 0, 1, 1] | Fitness: 7
Chromosome # 6: [1, 1, 0, 1, 0, 0, 1, 0, 1, 1] | Fitness: 7
Chromosome # 7: [1, 1, 0, 0, 0, 0, 1, 1, 1, 0] | Fitness: 6

________________________________________________
Generation # 8 | Fittest chromosome fitness: 9
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 1: [1, 0, 0, 0, 1, 1, 1, 0, 1, 1] | Fitness: 9
Chromosome # 2: [1, 1, 0, 0, 1, 0, 1, 1, 1, 1] | Fitness: 8
Chromosome # 3: [1, 0, 0, 1, 1, 1, 1, 0, 1, 1] | Fitness: 8
Chromosome # 4: [1, 0, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 5: [1, 1, 0, 0, 0, 1, 1, 0, 1, 0] | Fitness: 8
Chromosome # 6: [1, 1, 0, 0, 0, 0, 0, 0, 1, 1] | Fitness: 7
Chromosome # 7: [1, 0, 0, 0, 0, 0, 0, 0, 1, 1] | Fitness: 6

________________________________________________
Generation # 9 | Fittest chromosome fitness: 9
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 1: [1, 1, 0, 0, 1, 1, 0, 1, 1, 1] | Fitness: 8
Chromosome # 2: [1, 1, 0, 0, 1, 0, 1, 0, 1, 0] | Fitness: 8
Chromosome # 3: [0, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 4: [1, 1, 0, 0, 0, 1, 1, 0, 0, 1] | Fitness: 8
Chromosome # 5: [1, 0, 0, 0, 0, 1, 1, 0, 1, 1] | Fitness: 8
Chromosome # 6: [1, 1, 0, 0, 0, 0, 1, 0, 0, 1] | Fitness: 7
Chromosome # 7: [1, 1, 0, 1, 1, 1, 0, 0, 1, 0] | Fitness: 7

________________________________________________
Generation # 10 | Fittest chromosome fitness: 9
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 1: [1, 1, 0, 0, 0, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 2: [1, 1, 0, 0, 1, 0, 0, 0, 1, 1] | Fitness: 8
Chromosome # 3: [0, 1, 0, 0, 1, 0, 1, 0, 1, 0] | Fitness: 7
Chromosome # 4: [1, 1, 0, 0, 1, 0, 1, 1, 0, 1] | Fitness: 7
Chromosome # 5: [1, 0, 0, 0, 0, 0, 1, 0, 0, 1] | Fitness: 6
Chromosome # 6: [0, 0, 1, 0, 1, 0, 1, 0, 1, 0] | Fitness: 5
Chromosome # 7: [1, 1, 1, 0, 0, 0, 1, 1, 1, 0] | Fitness: 5

________________________________________________
Generation # 11 | Fittest chromosome fitness: 9
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 1: [1, 1, 0, 0, 0, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 2: [1, 1, 0, 0, 1, 0, 1, 0, 0, 1] | Fitness: 8
Chromosome # 3: [1, 1, 0, 0, 1, 0, 0, 0, 1, 1] | Fitness: 8
Chromosome # 4: [1, 1, 0, 1, 1, 0, 1, 0, 0, 1] | Fitness: 7
Chromosome # 5: [1, 1, 0, 1, 0, 0, 1, 0, 1, 1] | Fitness: 7
Chromosome # 6: [1, 1, 0, 0, 0, 0, 1, 1, 1, 1] | Fitness: 7
Chromosome # 7: [1, 0, 0, 0, 0, 0, 1, 1, 1, 1] | Fitness: 6

________________________________________________
Generation # 12 | Fittest chromosome fitness: 9
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 1: [1, 1, 0, 0, 1, 0, 0, 0, 1, 1] | Fitness: 8
Chromosome # 2: [1, 1, 0, 0, 1, 0, 0, 0, 1, 1] | Fitness: 8
Chromosome # 3: [1, 1, 0, 0, 1, 0, 0, 0, 1, 0] | Fitness: 7
Chromosome # 4: [0, 1, 0, 0, 1, 0, 0, 0, 1, 1] | Fitness: 7
Chromosome # 5: [1, 1, 0, 1, 1, 0, 0, 0, 1, 1] | Fitness: 7
Chromosome # 6: [1, 1, 0, 1, 0, 0, 1, 0, 1, 1] | Fitness: 7
Chromosome # 7: [1, 1, 0, 0, 1, 0, 1, 1, 0, 1] | Fitness: 7

________________________________________________
Generation # 13 | Fittest chromosome fitness: 9
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 1: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 2: [1, 1, 0, 1, 1, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 3: [1, 1, 0, 0, 1, 0, 0, 0, 1, 1] | Fitness: 8
Chromosome # 4: [1, 1, 0, 1, 1, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 5: [1, 1, 0, 0, 1, 0, 0, 1, 1, 1] | Fitness: 7
Chromosome # 6: [1, 1, 0, 0, 0, 0, 0, 0, 1, 0] | Fitness: 6
Chromosome # 7: [0, 1, 1, 0, 0, 0, 1, 0, 1, 1] | Fitness: 6

________________________________________________
Generation # 14 | Fittest chromosome fitness: 9
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 1: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 2: [1, 1, 0, 0, 1, 0, 0, 0, 1, 1] | Fitness: 8
Chromosome # 3: [1, 1, 0, 0, 1, 0, 1, 0, 1, 0] | Fitness: 8
Chromosome # 4: [1, 1, 1, 0, 1, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 5: [1, 1, 0, 1, 1, 0, 1, 0, 1, 0] | Fitness: 7
Chromosome # 6: [1, 1, 0, 0, 1, 0, 0, 0, 1, 0] | Fitness: 7
Chromosome # 7: [1, 1, 0, 1, 0, 0, 1, 0, 1, 1] | Fitness: 7

________________________________________________
Generation # 15 | Fittest chromosome fitness: 9
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 1: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 2: [1, 1, 1, 0, 1, 1, 1, 0, 1, 1] | Fitness: 9
Chromosome # 3: [1, 1, 0, 0, 1, 0, 0, 0, 1, 1] | Fitness: 8
Chromosome # 4: [1, 1, 0, 0, 0, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 5: [0, 1, 1, 0, 1, 0, 1, 0, 1, 1] | Fitness: 7
Chromosome # 6: [1, 0, 0, 0, 0, 0, 1, 0, 1, 1] | Fitness: 7
Chromosome # 7: [1, 1, 0, 0, 0, 0, 0, 0, 1, 1] | Fitness: 7

________________________________________________
Generation # 16 | Fittest chromosome fitness: 10
Target Chromosome: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
________________________________________________
Chromosome # 0: [1, 1, 0, 0, 1, 1, 1, 0, 1, 1] | Fitness: 10
Chromosome # 1: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 2: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] | Fitness: 9
Chromosome # 3: [1, 1, 0, 0, 1, 1, 1, 1, 1, 1] | Fitness: 9
Chromosome # 4: [0, 1, 0, 0, 1, 1, 1, 0, 1, 1] | Fitness: 9
Chromosome # 5: [1, 1, 0, 0, 1, 1, 1, 0, 1, 0] | Fitness: 9
Chromosome # 6: [1, 1, 1, 0, 1, 0, 1, 0, 1, 1] | Fitness: 8
Chromosome # 7: [1, 0, 0, 0, 0, 0, 1, 0, 1, 1] | Fitness: 7

"""






