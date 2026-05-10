import numpy as np
from src.individual import Individual

# Selects one parent via tournament selection by sampling k individuals 
# and returning the one with highest fitness.
def tournament_selection(population, tournament_size):
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament = [population[i] for i in selected_indices]
    best_ind = max(tournament, key=lambda ind: ind.fitness)
    return best_ind

# Performs one-point crossover by swapping the genotype tails 
# of two parents at a random cut point.
def one_point_crossover(parent1, parent2):
    length = len(parent1.genotype)
    point = np.random.randint(1, length)
    
    child1_genotype = np.concatenate((parent1.genotype[:point], parent2.genotype[point:]))
    child2_genotype = np.concatenate((parent2.genotype[:point], parent1.genotype[point:]))
    
    return Individual(genotype=child1_genotype), Individual(genotype=child2_genotype)

# Performs two-point crossover by exchanging 
# a randomly selected middle segment between two parents.
def two_point_crossover(parent1, parent2):
    length = len(parent1.genotype)
    if length < 3:
        return one_point_crossover(parent1, parent2)
    
    point1 = np.random.randint(1, length - 1)
    point2 = np.random.randint(point1 + 1, length)
    
    child1_genotype = np.concatenate((parent1.genotype[:point1], parent2.genotype[point1:point2], parent1.genotype[point2:]))
    child2_genotype = np.concatenate((parent2.genotype[:point1], parent1.genotype[point1:point2], parent2.genotype[point2:]))
    
    return Individual(genotype=child1_genotype), Individual(genotype=child2_genotype)

# Performs uniform crossover by independently choosing each gene 
# from either parent using a random boolean mask.
def uniform_crossover(parent1, parent2):
    length = len(parent1.genotype)
    mask = np.random.rand(length) > 0.5
    
    child1_genotype = np.where(mask, parent1.genotype, parent2.genotype)
    child2_genotype = np.where(mask, parent2.genotype, parent1.genotype)
    
    return Individual(genotype=child1_genotype), Individual(genotype=child2_genotype)


# Mutates an individual by randomly resetting each gene 
# to a new action with probability mutation_rate.
def random_resetting_mutation(individual, mutation_rate):
    length = len(individual.genotype)
    mask = np.random.rand(length) < mutation_rate
    mutations = np.random.randint(0, 4, size=length)
    individual.genotype = np.where(mask, mutations, individual.genotype)

# Applies swap mutation by exchanging a gene with 
# another random position with probability mutation_rate per gene.
def swap_mutation(individual, mutation_rate):
    length = len(individual.genotype)
    for i in range(length):
        if np.random.rand() < mutation_rate:
            j = np.random.randint(0, length)
            individual.genotype[i], individual.genotype[j] = individual.genotype[j], individual.genotype[i]

# With probability mutation_rate, removes one gene and 
# reinserts it at another position to create an insertion permutation.
def insert_mutation(individual, mutation_rate):
    length = len(individual.genotype)
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(length, 2, replace=False)
        gene = individual.genotype[i]
        genotype_list = individual.genotype.tolist()
        genotype_list.pop(i)
        insert_pos = j if i > j else j - 1
        genotype_list.insert(insert_pos, gene)
        individual.genotype = np.array(genotype_list)

# With probability mutation_rate, randomly shuffles the genes 
# within a selected contiguous segment (scramble mutation).
def scramble_mutation(individual, mutation_rate):
    length = len(individual.genotype)
    if np.random.rand() < mutation_rate:
        i, j = sorted(np.random.choice(length, 2, replace=False))
        if j > i:
            sub = individual.genotype[i:j+1]
            np.random.shuffle(sub)
            individual.genotype[i:j+1] = sub

# With probability mutation_rate, reverses the order of genes 
# within a randomly chosen segment (inversion mutation).
def inversion_mutation(individual, mutation_rate):
    length = len(individual.genotype)
    if np.random.rand() < mutation_rate:
        i, j = sorted(np.random.choice(length, 2, replace=False))
        if j > i:
            individual.genotype[i:j+1] = individual.genotype[i:j+1][::-1]
