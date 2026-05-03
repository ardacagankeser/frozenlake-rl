import numpy as np
from src.individual import Individual

def tournament_selection(population, tournament_size):
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament = [population[i] for i in selected_indices]
    best_ind = max(tournament, key=lambda ind: ind.fitness)
    return best_ind

def one_point_crossover(parent1, parent2):
    length = len(parent1.genotype)
    point = np.random.randint(1, length)
    
    child1_genotype = np.concatenate((parent1.genotype[:point], parent2.genotype[point:]))
    child2_genotype = np.concatenate((parent2.genotype[:point], parent1.genotype[point:]))
    
    return Individual(genotype=child1_genotype), Individual(genotype=child2_genotype)

def two_point_crossover(parent1, parent2):
    length = len(parent1.genotype)
    if length < 3:
        return one_point_crossover(parent1, parent2)
    
    point1 = np.random.randint(1, length - 1)
    point2 = np.random.randint(point1 + 1, length)
    
    child1_genotype = np.concatenate((parent1.genotype[:point1], parent2.genotype[point1:point2], parent1.genotype[point2:]))
    child2_genotype = np.concatenate((parent2.genotype[:point1], parent1.genotype[point1:point2], parent2.genotype[point2:]))
    
    return Individual(genotype=child1_genotype), Individual(genotype=child2_genotype)

def uniform_crossover(parent1, parent2):
    length = len(parent1.genotype)
    mask = np.random.rand(length) > 0.5
    
    child1_genotype = np.where(mask, parent1.genotype, parent2.genotype)
    child2_genotype = np.where(mask, parent2.genotype, parent1.genotype)
    
    return Individual(genotype=child1_genotype), Individual(genotype=child2_genotype)

def random_resetting_mutation(individual, mutation_rate):
    length = len(individual.genotype)
    mask = np.random.rand(length) < mutation_rate
    mutations = np.random.randint(0, 4, size=length)
    individual.genotype = np.where(mask, mutations, individual.genotype)

def swap_mutation(individual, mutation_rate):
    length = len(individual.genotype)
    for i in range(length):
        if np.random.rand() < mutation_rate:
            j = np.random.randint(0, length)
            individual.genotype[i], individual.genotype[j] = individual.genotype[j], individual.genotype[i]

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

def scramble_mutation(individual, mutation_rate):
    length = len(individual.genotype)
    if np.random.rand() < mutation_rate:
        i, j = sorted(np.random.choice(length, 2, replace=False))
        if j > i:
            sub = individual.genotype[i:j+1]
            np.random.shuffle(sub)
            individual.genotype[i:j+1] = sub

def inversion_mutation(individual, mutation_rate):
    length = len(individual.genotype)
    if np.random.rand() < mutation_rate:
        i, j = sorted(np.random.choice(length, 2, replace=False))
        if j > i:
            individual.genotype[i:j+1] = individual.genotype[i:j+1][::-1]
