import numpy as np
from src.individual import Individual
from src.ga_operators import tournament_selection, one_point_crossover, uniform_crossover, random_resetting_mutation, swap_mutation

class GeneticAlgorithm:
    def __init__(self, config, environment):
        self.config = config
        self.env = environment
        self.pop_size = config['pop_size']
        self.max_generations = config['max_generations']
        self.mutation_rate = config['mutation_rate']
        self.crossover_rate = config['crossover_rate']
        self.tournament_size = config['tournament_size']
        self.crossover_method = config['crossover_method']
        self.mutation_method = config['mutation_method']
        self.strategy = config['strategy']
        self.weights = config['fitness_weights']

        self.population = []
        self.history = {'best_fitness': [], 'avg_fitness': [], 'success_rate': []}

    def initialize_population(self):
        genotype_length = self.env.get_genotype_length()
        self.population = [Individual(genotype_length=genotype_length) for _ in range(self.pop_size)]

    def evaluate_population(self, population):
        for ind in population:
            ind.evaluate(self.env, self.weights)

    def select_parent(self):
        return tournament_selection(self.population, self.tournament_size)

    def apply_crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            if self.crossover_method == "one_point":
                return one_point_crossover(parent1, parent2)
            elif self.crossover_method == "uniform":
                return uniform_crossover(parent1, parent2)
        return parent1.clone(), parent2.clone()

    def apply_mutation(self, individual):
        if self.mutation_method == "random_resetting":
            random_resetting_mutation(individual, self.mutation_rate)
        elif self.mutation_method == "swap":
            swap_mutation(individual, self.mutation_rate)

    def generate_offspring(self):
        offspring = []
        while len(offspring) < self.pop_size:
            p1 = self.select_parent()
            p2 = self.select_parent()
            
            c1, c2 = self.apply_crossover(p1, p2)
            
            self.apply_mutation(c1)
            self.apply_mutation(c2)
            
            offspring.append(c1)
            if len(offspring) < self.pop_size:
                offspring.append(c2)
        return offspring

    def log_stats(self):
        best_fitness = max(ind.fitness for ind in self.population)
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        success_rate = sum(ind.success for ind in self.population) / self.pop_size
        
        self.history['best_fitness'].append(best_fitness)
        self.history['avg_fitness'].append(avg_fitness)
        self.history['success_rate'].append(success_rate)
        
        return best_fitness, avg_fitness, success_rate

    def run_evolution(self):
        self.initialize_population()
        self.evaluate_population(self.population)
        
        best_overall = max(self.population, key=lambda ind: ind.fitness)

        for gen in range(self.max_generations):
            offspring = self.generate_offspring()
            self.evaluate_population(offspring)
            
            if self.strategy == "mu_plus_lambda":
                combined = self.population + offspring
                combined.sort(key=lambda ind: ind.fitness, reverse=True)
                self.population = combined[:self.pop_size]
            elif self.strategy == "mu_comma_lambda":
                offspring.sort(key=lambda ind: ind.fitness, reverse=True)
                self.population = offspring[:self.pop_size]

            best_fitness, avg_fitness, success_rate = self.log_stats()
            
            gen_best = max(self.population, key=lambda ind: ind.fitness)
            if gen_best.fitness > best_overall.fitness:
                best_overall = gen_best.clone()

            # Optional: Add early stopping if success=1 and we want to stop
            # if best_overall.success == 1:
            #     pass 

        return best_overall, self.history
