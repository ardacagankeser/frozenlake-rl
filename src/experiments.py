import json
import matplotlib.pyplot as plt
import pandas as pd
import os
from src.environment import Environment
from src.genetic_algorithm import GeneticAlgorithm

class Experiments:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.base_config = json.load(f)
        
        if not os.path.exists('results'):
            os.makedirs('results')

    def run_experiment_1(self):
        print("Running Experiment 1: Operator Analysis (One-point vs Uniform Crossover)")
        
        # Test One-point
        config_op = self.base_config.copy()
        config_op['crossover_method'] = 'one_point'
        env = Environment(size=config_op['env_size'])
        ga_op = GeneticAlgorithm(config_op, env)
        best_op, history_op = ga_op.run_evolution()
        
        # Test Uniform
        config_un = self.base_config.copy()
        config_un['crossover_method'] = 'uniform'
        ga_un = GeneticAlgorithm(config_un, env)
        best_un, history_un = ga_un.run_evolution()
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(history_op['best_fitness'], label='One-point Crossover')
        plt.plot(history_un['best_fitness'], label='Uniform Crossover')
        plt.title('Experiment 1: Best Fitness vs Generation')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/experiment1_crossover.png')
        plt.close()
        
        # Generate Data Table
        data = {
            'Method': ['One-point', 'Uniform'],
            'Best Fitness': [best_op.fitness, best_un.fitness],
            'Success': [best_op.success, best_un.success],
            'Steps': [best_op.steps, best_un.steps]
        }
        df = pd.DataFrame(data)
        print("\nExperiment 1 Results:")
        print(df.to_string(index=False))
        df.to_csv('results/experiment1_results.csv', index=False)
        return df

    def run_experiment_2(self):
        print("\nRunning Experiment 2: Strategy Analysis ((mu + lambda) vs (mu, lambda))")
        
        # Test (mu + lambda)
        config_plus = self.base_config.copy()
        config_plus['strategy'] = 'mu_plus_lambda'
        env = Environment(size=config_plus['env_size'])
        ga_plus = GeneticAlgorithm(config_plus, env)
        best_plus, history_plus = ga_plus.run_evolution()
        
        # Test (mu, lambda)
        config_comma = self.base_config.copy()
        config_comma['strategy'] = 'mu_comma_lambda'
        ga_comma = GeneticAlgorithm(config_comma, env)
        best_comma, history_comma = ga_comma.run_evolution()
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(history_plus['best_fitness'], label='(μ + λ) Elitist')
        plt.plot(history_comma['best_fitness'], label='(μ, λ) Non-elitist')
        plt.title('Experiment 2: Convergence Strategy Comparison')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/experiment2_strategy.png')
        plt.close()
        
        # Generate Data Table
        data = {
            'Strategy': ['(mu + lambda)', '(mu, lambda)'],
            'Best Fitness': [best_plus.fitness, best_comma.fitness],
            'Success': [best_plus.success, best_comma.success],
            'Steps': [best_plus.steps, best_comma.steps]
        }
        df = pd.DataFrame(data)
        print("\nExperiment 2 Results:")
        print(df.to_string(index=False))
        df.to_csv('results/experiment2_results.csv', index=False)
        return df

    def run_experiment_3(self):
        print("\nRunning Experiment 3: Scalability (4x4 vs 8x8)")
        
        # Test 4x4
        config_4 = self.base_config.copy()
        config_4['env_size'] = 4
        env_4 = Environment(size=config_4['env_size'])
        ga_4 = GeneticAlgorithm(config_4, env_4)
        best_4, history_4 = ga_4.run_evolution()
        
        # Test 8x8
        config_8 = self.base_config.copy()
        config_8['env_size'] = 8
        env_8 = Environment(size=config_8['env_size'])
        ga_8 = GeneticAlgorithm(config_8, env_8)
        best_8, history_8 = ga_8.run_evolution()
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(history_4['best_fitness'], label='4x4 Map')
        plt.plot(history_8['best_fitness'], label='8x8 Map')
        plt.title('Experiment 3: Scalability (4x4 vs 8x8)')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/experiment3_scalability.png')
        plt.close()
        
        # Generate Data Table
        data = {
            'Map Size': ['4x4', '8x8'],
            'Best Fitness': [best_4.fitness, best_8.fitness],
            'Success': [best_4.success, best_8.success],
            'Steps': [best_4.steps, best_8.steps]
        }
        df = pd.DataFrame(data)
        print("\nExperiment 3 Results:")
        print(df.to_string(index=False))
        df.to_csv('results/experiment3_results.csv', index=False)
        return df

    def run_all(self):
        self.run_experiment_1()
        self.run_experiment_2()
        self.run_experiment_3()
