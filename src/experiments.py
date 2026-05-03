import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import imageio
import seaborn as sns
from scipy import stats
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from src.environment import Environment
from src.genetic_algorithm import GeneticAlgorithm

class Experiments:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.base_config = json.load(f)
        
        if not os.path.exists('results'):
            os.makedirs('results')
            
        # Global harita: Deney 1 ve 2'nin adil kiyaslanmasi icin tek bir global harita uretiyoruz
        self.global_desc = generate_random_map(size=self.base_config['env_size'])

    def save_gif(self, policy, weights, filename, env_size, desc=None):
        env = Environment(size=env_size, desc=desc, render_mode="rgb_array")
        _, _, _, frames = env.evaluate_policy(policy, weights, render=True)
        if frames:
            imageio.mimsave(filename, frames, fps=5)

    def run_experiment_1(self):
        print("\n--- Running Experiment 1: Algorithm Combinations (ANOVA/T-Test) ---")
        crossovers = ['one_point', 'two_point', 'uniform']
        mutations = ['random_resetting', 'swap', 'insert', 'scramble', 'inversion']
        strategies = ['mu_plus_lambda', 'mu_comma_lambda']
        n_runs = 5
        
        # Tüm deneylerde ayni global haritayi kullaniyoruz
        env_size = self.base_config['env_size']
        base_desc = self.global_desc
        
        results_data = []
        best_overall = None
        best_fitness = -float('inf')
        
        fitness_matrix = []
        histories_data = {}

        total_combos = len(crossovers) * len(mutations) * len(strategies)
        count = 0
        
        for strat in strategies:
            for cx in crossovers:
                for mx in mutations:
                    count += 1
                    print(f"[{count}/{total_combos}] Testing {strat} | {cx} | {mx}")
                    
                    config = self.base_config.copy()
                    config['strategy'] = strat
                    config['crossover_method'] = cx
                    config['mutation_method'] = mx
                    
                    runs_fitness = []
                    runs_histories = []
                    runs_success = []
                    runs_steps = []
                    
                    for _ in range(n_runs):
                        env = Environment(size=config['env_size'], desc=base_desc)
                        ga = GeneticAlgorithm(config, env)
                        best_ind, history = ga.run_evolution()
                        runs_fitness.append(best_ind.fitness)
                        runs_histories.append(history['best_fitness'])
                        runs_success.append(best_ind.success)
                        runs_steps.append(best_ind.steps)
                        
                        if best_ind.fitness > best_fitness:
                            best_fitness = best_ind.fitness
                            best_overall = (best_ind, config)
                            
                    combo_name = f"{strat}\n{cx}+{mx}"
                    histories_data[combo_name] = runs_histories
                            
                    results_data.append({
                        'Strategy': strat,
                        'Crossover': cx,
                        'Mutation': mx,
                        'Combo_Name': combo_name,
                        'Mean_Fitness': np.mean(runs_fitness),
                        'Std_Fitness': np.std(runs_fitness),
                        'Mean_Success_Rate': np.mean(runs_success),
                        'Mean_Steps': np.mean(runs_steps)
                    })
                    fitness_matrix.append((f"{strat}_{cx}_{mx}", runs_fitness))

        df = pd.DataFrame(results_data)
        df.to_csv('results/exp1_combinations.csv', index=False)
        
        # Barplot for top 10
        df_sorted = df.sort_values(by='Mean_Fitness', ascending=False).head(10)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_sorted, x='Mean_Fitness', y='Combo_Name')
        plt.title('Top 10 Algorithm Combinations')
        plt.tight_layout()
        plt.savefig('results/exp1_top10_barplot.png')
        plt.close()

        # Convergence Grid for Top 10
        top10_names = df_sorted['Combo_Name'].tolist()
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
        axes = axes.flatten()
        
        for idx, name in enumerate(top10_names):
            ax = axes[idx]
            hists = np.array(histories_data[name])
            mean_fit = np.mean(hists, axis=0)
            std_fit = np.std(hists, axis=0)
            gens = np.arange(len(mean_fit))
            
            ax.plot(gens, mean_fit, color='blue')
            ax.fill_between(gens, mean_fit - std_fit, mean_fit + std_fit, color='blue', alpha=0.2)
            ax.set_title(name.replace('\n', ' | '), fontsize=9)
            if idx >= 5: ax.set_xlabel('Generation')
            if idx % 5 == 0: ax.set_ylabel('Fitness')
            ax.grid(True, linestyle='--', alpha=0.6)
            
        plt.suptitle("Top 10 Combinations: Convergence History", fontsize=16)
        plt.tight_layout()
        plt.savefig('results/exp1_top10_convergence.png')
        plt.close()

        if best_overall:
            ind, conf = best_overall
            self.save_gif(ind.genotype, conf['fitness_weights'], f"results/exp1_best_{conf['strategy']}_{conf['crossover_method']}_{conf['mutation_method']}.gif", conf['env_size'], base_desc)

        fitness_matrix.sort(key=lambda x: np.mean(x[1]), reverse=True)
        best_name, best_runs = fitness_matrix[0]
        second_name, second_runs = fitness_matrix[1]
        
        t_stat, p_val = stats.ttest_ind(best_runs, second_runs)
        print(f"\nStatistical Test (T-Test) between Top 1 ({best_name}) and Top 2 ({second_name}):")
        print(f"P-Value = {p_val:.5f} (If < 0.05, Top 1 is significantly better)")
        with open('results/exp1_pvalue.txt', 'w') as f:
            f.write(f"T-Test between {best_name} and {second_name}\nP-Value: {p_val}\n")

    def run_experiment_2a(self):
        print("\n--- Running Experiment 2a: Heatmap (Mutation vs Crossover Rate) ---")
        mut_rates = self.base_config['exp2_mutation_rates']
        cx_rates = self.base_config['exp2_crossover_rates']
        n_runs = 5
        
        base_desc = self.global_desc
        
        heatmap_data = np.zeros((len(mut_rates), len(cx_rates)))
        
        for i, m_rate in enumerate(mut_rates):
            for j, c_rate in enumerate(cx_rates):
                config = self.base_config.copy()
                config['mutation_rate'] = m_rate
                config['crossover_rate'] = c_rate
                
                fitness_sum = 0
                for _ in range(n_runs):
                    env = Environment(size=config['env_size'], desc=base_desc)
                    ga = GeneticAlgorithm(config, env)
                    best_ind, _ = ga.run_evolution()
                    fitness_sum += best_ind.fitness
                heatmap_data[i, j] = fitness_sum / n_runs
                
        df_heatmap = pd.DataFrame(heatmap_data, index=mut_rates, columns=cx_rates)
        df_heatmap.to_csv('results/exp2a_heatmap_data.csv')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, xticklabels=cx_rates, yticklabels=mut_rates, cmap="viridis")
        plt.title('Heatmap: Average Fitness (Mutation vs Crossover)')
        plt.xlabel('Crossover Rate')
        plt.ylabel('Mutation Rate')
        plt.savefig('results/exp2a_heatmap.png')
        plt.close()

    def run_experiment_2b(self):
        print("\n--- Running Experiment 2b: Sensitivity (Pop Size vs Tournament Size) ---")
        pop_sizes = self.base_config['exp2_pop_sizes']
        tourn_sizes = self.base_config['exp2_tournament_sizes']
        n_runs = 5
        
        base_desc = self.global_desc
        
        results = []
        for pop in pop_sizes:
            for trn in tourn_sizes:
                if trn > pop: continue
                config = self.base_config.copy()
                config['pop_size'] = pop
                config['tournament_size'] = trn
                
                fitness_sum = 0
                for _ in range(n_runs):
                    env = Environment(size=config['env_size'], desc=base_desc)
                    ga = GeneticAlgorithm(config, env)
                    best_ind, _ = ga.run_evolution()
                    fitness_sum += best_ind.fitness
                results.append({'Pop Size': pop, 'Tourn Size': trn, 'Avg Fitness': fitness_sum / n_runs})
                
        df = pd.DataFrame(results)
        df_pivot = df.pivot(index='Pop Size', columns='Tourn Size', values='Avg Fitness')
        df_pivot.to_csv('results/exp2b_sensitivity_data.csv')
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_pivot, annot=True, cmap="magma")
        plt.title('Heatmap: Pop Size vs Tournament Size')
        plt.savefig('results/exp2b_sensitivity.png')
        plt.close()

    def run_experiment_2c(self):
        print("\n--- Running Experiment 2c: Fitness Weights Sensitivity ---")
        weight_configs = self.base_config['exp2_fitness_weights']
        n_runs = 5
        
        base_desc = self.global_desc
        
        results = []
        for w_conf in weight_configs:
            config = self.base_config.copy()
            name = w_conf.pop('name')
            config['fitness_weights'] = w_conf
            
            steps_list = []
            success_list = []
            fitness_list = []
            for _ in range(n_runs):
                env = Environment(size=config['env_size'], desc=base_desc)
                ga = GeneticAlgorithm(config, env)
                best_ind, _ = ga.run_evolution()
                steps_list.append(best_ind.steps)
                success_list.append(best_ind.success)
                fitness_list.append(best_ind.fitness)
                
            results.append({
                'Weight Config': name,
                'Avg Fitness': np.mean(fitness_list),
                'Success Rate': np.mean(success_list),
                'Avg Steps': np.mean(steps_list)
            })
            
        df = pd.DataFrame(results)
        df.to_csv('results/exp2c_weights.csv', index=False)
        
        # Plotting the 3 subplots for Exp 2c
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sns.barplot(data=df, x='Weight Config', y='Success Rate', ax=axes[0], palette='viridis')
        axes[0].set_title('Success Rate vs Weights')
        axes[0].set_ylim(0, 1.1)
        
        sns.barplot(data=df, x='Weight Config', y='Avg Steps', ax=axes[1], palette='magma')
        axes[1].set_title('Average Steps vs Weights')
        
        sns.barplot(data=df, x='Weight Config', y='Avg Fitness', ax=axes[2], palette='coolwarm')
        axes[2].set_title('Average Fitness vs Weights')
        
        plt.tight_layout()
        plt.savefig('results/exp2c_weights_analysis.png')
        plt.close()

    def run_experiment_3(self):
        print("\n--- Running Experiment 3: Scalability (8x8, 16x16, 32x32) ---")
        sizes = [8, 16, 32]
        n_runs = 5
        
        plt.figure(figsize=(10, 6))
        
        for size in sizes:
            print(f"Testing Map Size {size}x{size}")
            config = self.base_config.copy()
            config['env_size'] = size
            config['max_generations'] = 150
            config['pop_size'] = 150
            
            # Use global map for 8x8, new random map for 16x16 and 32x32
            if size == self.base_config['env_size']:
                base_desc = self.global_desc
            else:
                base_desc = generate_random_map(size=size)
            
            all_histories = []
            for _ in range(n_runs):
                env = Environment(size=size, desc=base_desc)
                ga = GeneticAlgorithm(config, env)
                best_ind, history = ga.run_evolution()
                all_histories.append(history['best_fitness'])
            
            all_histories = np.array(all_histories)
            mean_fit = np.mean(all_histories, axis=0)
            std_fit = np.std(all_histories, axis=0)
            
            generations = np.arange(config['max_generations'])
            line, = plt.plot(generations, mean_fit, label=f'{size}x{size} Map')
            plt.fill_between(generations, mean_fit - std_fit, mean_fit + std_fit, alpha=0.2, color=line.get_color())

        plt.title('Scalability: Fitness Convergence over Different Map Sizes')
        plt.xlabel('Generation')
        plt.ylabel('Average Best Fitness')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/exp3_scalability_ci.png')
        plt.close()

    def run_all(self):
        self.run_experiment_1()
        self.run_experiment_2a()
        self.run_experiment_2b()
        self.run_experiment_2c()
        self.run_experiment_3()
