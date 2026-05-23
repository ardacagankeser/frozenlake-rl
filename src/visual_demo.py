import time
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.environment import Environment
from src.genetic_algorithm import GeneticAlgorithm
from src.q_learning import QLearningAgent
from src.dqn import DQNAgent

# Premium ANSI Terminal Colors
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_MAGENTA = "\033[95m"
COLOR_CYAN = "\033[96m"
COLOR_GREY = "\033[90m"

ACTION_NAMES = {0: "⬅️ LEFT", 1: "⬇️ DOWN", 2: "➡️ RIGHT", 3: "⬆️ UP"}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_grid(size, desc, agent_pos, path_history, title, steps, max_steps, action_str, dist):
    clear_screen()
    print(f"{COLOR_BOLD}{COLOR_CYAN}============================================================={COLOR_RESET}")
    print(f"🎬 {COLOR_BOLD}{title}{COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_CYAN}============================================================={COLOR_RESET}\n")
    
    # Render the 8x8 Grid beautifully
    for r in range(size):
        row_str = "   "
        for c in range(size):
            cell_idx = r * size + c
            char = desc[r][c].decode('utf-8')
            pos = (r, c)
            
            if pos == agent_pos:
                row_str += f" 🤖 " # Agent icon
            elif pos == (0, 0):
                row_str += f" {COLOR_BOLD}{COLOR_YELLOW}S{COLOR_RESET} " # Start
            elif pos == (size - 1, size - 1):
                row_str += f" {COLOR_BOLD}{COLOR_GREEN}G{COLOR_RESET} " # Goal
            elif pos in path_history:
                row_str += f" {COLOR_GREY}·{COLOR_RESET} " # Visited path
            elif char == 'H':
                row_str += f" {COLOR_RED}H{COLOR_RESET} " # Hole
            else:
                row_str += f" {COLOR_BLUE}F{COLOR_RESET} " # Frozen Ice
        print(row_str)
    
    print(f"\n{COLOR_BOLD}{COLOR_CYAN}-------------------------------------------------------------{COLOR_RESET}")
    print(f"📍 Position: {COLOR_BOLD}{agent_pos}{COLOR_RESET} | Steps: {COLOR_BOLD}{steps}/{max_steps}{COLOR_RESET}")
    print(f"🚀 Action: {COLOR_BOLD}{action_str}{COLOR_RESET} | Manhattan Distance to Goal: {COLOR_BOLD}{dist}{COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_CYAN}============================================================={COLOR_RESET}")
    time.sleep(0.25) # Step animation delay

def animate_policy(env, policy, title):
    state, info = env.reset()
    terminated = False
    truncated = False
    steps = 0
    path_history = []
    
    action_str = "STARTING"
    r_pos = state // env.size
    c_pos = state % env.size
    dist = abs(env.goal_pos[0] - r_pos) + abs(env.goal_pos[1] - c_pos)
    
    draw_grid(env.size, env.desc, (r_pos, c_pos), path_history, title, steps, env.max_steps, action_str, dist)
    
    while not terminated and not truncated and steps < env.max_steps:
        path_history.append((r_pos, c_pos))
        action = policy[state]
        
        # Look up action name
        action_str = ACTION_NAMES[action]
        
        state, reward, terminated, truncated, info = env.step(action)
        steps += 1
        
        r_pos = state // env.size
        c_pos = state % env.size
        dist = abs(env.goal_pos[0] - r_pos) + abs(env.goal_pos[1] - c_pos)
        
        draw_grid(env.size, env.desc, (r_pos, c_pos), path_history, title, steps, env.max_steps, action_str, dist)
        
        if terminated:
            if reward > 0.0:
                print(f"\n🎉 {COLOR_BOLD}{COLOR_GREEN}GOAL SUCCESS! Agent safely navigated to the target!{COLOR_RESET} 🎉")
            else:
                print(f"\n💀 {COLOR_BOLD}{COLOR_RED}HOLE FATALITY! Agent slipped and fell into a freezing hole!{COLOR_RESET} 💀")
            time.sleep(2.0)
            break

def run_visual_demo():
    clear_screen()
    print(f"{COLOR_BOLD}{COLOR_MAGENTA}-------------------------------------------------------------{COLOR_RESET}")
    print(f"🏆 {COLOR_BOLD}FROZENLAKE-V1 MULTI-PARADIGM VISUAL DEMONSTRATION{COLOR_RESET} 🏆")
    print(f"{COLOR_BOLD}{COLOR_MAGENTA}-------------------------------------------------------------{COLOR_RESET}")
    print("This script will execute Q-Learning, DQN, and Genetic Algorithm runs")
    print("and demonstrate their navigation paths at various training stages (epochs).")
    time.sleep(3.0)

    # 1. Setup environment
    size = 8
    env = Environment(size=size)
    env_desc = env.desc

    # ==========================================
    # Tabular Q-Learning Epoch Demonstration
    # ==========================================
    print(f"\n🚀 {COLOR_BOLD}Initializing Tabular Q-Learning Agent...{COLOR_RESET}")
    ql_agent = QLearningAgent(num_states=size*size, num_actions=4, decay_rate=0.01)
    
    # Stage A: 100 Episodes (Early Exploration)
    ql_agent.train(env, total_episodes=100)
    ql_policy_100 = [np.argmax(ql_agent.q_table[s]) for s in range(size*size)]
    print("\n💡 Press ENTER to watch Q-Learning Policy Navigation after 100 episodes (Random Walk Stage)...")
    input()
    animate_policy(env, ql_policy_100, f"{COLOR_YELLOW}Q-LEARNING NAVIGATION (Stage: 100 Episodes){COLOR_RESET}")
    
    # Stage B: 300 Episodes (Getting Closer)
    ql_agent.train(env, total_episodes=200) # Trains 200 more, total 300
    ql_policy_300 = [np.argmax(ql_agent.q_table[s]) for s in range(size*size)]
    print("\n💡 Press ENTER to watch Q-Learning Policy Navigation after 300 episodes (Mid-Training Stage)...")
    input()
    animate_policy(env, ql_policy_300, f"{COLOR_YELLOW}Q-LEARNING NAVIGATION (Stage: 300 Episodes){COLOR_RESET}")

    # Stage C: 1000 Episodes (Fully Optimal)
    ql_agent.train(env, total_episodes=700) # Trains 700 more, total 1000
    ql_policy_1000 = [np.argmax(ql_agent.q_table[s]) for s in range(size*size)]
    print("\n💡 Press ENTER to watch Q-Learning Policy Navigation after 1000 episodes (Optimal 14-Step Path!)...")
    input()
    animate_policy(env, ql_policy_1000, f"{COLOR_GREEN}Q-LEARNING NAVIGATION (Stage: 1000 Episodes - Fully Learned){COLOR_RESET}")

    # ==========================================
    # DQN (Deep RL) Epoch Demonstration
    # ==========================================
    print(f"\n🚀 {COLOR_BOLD}Initializing PyTorch Deep Q-Network Agent...{COLOR_RESET}")
    dqn_agent = DQNAgent(num_states=size*size, num_actions=4, decay_rate=0.02)
    
    # Stage A: 100 Episodes
    dqn_agent.train(env, total_episodes=100)
    dqn_policy_100 = []
    for s in range(size*size):
        dqn_policy_100.append(dqn_agent.choose_action(s, evaluate=True))
    print("\n💡 Press ENTER to watch Deep DQN Navigation after 100 episodes (Neural Exploration Stage)...")
    input()
    animate_policy(env, dqn_policy_100, f"{COLOR_MAGENTA}DEEP DQN NAVIGATION (Stage: 100 Episodes){COLOR_RESET}")
    
    # Stage B: 500 Episodes (Optimal)
    dqn_agent.train(env, total_episodes=400) # Trains 400 more, total 500
    dqn_policy_500 = []
    for s in range(size*size):
        dqn_policy_500.append(dqn_agent.choose_action(s, evaluate=True))
    print("\n💡 Press ENTER to watch Deep DQN Navigation after 500 episodes (Optimal Neural path!)...")
    input()
    animate_policy(env, dqn_policy_500, f"{COLOR_GREEN}DEEP DQN NAVIGATION (Stage: 500 Episodes - Fully Converged){COLOR_RESET}")

    # ==========================================
    # Genetic Algorithm Epoch Demonstration
    # ==========================================
    print(f"\n🚀 {COLOR_BOLD}Initializing Genetic Algorithm Direct Policy Search...{COLOR_RESET}")
    config = {
        "env_size": size, "pop_size": 50, "max_generations": 5,
        "mutation_rate": 0.2, "crossover_rate": 0.8, "tournament_size": 3,
        "crossover_method": "uniform", "mutation_method": "random_resetting",
        "strategy": "mu_plus_lambda",
        "fitness_weights": {"success": 100.0, "manhattan_multiplier": 10.0, "step_penalty": 0.1, "hole_penalty": 50.0}
    }
    
    # Stage A: 5 Generations
    ga = GeneticAlgorithm(config, env)
    best_ind_5, _ = ga.run_evolution()
    print("\n💡 Press ENTER to watch GA Policy Navigation after 5 generations (Early Evolution)...")
    input()
    animate_policy(env, best_ind_5.genotype, f"{COLOR_CYAN}GENETIC ALGORITHM NAVIGATION (Stage: Generation 5){COLOR_RESET}")

    # Stage B: 50 Generations
    config["max_generations"] = 50
    ga_50 = GeneticAlgorithm(config, env)
    best_ind_50, _ = ga_50.run_evolution()
    print("\n💡 Press ENTER to watch GA Policy Navigation after 50 generations (Optimal Path!)...")
    input()
    animate_policy(env, best_ind_50.genotype, f"{COLOR_GREEN}GENETIC ALGORITHM NAVIGATION (Stage: Generation 50 - Fully Evolved){COLOR_RESET}")

    clear_screen()
    print(f"\n🎉 {COLOR_BOLD}{COLOR_GREEN}ALL DEMONSTRATION RUNS COMPLETED SUCCESSFULLY!{COLOR_RESET} 🎉")
    print("=============================================================")
    print("This visual demonstration showcases the complete learning process")
    print("from early exploration failures to final optimal navigation")
    print("across Tabular Q-learning, Deep DQN, and Genetic Algorithms.")
    print("=============================================================")

if __name__ == "__main__":
    run_visual_demo()
