import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.environment import Environment
from src.q_learning import QLearningAgent
from src.dqn import DQNAgent
from src.genetic_algorithm import GeneticAlgorithm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def test():
    size = 8
    desc = generate_random_map(size=size)
    print("--- Map ---")
    for r in desc:
        print(r)
        
    env = Environment(size=size, desc=desc, is_slippery=False)
    
    # Train Q-Learning
    print("\n--- Training Q-Learning ---")
    ql_agent = QLearningAgent(num_states=size*size, num_actions=4, decay_rate=0.01)
    history = ql_agent.train(env, total_episodes=800)
    print("Final Q-Learning Success Rate (training):", history['success_rate'][-1])
    
    ql_policy = [int(np.argmax(ql_agent.q_table[s])) for s in range(size*size)]
    print("Q-Learning Policy:")
    print(ql_policy)
    
    # Let's run a test rollout
    state, _ = env.reset()
    steps = 0
    terminated = False
    path = [state]
    while not terminated and steps < 100:
        action = ql_policy[state]
        state, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        path.append(state)
    print("QL Rollout path:", path, "Success:", reward > 0.0)

if __name__ == "__main__":
    test()
