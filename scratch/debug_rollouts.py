import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gui_demo import DemoManager

def debug_rollouts():
    manager = DemoManager(size=8, is_slippery=False)
    print("Training agents...")
    manager.train_all_agents(progress_callback=lambda txt, p: print(f" {p}%: {txt}"))
    
    print("\n--- Map ---")
    for r in manager.shared_desc:
        print(r)
        
    print("\n--- Policies and rollouts ---")
    for alg in ["GA", "QL", "DQN"]:
        print(f"\nEvaluating {alg} Fully Converged Policy (Stage 2):")
        policy = manager.policies[alg][2]
        print(f"Policy: {policy}")
        
        # Test rollout
        env = manager.env_ql if alg == "QL" else (manager.env_ga if alg == "GA" else manager.env_dqn)
        state, _ = env.reset()
        steps = 0
        path = [state]
        terminated = False
        truncated = False
        while not terminated and not truncated and steps < 100:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            path.append(state)
        
        print(f"Rollout path: {path}")
        print(f"Steps: {steps} | Terminated: {terminated} | Reward: {reward} | Success: {reward > 0.0}")

if __name__ == "__main__":
    debug_rollouts()
