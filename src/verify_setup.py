import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    print("--- Verifying PyTorch and Gymnasium Setup ---")
    try:
        import torch
        print(f"✅ PyTorch successfully imported. Version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch is NOT installed or could not be found.")
        sys.exit(1)
        
    try:
        import gymnasium as gym
        print(f"✅ Gymnasium successfully imported. Version: {gym.__version__}")
    except ImportError:
        print("❌ Gymnasium is NOT installed or could not be found.")
        sys.exit(1)
        
    print("\n--- Verifying Q-Learning Module ---")
    try:
        from src.q_learning import QLearningAgent
        from src.environment import Environment
        env = Environment(size=4)
        agent = QLearningAgent(num_states=16, num_actions=4)
        agent.train(env, total_episodes=5)
        print("✅ Q-Learning agent successfully instantiated and trained for 5 episodes.")
    except Exception as e:
        print(f"❌ Error in Q-Learning module: {str(e)}")
        sys.exit(1)

    print("\n--- Verifying DQN Module ---")
    try:
        from src.dqn import DQNAgent
        agent_dqn = DQNAgent(num_states=16, num_actions=4, hidden_dims=[16])
        agent_dqn.train(env, total_episodes=5)
        print("✅ DQN agent successfully instantiated and trained for 5 episodes.")
    except Exception as e:
        print(f"❌ Error in DQN module: {str(e)}")
        sys.exit(1)

    print("\n--- Setup Verification COMPLETE. Everything looks healthy! ---")

if __name__ == "__main__":
    main()
