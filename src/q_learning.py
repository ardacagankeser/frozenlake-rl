import numpy as np

class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.95, 
                 initial_epsilon=1.0, min_epsilon=0.01, decay_rate=0.005):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        
        # Initialize Q-table
        self.q_table = np.zeros((num_states, num_actions))
        self.history = {'rewards': [], 'steps': [], 'success_rate': [], 'epsilon': []}

    def choose_action(self, state, evaluate=False):
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            actions = np.where(q_values == max_q)[0]
            return np.random.choice(actions)

    def update(self, state, action, reward, next_state, terminated):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (0.0 if terminated else self.gamma * self.q_table[next_state, best_next_action])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        return td_error

    def decay_epsilon(self, episode):
        self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def train(self, env, total_episodes=1000):
        print(f"Training Tabular Q-Learning for {total_episodes} episodes...")
        
        success_window = []
        window_size = 100
        
        for ep in range(total_episodes):
            state, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            steps = 0
            
            # Setup for potential-based reward shaping
            prev_r = state // env.size
            prev_c = state % env.size
            prev_dist = abs(env.goal_pos[0] - prev_r) + abs(env.goal_pos[1] - prev_c)
            
            while not terminated and not truncated and steps < env.max_steps:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Reward shaping calculation (Potential-Based Manhattan potential difference)
                next_r = next_state // env.size
                next_c = next_state % env.size
                next_dist = abs(env.goal_pos[0] - next_r) + abs(env.goal_pos[1] - next_c)
                
                success = (reward == 1.0)
                fell_into_hole = (terminated and reward == 0.0)
                
                potential_prev = 10.0 / (1 + prev_dist)
                potential_next = 10.0 / (1 + next_dist)
                
                # Shaped reward that provides immediate step gradients while maintaining policy optimality
                shaped_reward = (
                    (100.0 if success else 0.0)
                    - 0.1
                    - (50.0 if fell_into_hole else 0.0)
                    + (self.gamma * potential_next - potential_prev)
                )
                
                self.update(state, action, shaped_reward, next_state, terminated)
                
                state = next_state
                prev_dist = next_dist
                total_reward += reward
                steps += 1
                
            success_val = 1.0 if (terminated and total_reward > 0.0) else 0.0
            success_window.append(success_val)
            if len(success_window) > window_size:
                success_window.pop(0)
                
            self.history['rewards'].append(total_reward)
            self.history['steps'].append(steps)
            self.history['success_rate'].append(np.mean(success_window))
            self.history['epsilon'].append(self.epsilon)
            
            self.decay_epsilon(ep)
            
            if (ep + 1) % 200 == 0:
                print(f"Episode {ep+1}/{total_episodes} | Success Rate (last 100): {np.mean(success_window):.2f} | Epsilon: {self.epsilon:.3f}")
                
        return self.history

    def evaluate(self, env, num_episodes=50):
        successes = 0
        total_steps = 0
        
        for _ in range(num_episodes):
            state, info = env.reset()
            terminated = False
            truncated = False
            steps = 0
            
            while not terminated and not truncated and steps < env.max_steps:
                action = self.choose_action(state, evaluate=True)
                state, reward, terminated, truncated, info = env.step(action)
                steps += 1
                
            if terminated and reward > 0.0:
                successes += 1
                total_steps += steps
                
        success_rate = successes / num_episodes
        avg_steps = total_steps / successes if successes > 0 else env.max_steps
        
        return success_rate, avg_steps
