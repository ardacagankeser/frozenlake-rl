import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32), 
                np.array(next_state), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DQNNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(self, num_states, num_actions, hidden_dims=[64, 64], learning_rate=1e-3, 
                 discount_factor=0.95, buffer_size=10000, batch_size=64, target_update_freq=100,
                 initial_epsilon=1.0, min_epsilon=0.01, decay_rate=0.005, device="cpu"):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.device = torch.device(device)

        # Main and Target networks
        self.q_network = DQNNetwork(num_states, hidden_dims, num_actions).to(self.device)
        self.target_network = DQNNetwork(num_states, hidden_dims, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        self.steps_done = 0
        self.history = {'rewards': [], 'steps': [], 'success_rate': [], 'loss': [], 'epsilon': []}

    def _state_to_onehot(self, state):
        onehot = np.zeros(self.num_states, dtype=np.float32)
        onehot[state] = 1.0
        return torch.tensor(onehot, device=self.device)

    def _states_to_onehot_batch(self, states):
        batch_size = len(states)
        onehots = np.zeros((batch_size, self.num_states), dtype=np.float32)
        onehots[np.arange(batch_size), states] = 1.0
        return torch.tensor(onehots, device=self.device)

    def choose_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = self._state_to_onehot(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        state_batch = self._states_to_onehot_batch(states)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_state_batch = self._states_to_onehot_batch(next_states)
        done_batch = torch.tensor(dones, device=self.device).unsqueeze(1)

        q_values = self.q_network(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(dim=1)[0].unsqueeze(1)
            target_q_values = reward_batch + (1.0 - done_batch) * self.gamma * next_q_values

        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self, episode):
        self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def train(self, env, total_episodes=500):
        print(f"Training Deep Q-Network for {total_episodes} episodes...")
        
        success_window = []
        window_size = 100
        
        for ep in range(total_episodes):
            state, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            steps = 0
            ep_losses = []
            
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
                
                shaped_reward = (
                    (100.0 if success else 0.0)
                    - 0.1
                    - (50.0 if fell_into_hole else 0.0)
                    + (self.gamma * potential_next - potential_prev)
                )
                
                # Push transition to replay memory
                self.memory.push(state, action, shaped_reward, next_state, float(terminated and reward > 0))
                
                # Train network
                loss_val = self.train_step()
                if loss_val > 0.0:
                    ep_losses.append(loss_val)
                    
                state = next_state
                prev_dist = next_dist
                total_reward += reward
                steps += 1
                
            success_val = 1.0 if (terminated and total_reward > 0.0) else 0.0
            success_window.append(success_val)
            if len(success_window) > window_size:
                success_window.pop(0)
                
            mean_loss = np.mean(ep_losses) if ep_losses else 0.0
            
            self.history['rewards'].append(total_reward)
            self.history['steps'].append(steps)
            self.history['success_rate'].append(np.mean(success_window))
            self.history['loss'].append(mean_loss)
            self.history['epsilon'].append(self.epsilon)
            
            self.decay_epsilon(ep)
            
            if (ep + 1) % 100 == 0:
                print(f"Episode {ep+1}/{total_episodes} | Success Rate (last 100): {np.mean(success_window):.2f} | Loss: {mean_loss:.4f} | Epsilon: {self.epsilon:.3f}")
                
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
