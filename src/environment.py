import gymnasium as gym

class Environment:
    def __init__(self, size=4):
        self.size = size
        self.map_name = "4x4" if size == 4 else "8x8"
        self.env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=False)
        self.goal_pos = (size - 1, size - 1)
        self.max_steps = size * size * 2

    def get_genotype_length(self):
        return self.size * self.size

    def evaluate_policy(self, policy, weights):
        state, info = self.env.reset()
        terminated = False  # episode is terminated if the agent falls into a hole or reaches the goal
        truncated = False   # episode is truncated if the agent takes too many steps
        steps = 0
        success = 0
        fell_into_hole = 0

        while not terminated and not truncated and steps < self.max_steps:
            action = policy[state]
            state, reward, terminated, truncated, info = self.env.step(action)
            steps += 1

            if terminated:
                if reward == 1.0:
                    success = 1
                else:
                    fell_into_hole = 1

        final_r = state // self.size
        final_c = state % self.size
        
        manhattan_dist = abs(self.goal_pos[0] - final_r) + abs(self.goal_pos[1] - final_c)

        fitness = (success * weights['success']) + \
                  (weights['manhattan_multiplier'] / (1 + manhattan_dist)) - \
                  (steps * weights['step_penalty']) - \
                  (fell_into_hole * weights['hole_penalty'])

        return fitness, success, steps
