import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

class Environment:
    def __init__(self, size=4, desc=None, is_slippery=False, render_mode=None):
        self.size = size
        self.render_mode = render_mode
        self.is_slippery = is_slippery
        
        if desc is not None:
            self.map_name = None
            self.desc = desc
        elif size == 4:
            self.map_name = "4x4"
            self.desc = None
        elif size == 8:
            self.map_name = "8x8"
            self.desc = None
        else:
            self.map_name = None
            self.desc = generate_random_map(size=size)
            
        self.env = gym.make('FrozenLake-v1', desc=self.desc, map_name=self.map_name, is_slippery=self.is_slippery, render_mode=self.render_mode)
        self.goal_pos = (size - 1, size - 1)
        self.max_steps = size * size * 2

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def get_genotype_length(self):
        return self.size * self.size

    def evaluate_policy(self, policy, weights, render=False):
        state, info = self.env.reset()
        terminated = False
        truncated = False
        steps = 0
        success = 0
        fell_into_hole = 0

        frames = []
        if render and self.render_mode == "rgb_array":
            frames.append(self.env.render())

        while not terminated and not truncated and steps < self.max_steps:
            action = policy[state]
            state, reward, terminated, truncated, info = self.env.step(action)
            steps += 1
            
            if render and self.render_mode == "rgb_array":
                frames.append(self.env.render())

            if terminated:
                if reward == 1.0:
                    success = 1
                else:
                    fell_into_hole = 1

        final_r = state // self.size
        final_c = state % self.size
        
        manhattan_dist = abs(self.goal_pos[0] - final_r) + abs(self.goal_pos[1] - final_c)

        fitness = (
            (success * weights["success"])
            + (weights["manhattan_multiplier"] / (1 + manhattan_dist))
            - (steps * weights["step_penalty"])
            - (fell_into_hole * weights["hole_penalty"])
        )

        if render:
            return fitness, success, steps, frames
        return fitness, success, steps
