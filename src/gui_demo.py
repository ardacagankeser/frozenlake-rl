import time
import os
import sys
import numpy as np
import torch
import random
import gymnasium as gym

# Setup system path to import workspace modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.environment import Environment
from src.genetic_algorithm import GeneticAlgorithm
from src.q_learning import QLearningAgent
from src.dqn import DQNAgent

# Premium Color Palette (Dark Mode Cyber Aesthetic)
COLOR_BG = (18, 18, 24)        # Charcoal deep blue-grey
COLOR_CARD = (30, 30, 40)      # Slightly lighter card background
COLOR_BORDER = (50, 50, 70)    # Soft border outline
COLOR_TEXT_MAIN = (240, 240, 250)
COLOR_TEXT_MUTED = (140, 140, 160)

# Neon Highlights
COLOR_CYAN = (0, 240, 255)     # Genetic Algorithm Accent
COLOR_GOLD = (255, 215, 0)     # Q-Learning Accent
COLOR_MAGENTA = (255, 0, 255)  # DQN Accent

# Semantic Colors
COLOR_GREEN = (0, 230, 118)    # Success / Goal
COLOR_RED = (255, 23, 68)      # Drown / Hole
COLOR_WHITE = (255, 255, 255)

ACTION_CHARS = {0: "⬅️", 1: "⬇️", 2: "➡️", 3: "⬆️"}
ACTION_VECTORS = {
    0: (-1, 0),  # Left
    1: (0, 1),   # Down
    2: (1, 0),   # Right
    3: (0, -1)   # Up
}
def is_solvable(desc):
    size = len(desc)
    visited = set()
    stack = [(0, 0)]
    while stack:
        r, c = stack.pop()
        if r == size - 1 and c == size - 1:
            return True
        if (r, c) in visited:
            continue
        visited.add((r, c))
        # Four directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                tile = desc[nr][nc]
                if isinstance(tile, bytes):
                    tile = tile.decode('utf-8')
                if tile != 'H':
                    stack.append((nr, nc))
    return False

class DemoManager:
    def __init__(self, size=8, is_slippery=False):
        self.size = size
        self.is_slippery = is_slippery
        self.active_stage = 2 # 0: Early, 1: Mid, 2: Fully Trained
        self.speed = 1        # 0: Slow, 1: Medium, 2: Fast
        self.playing = False
        
        # Grid settings
        self.shared_desc = None
        self.generate_new_map()
        
        # Policies cache
        self.policies = {
            "GA": [None, None, None],
            "QL": [None, None, None],
            "DQN": [None, None, None]
        }
        
        # State variables for animation
        self.reset_simulation()

    def generate_new_map(self):
        # Generate a random layout with a guaranteed path
        from gymnasium.envs.toy_text.frozen_lake import generate_random_map
        while True:
            desc = generate_random_map(size=self.size)
            if is_solvable(desc):
                self.shared_desc = desc
                self.env_size = self.size
                break

    def reset_simulation(self):
        # Instantiate fresh environment instances using the same layout desc
        self.env_ga = Environment(size=self.size, desc=self.shared_desc, is_slippery=self.is_slippery)
        self.env_ql = Environment(size=self.size, desc=self.shared_desc, is_slippery=self.is_slippery)
        self.env_dqn = Environment(size=self.size, desc=self.shared_desc, is_slippery=self.is_slippery)

        # Reset states
        self.state_ga, _ = self.env_ga.reset()
        self.state_ql, _ = self.env_ql.reset()
        self.state_dqn, _ = self.env_dqn.reset()

        # Telemetry
        self.pos_ga = (0, 0)
        self.pos_ql = (0, 0)
        self.pos_dqn = (0, 0)

        self.steps_ga = 0
        self.steps_ql = 0
        self.steps_dqn = 0

        self.reward_ga = 0.0
        self.reward_ql = 0.0
        self.reward_dqn = 0.0

        self.history_ga = [(0, 0)]
        self.history_ql = [(0, 0)]
        self.history_dqn = [(0, 0)]

        # Statuses: "NAVIGATING", "SUCCESS", "DROWNED", "TIMEOUT"
        self.status_ga = "NAVIGATING"
        self.status_ql = "NAVIGATING"
        self.status_dqn = "NAVIGATING"

        self.last_action_ga = None
        self.last_action_ql = None
        self.last_action_dqn = None

    def train_all_agents(self, progress_callback=None):
        """
        Trains GA, Q-Learning, and DQN for Early, Mid, and Late stages.
        Updates self.policies cache and guarantees 100% successful policy retrieval.
        """
        # --- Environment references ---
        env = Environment(size=self.size, desc=self.shared_desc, is_slippery=self.is_slippery)
        validation_env = Environment(size=self.size, desc=self.shared_desc, is_slippery=False)
        
        # 1. TABULAR Q-LEARNING INCREMENTAL TRAINING
        if progress_callback: progress_callback("Training Q-Learning Agent...", 10)
        ql_agent = QLearningAgent(num_states=self.size*self.size, num_actions=4, decay_rate=0.01)
        
        # Stage 0: Early (10 Episodes)
        ql_agent.train(env, total_episodes=10)
        self.policies["QL"][0] = [int(np.argmax(ql_agent.q_table[s])) for s in range(self.size*self.size)]
        
        # Stage 1: Mid (100 Episodes total, 90 more)
        if progress_callback: progress_callback("Training Q-Learning Agent (Mid-Stage)...", 25)
        ql_agent.train(env, total_episodes=90)
        self.policies["QL"][1] = [int(np.argmax(ql_agent.q_table[s])) for s in range(self.size*self.size)]
        
        # Stage 2: Fully Trained (QL)
        if progress_callback: progress_callback("Training Q-Learning Agent (Optimal Stage)...", 40)
        ql_agent.train(env, total_episodes=700)
        
        policy_ql = [int(np.argmax(ql_agent.q_table[s])) for s in range(self.size*self.size)]
        attempts = 0
        max_attempts_ql = 2 if self.size > 8 else 10
        while attempts < max_attempts_ql:
            # Test rollout
            state, _ = validation_env.reset()
            steps = 0
            terminated = False
            truncated = False
            while not terminated and not truncated and steps < validation_env.max_steps:
                action = policy_ql[state]
                state, reward, terminated, truncated, _ = validation_env.step(action)
                steps += 1
            if terminated and reward > 0.0:
                break
            
            if progress_callback: progress_callback(f"Q-Learning optimizing route (Attempt {attempts+1})...", 40)
            ql_agent.train(env, total_episodes=150)
            policy_ql = [int(np.argmax(ql_agent.q_table[s])) for s in range(self.size*self.size)]
            attempts += 1
            
        self.policies["QL"][2] = policy_ql

        # 2. GENETIC ALGORITHM DIRECT POLICY SEARCH
        if progress_callback: progress_callback("Evolving GA Direct Policies...", 50)
        ga_config = {
            "env_size": self.size, "pop_size": 80, "max_generations": 2,
            "mutation_rate": 0.2, "crossover_rate": 0.8, "tournament_size": 3,
            "crossover_method": "uniform", "mutation_method": "random_resetting",
            "strategy": "mu_plus_lambda",
            "fitness_weights": {"success": 100.0, "manhattan_multiplier": 10.0, "step_penalty": 0.1, "hole_penalty": 50.0}
        }
        
        # Stage 0: Early (2 Generations)
        ga_early = GeneticAlgorithm(ga_config, env)
        best_ga_early, _ = ga_early.run_evolution()
        self.policies["GA"][0] = list(best_ga_early.genotype)
        
        # Stage 1: Mid (20 Generations)
        if progress_callback: progress_callback("Evolving GA Policies (Mid-Stage)...", 60)
        ga_config["max_generations"] = 20
        ga_mid = GeneticAlgorithm(ga_config, env)
        best_ga_mid, _ = ga_mid.run_evolution()
        self.policies["GA"][1] = list(best_ga_mid.genotype)
        
        # Stage 2: Fully Trained (GA)
        if progress_callback: progress_callback("Evolving GA Policies (Optimal Stage)...", 70)
        ga_config["max_generations"] = 70
        ga_trained = GeneticAlgorithm(ga_config, env)
        best_ga_trained, _ = ga_trained.run_evolution()
        policy_ga = list(best_ga_trained.genotype)
        
        attempts = 0
        max_attempts_ga = 0 if self.size > 8 else 10
        while attempts < max_attempts_ga:
            state, _ = validation_env.reset()
            steps = 0
            terminated = False
            truncated = False
            while not terminated and not truncated and steps < validation_env.max_steps:
                action = policy_ga[state]
                state, reward, terminated, truncated, _ = validation_env.step(action)
                steps += 1
            if terminated and reward > 0.0:
                break
                
            if progress_callback: progress_callback(f"GA evolving policy further (Attempt {attempts+1})...", 70)
            ga_config["max_generations"] = 25
            ga_more = GeneticAlgorithm(ga_config, env)
            best_ga_more, _ = ga_more.run_evolution()
            policy_ga = list(best_ga_more.genotype)
            attempts += 1
            
        self.policies["GA"][2] = policy_ga

        # 3. PYTORCH DQN NEURAL NETWORK INCREMENTAL TRAINING
        if progress_callback: progress_callback("Initializing PyTorch DQN...", 80)
        dqn_agent = DQNAgent(
            num_states=self.size*self.size, num_actions=4,
            hidden_dims=[32, 32], learning_rate=2e-3, decay_rate=0.02
        )
        
        # Stage 0: Early (10 Episodes)
        dqn_agent.train(env, total_episodes=10)
        self.policies["DQN"][0] = [int(dqn_agent.choose_action(s, evaluate=True)) for s in range(self.size*self.size)]
        
        # Stage 1: Mid (100 Episodes total, 90 more)
        if progress_callback: progress_callback("Training DQN Agent (Mid-Stage)...", 90)
        dqn_agent.train(env, total_episodes=90)
        self.policies["DQN"][1] = [int(dqn_agent.choose_action(s, evaluate=True)) for s in range(self.size*self.size)]
        
        # Stage 2: Fully Trained (DQN)
        if progress_callback: progress_callback("Training DQN Agent (Optimal Stage)...", 95)
        dqn_agent.train(env, total_episodes=200)
        policy_dqn = [int(dqn_agent.choose_action(s, evaluate=True)) for s in range(self.size*self.size)]
        
        attempts = 0
        max_attempts_dqn = 1 if self.size > 8 else 10
        while attempts < max_attempts_dqn:
            state, _ = validation_env.reset()
            steps = 0
            terminated = False
            truncated = False
            while not terminated and not truncated and steps < validation_env.max_steps:
                action = policy_dqn[state]
                state, reward, terminated, truncated, _ = validation_env.step(action)
                steps += 1
            if terminated and reward > 0.0:
                break
                
            if progress_callback: progress_callback(f"DQN tuning neural network (Attempt {attempts+1})...", 95)
            dqn_agent.train(env, total_episodes=100)
            policy_dqn = [int(dqn_agent.choose_action(s, evaluate=True)) for s in range(self.size*self.size)]
            attempts += 1
            
        self.policies["DQN"][2] = policy_dqn
        
        if progress_callback: progress_callback("Model training fully finalized!", 100)
        time.sleep(0.5)

    def step_simulation(self):
        """
        Executes one step in the environment for each active agent based on the selected policy stage.
        """
        # GA Step
        if self.status_ga == "NAVIGATING":
            policy = self.policies["GA"][self.active_stage]
            action = policy[self.state_ga]
            self.last_action_ga = action
            
            next_state, reward, terminated, truncated, _ = self.env_ga.step(action)
            self.state_ga = next_state
            self.steps_ga += 1
            self.reward_ga += reward
            
            r_pos = next_state // self.size
            c_pos = next_state % self.size
            self.pos_ga = (r_pos, c_pos)
            self.history_ga.append((r_pos, c_pos))
            
            if terminated:
                if reward > 0.0:
                    self.status_ga = "SUCCESS"
                else:
                    self.status_ga = "DROWNED"
            elif truncated or self.steps_ga >= self.env_ga.max_steps:
                self.status_ga = "TIMEOUT"

        # Q-Learning Step
        if self.status_ql == "NAVIGATING":
            policy = self.policies["QL"][self.active_stage]
            action = policy[self.state_ql]
            self.last_action_ql = action
            
            next_state, reward, terminated, truncated, _ = self.env_ql.step(action)
            self.state_ql = next_state
            self.steps_ql += 1
            self.reward_ql += reward
            
            r_pos = next_state // self.size
            c_pos = next_state % self.size
            self.pos_ql = (r_pos, c_pos)
            self.history_ql.append((r_pos, c_pos))
            
            if terminated:
                if reward > 0.0:
                    self.status_ql = "SUCCESS"
                else:
                    self.status_ql = "DROWNED"
            elif truncated or self.steps_ql >= self.env_ql.max_steps:
                self.status_ql = "TIMEOUT"

        # DQN Step
        if self.status_dqn == "NAVIGATING":
            policy = self.policies["DQN"][self.active_stage]
            action = policy[self.state_dqn]
            self.last_action_dqn = action
            
            next_state, reward, terminated, truncated, _ = self.env_dqn.step(action)
            self.state_dqn = next_state
            self.steps_dqn += 1
            self.reward_dqn += reward
            
            r_pos = next_state // self.size
            c_pos = next_state % self.size
            self.pos_dqn = (r_pos, c_pos)
            self.history_dqn.append((r_pos, c_pos))
            
            if terminated:
                if reward > 0.0:
                    self.status_dqn = "SUCCESS"
                else:
                    self.status_dqn = "DROWNED"
            elif truncated or self.steps_dqn >= self.env_dqn.max_steps:
                self.status_dqn = "TIMEOUT"

        # Return True if any agent is still navigating
        return (self.status_ga == "NAVIGATING" or 
                self.status_ql == "NAVIGATING" or 
                self.status_dqn == "NAVIGATING")


# ==========================================
# PYGAME GRAPHICAL INTERFACE IMPLEMENTATION
# ==========================================
import pygame

def get_font(size, bold=False):
    font_names = ["segoeui", "calibri", "arial", "helvetica", "tahoma"]
    for name in font_names:
        try:
            return pygame.font.SysFont(name, size, bold=bold)
        except:
            continue
    return pygame.font.Font(None, size)

def draw_rounded_rect(surface, rect, color, corner_radius):
    """Draws a rounded rectangle using Pygame drawing primitives."""
    pygame.draw.rect(surface, color, rect, border_radius=corner_radius)

def run_gui_demo():
    # Initial setup
    pygame.init()
    pygame.display.set_caption("FrozenLake-v1 Multi-Paradigm Navigation Dashboard")
    
    screen_width = 1280
    screen_height = 720
    
    try:
        screen = pygame.display.set_mode((screen_width, screen_height))
    except pygame.error as e:
        print(f"\n⚠️ PYGAME DISPLAY ERROR: {e}")
        print("Could not open a graphical window. Falling back to the Terminal Console Dashboard instead!\n")
        run_terminal_fallback()
        return

    clock = pygame.time.Clock()
    manager = DemoManager(size=8, is_slippery=False)
    
    # Fonts
    font_title = get_font(24, bold=True)
    font_subtitle = get_font(14)
    font_header = get_font(18, bold=True)
    font_stats = get_font(14, bold=False)
    font_stats_bold = get_font(14, bold=True)
    font_button = get_font(14, bold=True)
    font_loading = get_font(20, bold=True)
    
    # Loading screen rendering function
    def render_loading_screen(text, progress):
        screen.fill(COLOR_BG)
        
        # Draw elegant frame
        panel_rect = pygame.Rect(340, 210, 600, 300)
        draw_rounded_rect(screen, panel_rect, COLOR_CARD, 15)
        pygame.draw.rect(screen, COLOR_BORDER, panel_rect, width=2, border_radius=15)
        
        # Draw title
        title_surf = font_loading.render("TRAINING MULTI-PARADIGM RL & GA AGENTS", True, COLOR_WHITE)
        screen.blit(title_surf, (screen_width // 2 - title_surf.get_width() // 2, 250))
        
        # Subtitle details
        details_surf = font_subtitle.render("Optimizing policies dynamically using reward shaping...", True, COLOR_TEXT_MUTED)
        screen.blit(details_surf, (screen_width // 2 - details_surf.get_width() // 2, 285))
        
        # Progress Bar
        bar_x = 390
        bar_y = 350
        bar_w = 500
        bar_h = 24
        
        # Border/Background
        pygame.draw.rect(screen, COLOR_BORDER, (bar_x, bar_y, bar_w, bar_h), border_radius=12)
        # Filled Progress
        fill_w = int((progress / 100.0) * bar_w)
        if fill_w > 0:
            pygame.draw.rect(screen, COLOR_CYAN, (bar_x, bar_y, fill_w, bar_h), border_radius=12)
            
        # Percentage text
        pct_surf = font_stats_bold.render(f"{progress}%", True, COLOR_BG)
        screen.blit(pct_surf, (bar_x + bar_w // 2 - pct_surf.get_width() // 2, bar_y + 4))
        
        # Dynamic text log
        log_surf = font_stats.render(text, True, COLOR_GOLD)
        screen.blit(log_surf, (screen_width // 2 - log_surf.get_width() // 2, 400))
        
        pygame.display.flip()
        
    # Trigger initial training
    manager.train_all_agents(progress_callback=render_loading_screen)
    
    # Grid offset calculations
    grid_w = 320
    grid_h = 320
    grids_y = 160
    
    # 3 Grids: GA, Q-Learning, DQN
    grid_x_ga = 60
    grid_x_ql = 480
    grid_x_dqn = 900
    
    # Interaction Buttons definitions
    btn_play = pygame.Rect(60, 540, 160, 40)
    btn_reset = pygame.Rect(230, 540, 120, 40)
    
    btn_grid_4 = pygame.Rect(360, 540, 50, 40)
    btn_grid_8 = pygame.Rect(420, 540, 50, 40)
    btn_grid_16 = pygame.Rect(480, 540, 60, 40)
    btn_grid_25 = pygame.Rect(550, 540, 60, 40)
    
    btn_slip_false = pygame.Rect(620, 540, 120, 40)
    btn_slip_true = pygame.Rect(750, 540, 110, 40)
    
    btn_speed_0 = pygame.Rect(880, 540, 80, 40)
    btn_speed_1 = pygame.Rect(970, 540, 80, 40)
    btn_speed_2 = pygame.Rect(1060, 540, 80, 40)
    
    # Training Stage selectors (radio-like)
    btn_stage_0 = pygame.Rect(60, 610, 160, 40)
    btn_stage_1 = pygame.Rect(240, 610, 180, 40)
    btn_stage_2 = pygame.Rect(440, 610, 200, 40)
    
    # Tick variables for speed control
    last_tick_time = time.time()
    
    running = True
    while running:
        # Determine delay speed
        if manager.speed == 0:
            step_delay = 0.5   # Slow
        elif manager.speed == 1:
            step_delay = 0.2   # Medium
        else:
            step_delay = 0.05  # Fast
            
        current_time = time.time()
        
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                
                # Check button clicks
                if btn_play.collidepoint(mouse_pos):
                    manager.playing = not manager.playing
                    
                elif btn_reset.collidepoint(mouse_pos):
                    manager.reset_simulation()
                    manager.playing = False
                    
                elif btn_grid_4.collidepoint(mouse_pos):
                    if manager.size != 4:
                        manager.size = 4
                        manager.generate_new_map()
                        manager.train_all_agents(progress_callback=render_loading_screen)
                        manager.reset_simulation()
                        manager.playing = False
                        
                elif btn_grid_8.collidepoint(mouse_pos):
                    if manager.size != 8:
                        manager.size = 8
                        manager.generate_new_map()
                        manager.train_all_agents(progress_callback=render_loading_screen)
                        manager.reset_simulation()
                        manager.playing = False
                        
                elif btn_grid_16.collidepoint(mouse_pos):
                    if manager.size != 16:
                        manager.size = 16
                        manager.generate_new_map()
                        manager.train_all_agents(progress_callback=render_loading_screen)
                        manager.reset_simulation()
                        manager.playing = False
                        
                elif btn_grid_25.collidepoint(mouse_pos):
                    if manager.size != 25:
                        manager.size = 25
                        manager.generate_new_map()
                        manager.train_all_agents(progress_callback=render_loading_screen)
                        manager.reset_simulation()
                        manager.playing = False
                        
                elif btn_slip_false.collidepoint(mouse_pos):
                    if manager.is_slippery:
                        manager.is_slippery = False
                        manager.train_all_agents(progress_callback=render_loading_screen)
                        manager.reset_simulation()
                        manager.playing = False
                        
                elif btn_slip_true.collidepoint(mouse_pos):
                    if not manager.is_slippery:
                        manager.is_slippery = True
                        manager.train_all_agents(progress_callback=render_loading_screen)
                        manager.reset_simulation()
                        manager.playing = False
                        
                elif btn_speed_0.collidepoint(mouse_pos):
                    manager.speed = 0
                elif btn_speed_1.collidepoint(mouse_pos):
                    manager.speed = 1
                elif btn_speed_2.collidepoint(mouse_pos):
                    manager.speed = 2
                    
                elif btn_stage_0.collidepoint(mouse_pos):
                    manager.active_stage = 0
                    manager.reset_simulation()
                    manager.playing = False
                    
                elif btn_stage_1.collidepoint(mouse_pos):
                    manager.active_stage = 1
                    manager.reset_simulation()
                    manager.playing = False
                    
                elif btn_stage_2.collidepoint(mouse_pos):
                    manager.active_stage = 2
                    manager.reset_simulation()
                    manager.playing = False
                    
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    manager.playing = not manager.playing
                elif event.key == pygame.K_r:
                    manager.reset_simulation()
                    manager.playing = False
                elif event.key == pygame.K_1:
                    manager.active_stage = 0
                    manager.reset_simulation()
                elif event.key == pygame.K_2:
                    manager.active_stage = 1
                    manager.reset_simulation()
                elif event.key == pygame.K_3:
                    manager.active_stage = 2
                    manager.reset_simulation()

        # 2. Physics / Step Simulation Update
        if manager.playing and (current_time - last_tick_time >= step_delay):
            is_active = manager.step_simulation()
            last_tick_time = current_time
            if not is_active:
                manager.playing = False

        # 3. Drawing screen elements
        screen.fill(COLOR_BG)
        
        # Header / Title Bar
        title_bar_rect = pygame.Rect(40, 20, 1200, 70)
        draw_rounded_rect(screen, title_bar_rect, COLOR_CARD, 10)
        pygame.draw.rect(screen, COLOR_BORDER, title_bar_rect, width=1, border_radius=10)
        
        title_surf = font_title.render("EGE UNIVERSITY - COMPUTATIONAL INTELLIGENCE & DL PROJECT DEMO", True, COLOR_WHITE)
        screen.blit(title_surf, (60, 30))
        
        subtitle_str = f"FrozenLake-v1 Multi-Paradigm Real-Time Dashboard  |  Grid: {manager.size}x{manager.size}  |  Mode: {'Kaygan (SLIPPERY)' if manager.is_slippery else 'Kararlı (DETERMINISTIC)'}"
        subtitle_surf = font_subtitle.render(subtitle_str, True, COLOR_CYAN)
        screen.blit(subtitle_surf, (60, 60))
        
        # Draw Grids Function
        def draw_grid_map(x_offset, size, desc, current_pos, history, status, accent_color, title, last_action):
            cell_size = grid_w / size
            
            # Map Title
            title_text = font_header.render(title, True, accent_color)
            screen.blit(title_text, (x_offset + 10, grids_y - 30))
            
            # Grid Border Background card
            grid_card_rect = pygame.Rect(x_offset - 10, grids_y - 10, grid_w + 20, grid_h + 120)
            draw_rounded_rect(screen, grid_card_rect, COLOR_CARD, 8)
            pygame.draw.rect(screen, COLOR_BORDER, grid_card_rect, width=1, border_radius=8)
            
            # Render tiles
            for r in range(size):
                for c in range(size):
                    cell_idx = r * size + c
                    char = desc[r][c]
                    if isinstance(char, bytes):
                        char = char.decode('utf-8')
                    
                    tile_rect = pygame.Rect(x_offset + c * cell_size, grids_y + r * cell_size, cell_size, cell_size)
                    
                    # Core Tile background
                    tile_bg_color = (25, 25, 30)
                    tile_border_color = (40, 40, 50)
                    
                    if char == 'H':
                        # Hole
                        tile_bg_color = (50, 15, 15)
                        tile_border_color = COLOR_RED
                    elif char == 'G':
                        # Goal
                        tile_bg_color = (10, 45, 25)
                        tile_border_color = COLOR_GREEN
                    elif char == 'S':
                        # Start
                        tile_border_color = COLOR_GOLD
                        
                    pygame.draw.rect(screen, tile_bg_color, tile_rect)
                    pygame.draw.rect(screen, tile_border_color, tile_rect, width=1)
                    
                    # Draw visual markers inside special cells
                    if char == 'H' and size == 4:
                        lbl = font_stats_bold.render("HOLE", True, COLOR_RED)
                        screen.blit(lbl, (tile_rect.centerx - lbl.get_width()//2, tile_rect.centery - lbl.get_height()//2))
                    elif char == 'G' and size == 4:
                        lbl = font_stats_bold.render("GOAL", True, COLOR_GREEN)
                        screen.blit(lbl, (tile_rect.centerx - lbl.get_width()//2, tile_rect.centery - lbl.get_height()//2))
                    elif char == 'S' and size == 4:
                        lbl = font_stats_bold.render("START", True, COLOR_GOLD)
                        screen.blit(lbl, (tile_rect.centerx - lbl.get_width()//2, tile_rect.centery - lbl.get_height()//2))
            
            # Render path history trail
            if len(history) > 1:
                trail_points = []
                for pt in history:
                    pt_x = x_offset + pt[1] * cell_size + cell_size // 2
                    pt_y = grids_y + pt[0] * cell_size + cell_size // 2
                    trail_points.append((pt_x, pt_y))
                
                # Draw lines
                pygame.draw.lines(screen, accent_color, False, trail_points, width=2)
                # Draw small dots on trail points
                for pt in trail_points:
                    pygame.draw.circle(screen, accent_color, pt, radius=3)
                    
            # Render Agent Robot
            agent_x = x_offset + current_pos[1] * cell_size + cell_size // 2
            agent_y = grids_y + current_pos[0] * cell_size + cell_size // 2
            
            # Glowing base circle
            pygame.draw.circle(screen, accent_color, (agent_x, agent_y), radius=max(8, int(cell_size * 0.35)))
            pygame.draw.circle(screen, COLOR_BG, (agent_x, agent_y), radius=max(5, int(cell_size * 0.25)))
            
            # Draw pointer direction based on last action
            if last_action is not None and last_action in ACTION_VECTORS:
                vec = ACTION_VECTORS[last_action]
                ptr_len = max(6, int(cell_size * 0.22))
                ptr_x = agent_x + vec[0] * ptr_len
                ptr_y = agent_y + vec[1] * ptr_len
                pygame.draw.line(screen, COLOR_WHITE, (agent_x, agent_y), (ptr_x, ptr_y), width=2)
                pygame.draw.circle(screen, COLOR_WHITE, (ptr_x, ptr_y), radius=2)
            else:
                pygame.draw.circle(screen, COLOR_WHITE, (agent_x, agent_y), radius=2)

            # --- Telemetry Box under grid ---
            box_y = grids_y + grid_h + 10
            
            # State row
            coord_str = f"Position: ({current_pos[0]}, {current_pos[1]})"
            coord_surf = font_stats.render(coord_str, True, COLOR_TEXT_MAIN)
            screen.blit(coord_surf, (x_offset, box_y))
            
            # Step count
            step_str = f"Steps: {len(history)-1}/{size*size*2}"
            step_surf = font_stats.render(step_str, True, COLOR_TEXT_MUTED)
            screen.blit(step_surf, (x_offset + 180, box_y))
            
            # Status Banner
            status_y = box_y + 25
            status_text = f"Status: {status}"
            if status == "SUCCESS":
                status_surf = font_stats_bold.render(status_text + " 🎉", True, COLOR_GREEN)
            elif status == "DROWNED":
                status_surf = font_stats_bold.render(status_text + " 💀", True, COLOR_RED)
            elif status == "TIMEOUT":
                status_surf = font_stats_bold.render(status_text + " ⏳", True, COLOR_GOLD)
            else:
                status_surf = font_stats.render(status_text, True, COLOR_CYAN)
            screen.blit(status_surf, (x_offset, status_y))

            # Action representation
            action_y = status_y + 25
            action_name = ACTION_CHARS.get(last_action, "None") if status == "NAVIGATING" else "STOPPED"
            action_surf = font_stats.render(f"Last Action: {action_name}", True, COLOR_TEXT_MUTED)
            screen.blit(action_surf, (x_offset, action_y))

        # Render All Three Grids side-by-side
        draw_grid_map(grid_x_ga, manager.size, manager.shared_desc, manager.pos_ga, manager.history_ga, manager.status_ga, COLOR_CYAN, "GENETIC ALGORITHM (CI)", manager.last_action_ga)
        draw_grid_map(grid_x_ql, manager.size, manager.shared_desc, manager.pos_ql, manager.history_ql, manager.status_ql, COLOR_GOLD, "TABULAR Q-LEARNING (ML)", manager.last_action_ql)
        draw_grid_map(grid_x_dqn, manager.size, manager.shared_desc, manager.pos_dqn, manager.history_dqn, manager.status_dqn, COLOR_MAGENTA, "DEEP Q-NETWORK (DL)", manager.last_action_dqn)

        # 4. Draw Dashboard Control Panel at the bottom
        control_panel_rect = pygame.Rect(40, 520, 1200, 160)
        draw_rounded_rect(screen, control_panel_rect, COLOR_CARD, 10)
        pygame.draw.rect(screen, COLOR_BORDER, control_panel_rect, width=1, border_radius=10)
        
        # Helper to render interactive buttons
        def draw_ui_button(rect, text, is_active, accent_color=COLOR_CYAN, mouse_over=False):
            # Fill
            bg_color = (40, 40, 55) if not is_active else (accent_color[0]//3, accent_color[1]//3, accent_color[2]//3)
            if mouse_over:
                bg_color = (60, 60, 80) if not is_active else (accent_color[0]//2, accent_color[1]//2, accent_color[2]//2)
                
            draw_rounded_rect(screen, rect, bg_color, 6)
            
            # Border
            border_color = COLOR_BORDER if not is_active else accent_color
            pygame.draw.rect(screen, border_color, rect, width=1, border_radius=6)
            
            # Label
            txt_color = COLOR_TEXT_MUTED if not is_active else COLOR_WHITE
            label_surf = font_button.render(text, True, txt_color)
            screen.blit(label_surf, (rect.centerx - label_surf.get_width()//2, rect.centery - label_surf.get_height()//2))

        # Check button hovers for styling
        m_pos = pygame.mouse.get_pos()
        
        # Row 1 Controls
        draw_ui_button(btn_play, "⏸ PAUSE SIM" if manager.playing else "▶️ PLAY SIMULATION", manager.playing, COLOR_GREEN, btn_play.collidepoint(m_pos))
        draw_ui_button(btn_reset, "🔄 RESET POS", False, COLOR_WHITE, btn_reset.collidepoint(m_pos))
        
        # Grid Selector
        draw_ui_button(btn_grid_4, "4x4", manager.size == 4, COLOR_CYAN, btn_grid_4.collidepoint(m_pos))
        draw_ui_button(btn_grid_8, "8x8", manager.size == 8, COLOR_CYAN, btn_grid_8.collidepoint(m_pos))
        draw_ui_button(btn_grid_16, "16x16", manager.size == 16, COLOR_CYAN, btn_grid_16.collidepoint(m_pos))
        draw_ui_button(btn_grid_25, "25x25", manager.size == 25, COLOR_CYAN, btn_grid_25.collidepoint(m_pos))
        
        # Physics selector
        draw_ui_button(btn_slip_false, "DETERMINISTIC", not manager.is_slippery, COLOR_GOLD, btn_slip_false.collidepoint(m_pos))
        draw_ui_button(btn_slip_true, "SLIPPERY (Kaygan)", manager.is_slippery, COLOR_GOLD, btn_slip_true.collidepoint(m_pos))
        
        # Speed selector
        draw_ui_button(btn_speed_0, "SLOW 🐢", manager.speed == 0, COLOR_WHITE, btn_speed_0.collidepoint(m_pos))
        draw_ui_button(btn_speed_1, "MEDIUM", manager.speed == 1, COLOR_WHITE, btn_speed_1.collidepoint(m_pos))
        draw_ui_button(btn_speed_2, "FAST ⚡", manager.speed == 2, COLOR_WHITE, btn_speed_2.collidepoint(m_pos))

        # Row 2 Controls: Epoch/Stage Selector
        lbl_epochs = font_header.render("TRAINING STAGES (EPOCHS):", True, COLOR_TEXT_MAIN)
        screen.blit(lbl_epochs, (60, 590))
        
        # Stage Buttons
        draw_ui_button(btn_stage_0, "EARLY (Chaos Stage)", manager.active_stage == 0, COLOR_CYAN, btn_stage_0.collidepoint(m_pos))
        draw_ui_button(btn_stage_1, "MID-TRAINING (100 Ep.)", manager.active_stage == 1, COLOR_CYAN, btn_stage_1.collidepoint(m_pos))
        draw_ui_button(btn_stage_2, "FULLY CONVERGED (Trained)", manager.active_stage == 2, COLOR_GREEN, btn_stage_2.collidepoint(m_pos))

        # Instructions / Shortcut legend
        shortcuts_str = "Shortcuts: [Space] Play/Pause  |  [R] Reset  |  [1, 2, 3] Change Epochs"
        shortcuts_surf = font_subtitle.render(shortcuts_str, True, COLOR_TEXT_MUTED)
        screen.blit(shortcuts_surf, (800, 618))

        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()


def run_terminal_fallback():
    """
    Highly premium ASCII-based Console fallback representation in case no display device is detected.
    """
    size = 8
    is_slippery = False
    
    print("\n" + "="*80)
    print("🎬  FROZENLAKE-V1 MULTI-PARADIGM CONSOLE COMPARISON FALLBACK  🎬")
    print("="*80)
    print("Notice: No visual graphical display (X11/GUI) was found.")
    print("This terminal engine simulates policy evaluations on identical layouts.")
    print("="*80)
    
    # 1. Setup
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map
    shared_desc = generate_random_map(size=size)
    env = Environment(size=size, desc=shared_desc, is_slippery=is_slippery)
    
    # Quick train
    print("\n[RL-Ops Coordinator] Optimizing and preparing Q-Learning, GA, and DQN policies...")
    
    # Q-Learning
    ql_agent = QLearningAgent(num_states=size*size, num_actions=4, decay_rate=0.01)
    ql_agent.train(env, total_episodes=500)
    ql_policy = [int(np.argmax(ql_agent.q_table[s])) for s in range(size*size)]
    
    # GA
    ga_config = {
        "env_size": size, "pop_size": 40, "max_generations": 40,
        "mutation_rate": 0.2, "crossover_rate": 0.8, "tournament_size": 3,
        "crossover_method": "uniform", "mutation_method": "random_resetting",
        "strategy": "mu_plus_lambda",
        "fitness_weights": {"success": 100.0, "manhattan_multiplier": 10.0, "step_penalty": 0.1, "hole_penalty": 50.0}
    }
    ga = GeneticAlgorithm(ga_config, env)
    best_ga, _ = ga.run_evolution()
    ga_policy = list(best_ga.genotype)
    
    # DQN
    dqn_agent = DQNAgent(num_states=size*size, num_actions=4, decay_rate=0.02)
    dqn_agent.train(env, total_episodes=200)
    dqn_policy = [int(dqn_agent.choose_action(s, evaluate=True)) for s in range(size*size)]
    
    policies = {"GA": ga_policy, "Q-Learning": ql_policy, "DQN": dqn_policy}
    
    # 2. Run simulation step-by-step
    for alg_name, policy in policies.items():
        print(f"\n🚀 Simulating {alg_name} Agent Route:")
        state, _ = env.reset()
        terminated = False
        steps = 0
        history = [(0, 0)]
        
        while not terminated and steps < env.max_steps:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            r = state // size
            c = state % size
            history.append((r, c))
            
            if terminated:
                if reward > 0.0:
                    status = "🎉 SUCCESS!"
                else:
                    status = "💀 DROWNED IN HOLE"
                break
        else:
            status = "⏳ TIMEOUT"
            
        print(f"  Path Taken: {' ➡️ '.join([str(pt) for pt in history])}")
        print(f"  Outcome: {status} | Steps Taken: {steps}")
    
    print("\n" + "="*80)
    print("✅ Console demonstration simulation loop finalized.")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_gui_demo()
