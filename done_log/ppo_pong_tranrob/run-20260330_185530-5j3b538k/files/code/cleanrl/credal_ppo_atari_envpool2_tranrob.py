# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import copy

# Import functions for epistemic uncertainty calculation
from scipy.optimize import linprog
from scipy.optimize import minimize


class ObstacleWallEnv(gym.Wrapper):
    """
    Wrapper that adds a random obstacle wall at the center of the screen for Pong.
    If the ball hits the obstacle wall, the player immediately loses the point and
    a new round starts (simulated by returning done=True and negative reward).

    This wrapper:
    1. Adds a visual white wall obstacle to the observation
    2. Tracks ball position via ALE RAM
    3. Detects when ball hits the obstacle
    4. Forces immediate point loss by returning done=True and -1 reward

    :param env: Environment to wrap
    :param obstacle_height: Height of the obstacle wall in pixels
    :param obstacle_width: Width of the obstacle wall in pixels
    :param seed: Random seed for obstacle position
    """

    def __init__(
        self,
        env,
        obstacle_height: int = 20,  # Wall length reduced to 20 to lower collision rate
        obstacle_width: int = 4,
        seed: int = None,
    ) -> None:
        super().__init__(env)
        self.obstacle_height = obstacle_height
        self.obstacle_width = obstacle_width
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Obstacle position (set at reset)
        self.wall_y_start = None
        self.wall_y_end = None
        self.wall_x_center = None

        # Ball tracking for collision detection
        self.prev_ball_x = None
        self.prev_ball_y = None
        self.ball_dir_x = 1  # 1 = right (toward player), -1 = left (away)
        self.ball_dir_y = 1

        # Original Atari resolution
        self.orig_height = 210
        self.orig_width = 160

        # Collision state
        self.collision_occurred = False

        # Debug: track detected ball positions
        self.debug_ball_positions = []

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        # Random vertical position for the wall - can be anywhere on screen
        # Wall must fit within screen: wall_y_start + obstacle_height <= orig_height
        max_y_start = self.orig_height - self.obstacle_height
        self.wall_y_start = self.rng.randint(0, max_y_start + 1)
        self.wall_y_end = self.wall_y_start + self.obstacle_height
        self.wall_x_center = self.orig_width // 2

        # Reset ball tracking
        self.prev_ball_x = None
        self.prev_ball_y = None
        self.ball_dir_x = 1
        self.ball_dir_y = 1
        self.collision_occurred = False
        self.debug_ball_positions = []

        # Apply obstacle to initial observation
        obs = self._add_obstacle_to_obs(obs)

        return obs

    def step(self, action):
        # Get ball position from RAM
        ball_x, ball_y = self._get_ball_position()
        
        # Update ball direction
        if self.prev_ball_x is not None:
            self.ball_dir_x = 1 if ball_x > self.prev_ball_x else -1
            self.ball_dir_y = 1 if ball_y > self.prev_ball_y else -1
        
        # Check for obstacle collision BEFORE executing the step
        collision_detected = self._check_obstacle_collision(ball_x, ball_y)
        
        self.prev_ball_x = ball_x
        self.prev_ball_y = ball_y
        
        # Execute the step
        next_obs, reward, terminated, info = self.env.step(action)
        
        # Add visual obstacle to observation
        next_obs = self._add_obstacle_to_obs(next_obs)
        
        # If collision detected, force immediate point loss and end the episode
        if collision_detected:
            # Return negative reward (point loss), done=True to end episode
            # This simulates the opponent scoring when ball hits the obstacle
            info['wall_collision'] = True
            return next_obs, -1.0, True, info
        
        return next_obs, reward, terminated, info

    def _get_ball_position(self):
        """Get ball position from ALE RAM via envpool's get_ram() method."""
        try:
            # Envpool provides RAM access through 'get_ram' or info dict
            ram = self.env.get_ram() if hasattr(self.env, 'get_ram') else None
            if ram is not None:
                # Pong RAM addresses: ball_x=0x08, ball_y=0x09
                return int(ram[0x08]), int(ram[0x09])
        except (AttributeError, IndexError, KeyError):
            pass
        return 80, 105  # Default center position if RAM access fails

    def _check_obstacle_collision(self, ball_x, ball_y):
        """
        Check if ball hits the obstacle wall.
        Returns True if collision detected.
        
        The obstacle is a vertical wall at the center of the screen.
        When the ball hits it, the player loses the point (as if they missed the ball).
        """
        # Obstacle boundaries
        wall_x_left = self.wall_x_center - self.obstacle_width // 2
        wall_x_right = self.wall_x_center + self.obstacle_width // 2
        
        # Check if ball is within obstacle's vertical range
        in_vertical_range = self.wall_y_start <= ball_y <= self.wall_y_end
        
        # Check if ball is at the wall's horizontal position
        at_wall_horizontally = wall_x_left <= ball_x <= wall_x_right
        
        # Collision: ball is at wall position within vertical range
        # This means the ball hit the obstacle instead of passing through the gap
        if in_vertical_range and at_wall_horizontally:
            return True
        
        return False

    def _add_obstacle_to_obs(self, obs):
        """Add visual obstacle wall to the observation."""
        # Handle batched observations from envpool
        single_obs = False
        if len(obs.shape) == 3:
            obs = obs[np.newaxis, ...]
            single_obs = True

        num_envs = obs.shape[0]
        height, width, channels = obs.shape[1], obs.shape[2], obs.shape[3]

        modified_obs = obs.copy()

        # Scale obstacle position to observation resolution
        scale_y = height / self.orig_height
        scale_x = width / self.orig_width
        
        wall_y_start_scaled = int(self.wall_y_start * scale_y)
        wall_y_end_scaled = int(self.wall_y_end * scale_y)
        wall_x_scaled = int(self.wall_x_center * scale_x)
        wall_width_scaled = max(1, int(self.obstacle_width * scale_x))

        # Draw the obstacle wall (white color = 255)
        for i in range(num_envs):
            modified_obs[i,
                         wall_y_start_scaled:wall_y_end_scaled,
                         wall_x_scaled - wall_width_scaled // 2:wall_x_scaled + wall_width_scaled // 2,
                         :] = 255

        if single_obs:
            modified_obs = modified_obs[0]

        return modified_obs


def add_random_obstacle_wall(obs, env_id, obstacle_height=20, obstacle_width=2, seed=None):
    """Add a random obstacle wall at the center of the screen for robustness testing.

    Args:
        obs: Observation array of shape (num_envs, height, width, channels) or (height, width, channels)
        env_id: Environment ID (e.g., "Pong-v5")
        obstacle_height: Height of the obstacle wall in pixels
        obstacle_width: Width of the obstacle wall in pixels
        seed: Random seed for obstacle position (if None, uses random position each time)

    Returns:
        Modified observation with obstacle wall
    """
    # Only apply to Pong environment
    if "Pong" not in env_id:
        return obs

    # Handle both single obs and batched obs
    single_obs = False
    if len(obs.shape) == 3:
        obs = obs[np.newaxis, ...]
        single_obs = True

    num_envs = obs.shape[0]
    height, width, channels = obs.shape[1], obs.shape[2], obs.shape[3]

    # Create a copy to avoid modifying the original
    modified_obs = obs.copy()

    # Generate random vertical position for the wall at center
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # Calculate center region and random vertical offset
    center_x = width // 2
    max_y_offset = (height - obstacle_height) // 2

    # Random vertical start position for the wall
    y_start = rng.randint(0, max_y_offset + 1)

    # Draw the obstacle wall (white color = 255 for Atari grayscale)
    for i in range(num_envs):
        # Draw vertical wall at center
        modified_obs[i,
                     y_start:y_start + obstacle_height,
                     center_x - obstacle_width // 2:center_x + obstacle_width // 2,
                     :] = 255

    if single_obs:
        modified_obs = modified_obs[0]

    return modified_obs


def get_gym_env_id(envpool_env_id):
    """Convert envpool environment name to gym ALE format.

    Envpool uses: Pong-v5, Breakout-v5
    Gym (ALE) uses: ALE/Pong-v5, ALE/Breakout-v5
    """
    # Check if already in ALE format
    if envpool_env_id.startswith("ALE/"):
        return envpool_env_id
    # Convert envpool format to ALE format
    return f"ALE/{envpool_env_id}"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 2500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 10
    """the number of parallel game environments"""
    num_steps: int = 500
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    credal_lambda: float = 1.0
    """coefficient for the epistemic uncertainty penalty in the loss"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def calculate_reachable_probabilities_torch(q_lower, q_upper):
    """
    Differentiable PyTorch implementation of Equations 8, 12, and 13 from the CreINN paper.
    Computes reachable probability intervals and intersection probabilities.

    Args:
        q_lower (torch.Tensor): Tensor of lower probability bounds, shape (..., num_classes)
        q_upper (torch.Tensor): Tensor of upper probability bounds, shape (..., num_classes)

    Returns:
        tuple: (q_reachable_lower, q_reachable_upper, q_int)
    """
    # Get the number of classes
    num_classes = q_lower.shape[-1]

    # Keep original shape for batch operations
    original_shape = q_lower.shape
    if len(original_shape) == 1:
        # Add batch dimension if needed
        q_l = q_lower.unsqueeze(0)
        q_u = q_upper.unsqueeze(0)
        squeeze_result = True
    else:
        q_l = q_lower
        q_u = q_upper
        squeeze_result = False

    batch_size = q_l.shape[0]

    # --- Equation 8: Reachable Probability Intervals ---
    # q_star_lower_k = max(q_lower_k, 1 - sum_{j!=k} q_upper_j)
    # q_star_upper_k = min(q_upper_k, 1 - sum_{j!=k} q_lower_j)

    # Calculate sum of all upper and lower probabilities for each sample
    sum_upper = torch.sum(q_u, dim=-1, keepdim=True)  # Shape: (batch, 1)
    sum_lower = torch.sum(q_l, dim=-1, keepdim=True)  # Shape: (batch, 1)

    # Calculate sum excluding current class k: sum_{j!=k} = sum_total - value_k
    # For all classes at once using broadcasting
    sum_upper_jk = sum_upper - q_u  # Shape: (batch, num_classes)
    sum_lower_jk = sum_lower - q_l  # Shape: (batch, num_classes)

    # Calculate reachable lower and upper bounds
    q_star_lower = torch.maximum(q_l, 1.0 - sum_upper_jk)
    q_star_upper = torch.minimum(q_u, 1.0 - sum_lower_jk)

    # --- Equation 13: Calculate Alpha ---
    # alpha = (1 - sum(q_star_lower)) / sum(q_star_upper - q_star_lower)

    sum_q_star_lower = torch.sum(q_star_lower, dim=-1, keepdim=True)  # Shape: (batch, 1)
    sum_diff = torch.sum(q_star_upper - q_star_lower, dim=-1, keepdim=True)  # Shape: (batch, 1)

    # Avoid division by zero - use a small epsilon
    epsilon = 1e-10
    alpha = (1.0 - sum_q_star_lower) / (sum_diff + epsilon)

    # Clamp alpha to [0, 1] range for numerical stability
    alpha = torch.clamp(alpha, 0.0, 1.0)

    # --- Equation 12: Intersection Probability (q_int) ---
    # q_int_k = q_star_lower_k + alpha * (q_star_upper_k - q_star_lower_k)

    q_int = q_star_lower + alpha * (q_star_upper - q_star_lower)

    # Squeeze if we added a batch dimension
    if squeeze_result:
        q_star_lower = q_star_lower.squeeze(0)
        q_star_upper = q_star_upper.squeeze(0)
        q_int = q_int.squeeze(0)

    return q_star_lower, q_star_upper, q_int


def interval_softmax(logits):
    """
    Compute interval softmax to get lower and upper probability bounds
    logits: tensor of shape (..., 2 * n_actions) where first half are centers and second half are radii
    Returns: tuple of (center_probs, radius_probs, lower_bounds, upper_bounds, reachable_lower, reachable_upper, intersection_probs)
    """
    n_actions = logits.shape[-1] // 2

    # Split logits into center and radius parts
    centers = logits[..., :n_actions]
    raw_radii = logits[..., n_actions:]

    # Ensure non-negative radii using softplus
    radii = torch.nn.functional.softplus(raw_radii)

    # Compute probabilities using softmax
    center_exp = torch.exp(centers)
    center_sum = torch.sum(center_exp, dim=-1, keepdim=True)

    # Calculate lower bound probabilities
    lower_exp = torch.exp(centers - radii)
    lower_probs = lower_exp / (center_sum - center_exp + lower_exp)

    # Calculate upper bound probabilities
    upper_exp = torch.exp(centers + radii)
    upper_probs = upper_exp / (center_sum - center_exp + upper_exp)

    # Calculate center probabilities (standard softmax)
    center_probs = torch.softmax(centers, dim=-1)

    # Calculate reachable probabilities and intersection probabilities (differentiable)
    reachable_lower, reachable_upper, intersection_probs = calculate_reachable_probabilities_torch(lower_probs, upper_probs)

    # Return center, radius, lower, upper, reachable lower, reachable upper, and intersection probabilities
    return center_probs, radii, lower_probs, upper_probs, reachable_lower, reachable_upper, intersection_probs


def search_values(l, u, delta=1e-5):
    n = len(l)
    c = np.zeros(n)
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    u_clip = np.clip(u, delta, np.max(u))

    bounds = [(l[i], u_clip[i]) for i in range(n)]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    if res.success:
        return res.x
    else:
        return None


def min_max_entropy(l, u, x0, delta=1e-5):
    n = len(l)
    u_clip = np.clip(u, delta, np.max(u))

    bounds = [(l[i], u_clip[i]) for i in range(n)]
    # Objective function
    def objective_min(x):
        return -np.sum(x * np.log2(np.clip(x, 1e-12, np.max(x))))

    def objective_max(x):
        return np.sum(x * np.log2(np.clip(x, 1e-12, np.max(x))))

    # Constraint function
    def constraint(x):
        return np.sum(x) - 1

    problem = {"type": "eq", "fun": constraint}
    sol1 = minimize(objective_min, x0, method="SLSQP", bounds=bounds, constraints=problem)
    sol2 = minimize(objective_max, x0, method="SLSQP", bounds=bounds, constraints=problem)
    return sol1.fun, -sol2.fun


def min_max_entropy_calculation(reachable_lower, reachable_upper):
    """
    Calculate min-max entropy using reachable probability intervals.

    Args:
        reachable_lower (np.ndarray): Array of reachable lower probability bounds.
        reachable_upper (np.ndarray): Array of reachable upper probability bounds.

    Returns:
    prediction uncertainty
    """
    # Compute the uncertainty measures
    mins = list()
    maxs = list()
    for i in range(reachable_lower.shape[0]):
        x0 = search_values(reachable_lower[i, :], reachable_upper[i, :])
        if x0 is None:
            print(str(i) + ": No valid solution found.")
        else:
            min, max = min_max_entropy(reachable_lower[i, :], reachable_upper[i, :], x0, delta=1e-5)
            if min is None or max is None:
                print(str(i) + ": No valid solution found.")
            else:
                mins.append(min)
                maxs.append(max)
    u_pred = {
        "Hu": np.array(maxs),
        "EU": np.array(maxs) - np.array(mins),
        "Hl": np.array(mins),
    }
    return u_pred


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        # Output 2 * number of actions for centers and radii
        self.actor = layer_init(nn.Linear(512, 2 * envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)

        # Compute center, radius, lower, upper, reachable lower, reachable upper, and intersection probabilities
        center_probs, radius_probs, lower_probs, upper_probs, reachable_lower, reachable_upper, intersection_probs = interval_softmax(logits)

        # Use intersection probabilities for sampling and computing log probabilities
        probs = Categorical(probs=intersection_probs)

        if action is None:
            action = probs.sample()

        # Return action, logprob using intersection probs, entropy, and value
        # Also return the additional probability information for logging/debugging
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
            center_probs,
            radius_probs,
            lower_probs,
            upper_probs,
            reachable_lower,
            reachable_upper,
            intersection_probs,
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Additional storage for tracking interval probabilities
    center_probs_storage = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
    radius_probs_storage = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
    lower_probs_storage = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
    upper_probs_storage = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
    reachable_lower_storage = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
    reachable_upper_storage = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
    intersection_probs_storage = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)

    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, center_probs, radius_probs, lower_probs, upper_probs, reachable_lower, reachable_upper, intersection_probs = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Store the interval probability information
            center_probs_storage[step] = center_probs
            radius_probs_storage[step] = radius_probs
            lower_probs_storage[step] = lower_probs
            upper_probs_storage[step] = upper_probs
            reachable_lower_storage[step] = reachable_lower
            reachable_upper_storage[step] = reachable_upper
            intersection_probs_storage[step] = intersection_probs

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            for idx, d in enumerate(next_done):
                if d and info["lives"][idx] == 0:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, center_probs_mb, radius_probs_mb, lower_probs_mb, upper_probs_mb, reachable_lower_mb, reachable_upper_mb, intersection_probs_mb = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                # Calculate epistemic uncertainty penalty for the current mini-batch using reachable probabilities
                mb_reachable_lower_np = reachable_lower_mb.detach().cpu().numpy()
                mb_reachable_upper_np = reachable_upper_mb.detach().cpu().numpy()

                # Calculate epistemic uncertainty using the min_max_entropy_calculation function
                try:
                    mb_uncertainty_results = min_max_entropy_calculation(mb_reachable_lower_np, mb_reachable_upper_np)
                    mb_epistemic_uncertainty = np.mean(mb_uncertainty_results["EU"]) if len(mb_uncertainty_results["EU"]) > 0 else 0.0
                    epistemic_penalty = args.credal_lambda * torch.tensor(mb_epistemic_uncertainty, dtype=torch.float32, device=device)
                except Exception as e:
                    print(f"Error calculating epistemic uncertainty for mini-batch: {e}")
                    epistemic_penalty = torch.tensor(0.0, dtype=torch.float32, device=device)

                # Total loss includes the epistemic uncertainty penalty
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + epistemic_penalty

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Process and log interval probability statistics
        # Flatten the stored probability tensors for analysis
        b_center_probs = center_probs_storage.reshape((-1, envs.single_action_space.n))
        b_radius_probs = radius_probs_storage.reshape((-1, envs.single_action_space.n))
        b_lower_probs = lower_probs_storage.reshape((-1, envs.single_action_space.n))
        b_upper_probs = upper_probs_storage.reshape((-1, envs.single_action_space.n))
        b_reachable_lower = reachable_lower_storage.reshape((-1, envs.single_action_space.n))
        b_reachable_upper = reachable_upper_storage.reshape((-1, envs.single_action_space.n))

        # Calculate epistemic uncertainty using reachable probabilities
        try:
            b_reachable_lower_np = b_reachable_lower.cpu().numpy()
            b_reachable_upper_np = b_reachable_upper.cpu().numpy()
            uncertainty_results = min_max_entropy_calculation(b_reachable_lower_np, b_reachable_upper_np)
            mean_epistemic_uncertainty = np.mean(uncertainty_results["EU"]) if len(uncertainty_results["EU"]) > 0 else 0.0
            mean_total_uncertainty = np.mean(uncertainty_results["Hu"]) if len(uncertainty_results["Hu"]) > 0 else 0.0
            mean_lower_uncertainty = np.mean(uncertainty_results["Hl"]) if len(uncertainty_results["Hl"]) > 0 else 0.0
        except Exception as e:
            print(f"Error calculating epistemic uncertainty: {e}")
            mean_epistemic_uncertainty = 0.0
            mean_total_uncertainty = 0.0
            mean_lower_uncertainty = 0.0

        # Log epistemic uncertainty measures
        writer.add_scalar("uncertainty/epistemic_mean", mean_epistemic_uncertainty, global_step)
        writer.add_scalar("uncertainty/total_mean", mean_total_uncertainty, global_step)
        writer.add_scalar("uncertainty/lower_mean", mean_lower_uncertainty, global_step)

        # Evaluate using greedy actors with envpool (single env, no episodic life, no reward clip)
        # For Pong-v5, use ObstacleWallEnv wrapper which adds obstacle wall that affects ball physics
        # (ball hitting wall causes immediate point loss)
        eval_rewards = []
        wall_collision_count = 0  # Track wall collisions during evaluation
        wall_center_y_positions = []  # Track wall center y positions for logging

        # Deepcopy the agent so the evaluation doesn't interfere with training
        eval_agent = copy.deepcopy(agent)
        eval_agent.eval()

        # Use envpool for evaluation with full episode settings (episodic_life=False, reward_clip=False)
        eval_env = envpool.make(
            args.env_id,
            env_type="gym",
            num_envs=1,
            episodic_life=False,  # Full episodes for evaluation
            reward_clip=False,    # No reward clipping for evaluation
            seed=args.seed + 1,   # Different seed for evaluation
        )

        # Wrap with ObstacleWallEnv for Pong to add obstacle with real reward impact
        if "Pong" in args.env_id:
            eval_env = ObstacleWallEnv(eval_env, obstacle_height=20, obstacle_width=4, seed=None)

        for eval_idx in range(10):  # Run 10 greedy evaluations
            # Re-seed the wrapper for each episode to get different wall positions
            # Use iteration-dependent seed so wall positions vary across training
            if "Pong" in args.env_id:
                eval_env.rng = np.random.RandomState(args.seed + 100 + eval_idx + iteration * 1000)
            eval_obs_raw = eval_env.reset()
            eval_obs = torch.Tensor(eval_obs_raw).to(device)
            eval_done = False
            eval_total_reward = 0

            # Log wall center y position for this episode
            if "Pong" in args.env_id:
                wall_center_y = (eval_env.wall_y_start + eval_env.wall_y_end) / 2
                wall_center_y_positions.append(wall_center_y)
                # Debug: print wall info for first iteration
                if iteration == 1:
                    print(f"Eval ep {eval_idx}: wall_y=[{eval_env.wall_y_start}, {eval_env.wall_y_end}], center_y={wall_center_y}")

            while not eval_done:
                with torch.no_grad():
                    eval_action, _, _, _, _, _, _, _, _, _, _ = eval_agent.get_action_and_value(eval_obs)

                eval_next_obs_raw, eval_reward, eval_terminated, eval_info = eval_env.step(eval_action.cpu().numpy())
                eval_obs = torch.Tensor(eval_next_obs_raw).to(device)
                eval_total_reward += float(eval_reward[0]) if isinstance(eval_reward, (list, np.ndarray)) else float(eval_reward)
                eval_done = eval_terminated

                # Check for wall collision in info dict
                if eval_info.get('wall_collision', False):
                    wall_collision_count += 1

            # Debug: print ball positions for first episode
            if iteration == 1 and eval_idx == 0 and "Pong" in args.env_id:
                positions = eval_env.debug_ball_positions[:10] if hasattr(eval_env, 'debug_ball_positions') else []
                print(f"  Ball positions (first 10): {positions}")
                print(f"  Total steps: {len(eval_env.debug_ball_positions) if hasattr(eval_env, 'debug_ball_positions') else 'N/A'}")

            eval_rewards.append(eval_total_reward)

        eval_env.close()
        eval_mean_reward = np.average(eval_rewards)
        writer.add_scalar("evaluation/mean_reward_vs_steps", eval_mean_reward, global_step)
        writer.add_scalar("evaluation/mean_reward_vs_time", eval_mean_reward, time.time() - start_time)

        # Log wall collision metrics for Pong
        if "Pong" in args.env_id:
            writer.add_scalar("evaluation/wall_collision_count", wall_collision_count, global_step)
            writer.add_scalar("evaluation/wall_collision_rate", wall_collision_count / 10, global_step)
            # Log wall center y position (average across all evaluation episodes)
            if len(wall_center_y_positions) > 0:
                wall_center_y_mean = np.mean(wall_center_y_positions)
                writer.add_scalar("evaluation/wall_center_y", wall_center_y_mean, global_step)
                # Also log to wandb directly if tracking is enabled
                if args.track:
                    wandb.log({"evaluation/wall_center_y": wall_center_y_mean}, step=global_step)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
