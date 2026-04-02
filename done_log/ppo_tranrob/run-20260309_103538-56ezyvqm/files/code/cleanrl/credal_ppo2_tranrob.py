# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Import functions for epistemic uncertainty calculation
from scipy.optimize import linprog
from scipy.optimize import minimize
import copy

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
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 200000
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
    clip_coef: float = 0.2
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
    credal_lambda: float = 1
    """coefficient for the epistemic uncertainty penalty in the loss"""

    # Transition robustness evaluation arguments
    eval_perturbations: bool = True
    """if toggled, evaluation will use blocked-side transition dynamics"""
    eval_episodes_with_perturbations: int = 10
    """number of evaluation episodes with random blocked sides"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def interval_softmax(logits):
    """
    Compute interval softmax to get lower and upper probability bounds
    logits: tensor of shape (..., 2 * n_actions) where first half are centers and second half are radii
    Returns: tuple of (center_probs, radius_probs, lower_bounds, upper_bounds, intersection_probs)
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


def calculate_reachable_probabilities(q_lower, q_upper):
    """
    Implements Equations 8, 12, and 13 from the CreINN paper.
    Computes reachable probability intervals and intersection probabilities.

    Args:
        q_lower (np.ndarray): Array of lower probability bounds for each class.
        q_upper (np.ndarray): Array of upper probability bounds for each class.

    Returns:
        tuple: (q_reachable_lower, q_reachable_upper, q_int)
    """
    # Ensure inputs are numpy arrays for vectorization
    q_l = np.array(q_lower)
    q_u = np.array(q_upper)
    num_classes = len(q_l)

    # --- Equation 8: Reachable Probability Intervals ---
    # Calculates the tightened, reachable bounds for the credal set.
    # q_star_lower_k = max(q_lower_k, 1 - sum_{j!=k} q_upper_j)
    # q_star_upper_k = min(q_upper_k, 1 - sum_{j!=k} q_lower_j)

    q_star_lower = np.zeros(num_classes)
    q_star_upper = np.zeros(num_classes)

    for k in range(num_classes):
        # Sum of upper/lower bounds excluding the current class k
        sum_upper_jk = np.sum(q_u) - q_u[k]
        sum_lower_jk = np.sum(q_l) - q_l[k]

        q_star_lower[k] = max(q_l[k], 1 - sum_upper_jk)
        q_star_upper[k] = min(q_u[k], 1 - sum_lower_jk)

    # --- Equation 13: Calculate Alpha ---
    # Alpha determines how to distribute the remaining probability mass.
    # alpha = (1 - sum(q_star_lower)) / sum(q_star_upper - q_star_lower)

    sum_q_star_lower = np.sum(q_star_lower)
    sum_diff = np.sum(q_star_upper - q_star_lower)

    # Avoid division by zero if intervals are point-wise
    if sum_diff == 0:
        alpha = 0.0
    else:
        alpha = (1 - sum_q_star_lower) / sum_diff

    # --- Equation 12: Intersection Probability (q_int) ---
    # Computes the single probability distribution for class prediction.
    # q_int_k = q_star_lower_k + alpha * (q_star_upper_k - q_star_lower_k)

    q_int = q_star_lower + alpha * (q_star_upper - q_star_lower)

    return q_star_lower, q_star_upper, q_int


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
    # from scipy.optimize import maximize
    # Objective function
    def objective_min(x):
        return -np.sum(x*np.log2(np.clip(x, 1e-12, np.max(x))))
    def objective_max(x):
        return np.sum(x*np.log2(np.clip(x, 1e-12, np.max(x))))
    # Constraint function
    def constraint(x):
        return np.sum(x) - 1

    problem = {'type': 'eq', 'fun': constraint}
    sol1 = minimize(objective_min, x0, method='SLSQP', bounds=bounds, constraints=problem)
    sol2 = minimize(objective_max, x0, method='SLSQP', bounds=bounds, constraints=problem)
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
            print(str(i), ": No valid solution found.")
        else:
           min, max = min_max_entropy(reachable_lower[i, :], reachable_upper[i, :], x0, delta=1e-5)
           if min is None or max is None:
              print(str(i), ": No valid solution found.")
           else:
              mins.append(min)
              maxs.append(max)
    u_pred = {
        'Hu': np.array(maxs),
        'EU': np.array(maxs) - np.array(mins),
        'Hl': np.array(mins)
    }
    return u_pred


class AcrobotBlockedSideEnv(gym.Wrapper):
    """
    Wrapper for Acrobot-v1 that blocks one side (left or right) halfway up.
    
    In Acrobot-v1, the goal is to swing the tip to reach a target height (-1.0).
    This wrapper blocks one side by adding a constraint: if the first joint angle
    goes into the blocked half-plane beyond a threshold, the episode terminates
    with a large negative reward, forcing the agent to reach the target on the 
    unblocked side.
    
    Args:
        env: The base Acrobot-v1 environment
        blocked_side: Which side to block ('left' or 'right'), or None for random
        perturbation_seed: Optional seed for reproducible side selection (None for random)
    """
    def __init__(self, env, blocked_side=None, perturbation_seed=None):
        super().__init__(env)
        self.blocked_side = blocked_side
        self.perturbation_seed = perturbation_seed
        self.current_blocked_side = None
        # Default target height in Acrobot-v1 is -1.0
        self.base_target_height = -1.0
        # Block angle threshold: pi/2 radians (90 degrees) from vertical
        # This blocks halfway up on one side
        self.block_angle_threshold = np.pi / 2
        
    def reset(self, seed=None, **kwargs):
        # Set seed for side selection if provided
        if self.perturbation_seed is not None:
            np.random.seed(self.perturbation_seed)
        
        # Randomly select which side to block if not specified
        if self.blocked_side is None:
            self.current_blocked_side = np.random.choice(['left', 'right'])
        else:
            self.current_blocked_side = self.blocked_side
        
        return super().reset(seed=seed, **kwargs)
    
    def step(self, action):
        # Take the action in the base environment
        obs, reward, terminated, truncated, info = super().step(action)
        
        # obs = [cos(theta1), sin(theta1), cos(theta2), sin(theta2), angular_vel1, angular_vel2]
        cos_theta1 = obs[0]
        sin_theta1 = obs[1]
        
        # Calculate theta1 (first joint angle) from cos and sin
        # theta1 = 0 means pointing straight down
        # theta1 = pi means pointing straight up
        theta1 = np.arctan2(sin_theta1, cos_theta1)
        
        # Calculate tip y-coordinate for reward checking
        cos_theta2 = obs[2]
        sin_theta2 = obs[3]
        cos_theta1_plus_theta2 = cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2
        tip_y = -cos_theta1 - cos_theta1_plus_theta2
        
        # Check if agent reached target height
        target_reached = tip_y >= self.base_target_height
        
        # Check if agent violated the blocked side constraint
        # Left side blocked: theta1 should not go positive beyond threshold (right half-plane is safe)
        # Right side blocked: theta1 should not go negative beyond threshold (left half-plane is safe)
        side_violated = False
        if self.current_blocked_side == 'left':
            # Left side is blocked, agent must swing up on the right side
            # theta1 > 0 means leaning right, theta1 < 0 means leaning left
            # Block if theta1 goes too far left (negative) beyond -threshold
            if theta1 < -self.block_angle_threshold:
                side_violated = True
        elif self.current_blocked_side == 'right':
            # Right side is blocked, agent must swing up on the left side
            # Block if theta1 goes too far right (positive) beyond threshold
            if theta1 > self.block_angle_threshold:
                side_violated = True
        
        # Modify reward and termination based on constraint violation
        if side_violated:
            reward = -10.0  # Large penalty for hitting the blocked side
            terminated = True
            info['blocked_side_violation'] = True
            info['blocked_side'] = self.current_blocked_side
        elif target_reached:
            reward = 0.0
            terminated = True
            info['target_reached'] = True
            info['blocked_side'] = self.current_blocked_side
        else:
            reward = -1.0
            info['blocked_side'] = self.current_blocked_side
        
        return obs, reward, terminated, truncated, info


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Output 2 * number of actions for centers and radii
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 2 * envs.single_action_space.n), std=0.01),  # Double output for center/radius
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)

        # Compute center, radius, lower, upper, reachable lower, reachable upper, and intersection probabilities
        center_probs, radius_probs, lower_probs, upper_probs, reachable_lower, reachable_upper, intersection_probs = interval_softmax(logits)

        # Use intersection probabilities for sampling and computing log probabilities
        probs = Categorical(probs=intersection_probs)

        if action is None:
            action = probs.sample()

        # Return action, logprob using intersection probs, entropy, and value
        # Also return the additional probability information for logging/debugging
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), center_probs, radius_probs, lower_probs, upper_probs, reachable_lower, reachable_upper, intersection_probs


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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

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

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
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
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

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

        # Flatten the stored probability tensors for reachable probability calculation
        b_center_probs = center_probs_storage.reshape((-1, envs.single_action_space.n))
        b_radius_probs = radius_probs_storage.reshape((-1, envs.single_action_space.n))
        b_lower_probs = lower_probs_storage.reshape((-1, envs.single_action_space.n))
        b_upper_probs = upper_probs_storage.reshape((-1, envs.single_action_space.n))
        b_reachable_lower = reachable_lower_storage.reshape((-1, envs.single_action_space.n))
        b_reachable_upper = reachable_upper_storage.reshape((-1, envs.single_action_space.n))
        b_intersection_probs = intersection_probs_storage.reshape((-1, envs.single_action_space.n))

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

                # Policy loss using intersection probabilities (already used in forward pass)
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

                # Calculate epistemic uncertainty penalty for the current mini-batch
                # Use reachable probabilities for uncertainty calculation
                mb_reachable_lower_np = reachable_lower_mb.detach().cpu().numpy()
                mb_reachable_upper_np = reachable_upper_mb.detach().cpu().numpy()

                # Calculate epistemic uncertainty using the min_max_entropy_calculation function
                try:
                    mb_uncertainty_results = min_max_entropy_calculation(mb_reachable_lower_np, mb_reachable_upper_np)
                    mb_epistemic_uncertainty = np.mean(mb_uncertainty_results['EU']) if len(mb_uncertainty_results['EU']) > 0 else 0.0
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
        b_intersection_probs = intersection_probs_storage.reshape((-1, envs.single_action_space.n))

        # Calculate epistemic uncertainty using reachable probabilities
        try:
            b_reachable_lower_np = b_reachable_lower.cpu().numpy()
            b_reachable_upper_np = b_reachable_upper.cpu().numpy()
            uncertainty_results = min_max_entropy_calculation(b_reachable_lower_np, b_reachable_upper_np)
            mean_epistemic_uncertainty = np.mean(uncertainty_results['EU']) if len(uncertainty_results['EU']) > 0 else 0.0
            mean_total_uncertainty = np.mean(uncertainty_results['Hu']) if len(uncertainty_results['Hu']) > 0 else 0.0
            mean_lower_uncertainty = np.mean(uncertainty_results['Hl']) if len(uncertainty_results['Hl']) > 0 else 0.0
        except Exception as e:
            print(f"Error calculating epistemic uncertainty: {e}")
            mean_epistemic_uncertainty = 0.0
            mean_total_uncertainty = 0.0
            mean_lower_uncertainty = 0.0

        # Log epistemic uncertainty measures
        writer.add_scalar("uncertainty/epistemic_mean", mean_epistemic_uncertainty, global_step)
        writer.add_scalar("uncertainty/total_mean", mean_total_uncertainty, global_step)
        writer.add_scalar("uncertainty/lower_mean", mean_lower_uncertainty, global_step)

        # Evaluate with random blocked sides if enabled (for Acrobot-v1)
        if args.eval_perturbations and "Acrobot" in args.env_id:
            eval_rewards = []
            blocked_sides_used = []
            target_reached_count = 0
            side_violation_count = 0
            
            # Deepcopy the agent so the evaluation doesn't interfere with training
            eval_agent = copy.deepcopy(agent)
            eval_agent.eval()
            
            for ep in range(args.eval_episodes_with_perturbations):
                # Create blocked-side environment for this episode
                perturbation_seed = args.seed + ep + iteration * 1000  # Different seed each episode
                eval_env = AcrobotBlockedSideEnv(
                    gym.make(args.env_id),
                    blocked_side=None,  # Random side selection
                    perturbation_seed=perturbation_seed
                )
                
                eval_obs, _ = eval_env.reset()
                eval_obs = torch.tensor(eval_obs, dtype=torch.float32).to(device).unsqueeze(0)
                eval_done = False
                eval_total_reward = 0
                
                # Track blocked side for logging
                blocked_sides_used.append(eval_env.current_blocked_side)
                
                while not eval_done:
                    with torch.no_grad():
                        eval_action, _, _, _, _, _, _, _, _, _, _ = eval_agent.get_action_and_value(eval_obs)
                    
                    eval_next_obs, eval_reward, eval_terminated, eval_truncated, eval_info = eval_env.step(eval_action.item())
                    eval_obs = torch.tensor(eval_next_obs, dtype=torch.float32).to(device).unsqueeze(0)
                    eval_total_reward += eval_reward
                    eval_done = eval_terminated or eval_truncated
                    
                    # Track success/failure types
                    if eval_info.get('target_reached', False):
                        target_reached_count += 1
                    if eval_info.get('blocked_side_violation', False):
                        side_violation_count += 1
                
                eval_rewards.append(eval_total_reward)
                eval_env.close()
            
            eval_mean_reward = np.average(eval_rewards)
            eval_std_reward = np.std(eval_rewards)
            
            # Calculate statistics per blocked side
            left_side_rewards = [r for r, s in zip(eval_rewards, blocked_sides_used) if s == 'left']
            right_side_rewards = [r for r, s in zip(eval_rewards, blocked_sides_used) if s == 'right']
            left_side_count = len(left_side_rewards)
            right_side_count = len(right_side_rewards)
            
            # Log evaluation metrics
            writer.add_scalar("evaluation/mean_reward_vs_steps", eval_mean_reward, global_step)
            writer.add_scalar("evaluation/mean_reward_vs_time", eval_mean_reward, time.time() - start_time)
            writer.add_scalar("evaluation/std_reward", eval_std_reward, global_step)
            
            # Log blocked side statistics
            writer.add_scalar("evaluation/target_reached_count", target_reached_count, global_step)
            writer.add_scalar("evaluation/side_violation_count", side_violation_count, global_step)
            writer.add_scalar("evaluation/left_side_episodes", left_side_count, global_step)
            writer.add_scalar("evaluation/right_side_episodes", right_side_count, global_step)
            
            if left_side_count > 0:
                writer.add_scalar("evaluation/mean_reward_left_side", np.mean(left_side_rewards), global_step)
            if right_side_count > 0:
                writer.add_scalar("evaluation/mean_reward_right_side", np.mean(right_side_rewards), global_step)
            
            print(f"global_step={global_step}, eval_mean_reward={eval_mean_reward:.2f}, "
                  f"eval_std_reward={eval_std_reward:.2f}, target_reached={target_reached_count}/{len(eval_rewards)}, "
                  f"side_violations={side_violation_count}/{len(eval_rewards)}")
        else:
            # Standard evaluation without perturbations
            eval_env = gym.make(args.env_id)
            eval_rewards = []
            
            # Deepcopy the agent so the evaluation doesn't interfere with training
            eval_agent = copy.deepcopy(agent)
            eval_agent.eval()
            
            for _ in range(10):  # Run 10 greedy evaluations
                eval_obs, _ = eval_env.reset()
                eval_obs = torch.tensor(eval_obs, dtype=torch.float32).to(device).unsqueeze(0)  # Add batch dim
                eval_done = False
                eval_total_reward = 0
                
                while not eval_done:
                    with torch.no_grad():
                        eval_action, _, _, _, _, _, _, _, _, _, _ = eval_agent.get_action_and_value(eval_obs)
                    
                    eval_next_obs, eval_reward, eval_terminated, eval_truncated, _ = eval_env.step(eval_action.item())
                    eval_obs = torch.tensor(eval_next_obs, dtype=torch.float32).to(device).unsqueeze(0)
                    eval_total_reward += eval_reward
                    eval_done = eval_terminated or eval_truncated
                
                eval_rewards.append(eval_total_reward)
            eval_mean_reward = np.average(eval_rewards)
            writer.add_scalar("evaluation/mean_reward_vs_steps", eval_mean_reward, global_step)
            writer.add_scalar("evaluation/mean_reward_vs_time", eval_mean_reward, time.time() - start_time)
            
            # Optional: Close the eval environment after use
            eval_env.close()

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
