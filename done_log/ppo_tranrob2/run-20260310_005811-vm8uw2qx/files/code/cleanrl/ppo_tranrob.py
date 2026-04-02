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
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

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


class AcrobotBlockedSideEnv(gym.Wrapper):
    """
    Wrapper for Acrobot-v1 that blocks one side (left or right) with a randomized angle threshold.

    In Acrobot-v1, the goal is to swing the tip to reach a target height (-1.0).
    This wrapper blocks one side by adding a constraint: if the first joint angle
    goes into the blocked half-plane beyond a threshold, the episode terminates
    with a large negative reward, forcing the agent to reach the target on the
    unblocked side.

    The block angle is randomized per episode to test robustness across varying difficulty levels:
    - Smaller angles (e.g., 60°) = harder task (narrower allowed swing region)
    - Larger angles (e.g., 120°) = easier task (wider allowed swing region)

    Args:
        env: The base Acrobot-v1 environment
        blocked_side: Which side to block ('left' or 'right'), or None for random
        perturbation_seed: Optional seed for reproducible randomization (None for random)
        angle_range: Tuple of (min_angle, max_angle) in degrees for randomization
    """
    def __init__(self, env, blocked_side=None, perturbation_seed=None, angle_range=(60, 120)):
        super().__init__(env)
        self.blocked_side = blocked_side
        self.perturbation_seed = perturbation_seed
        self.current_blocked_side = None
        self.current_block_angle = None
        # Default target height in Acrobot-v1 is -1.0
        self.base_target_height = -1.0
        # Angle range for randomization (in degrees)
        # [60, 120] provides a good range: 60° allows some swing but is challenging,
        # 120° is more permissive. Angles < 60° make swing-up nearly impossible.
        self.angle_range = angle_range

    def reset(self, seed=None, **kwargs):
        # Set seed for side selection if provided
        if self.perturbation_seed is not None:
            np.random.seed(self.perturbation_seed)

        # Randomly select which side to block if not specified
        if self.blocked_side is None:
            self.current_blocked_side = np.random.choice(['left', 'right'])
        else:
            self.current_blocked_side = self.blocked_side

        # Randomize the block angle threshold for this episode
        # This creates varying difficulty levels to better test robustness
        self.current_block_angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        self.block_angle_threshold = np.deg2rad(self.current_block_angle)

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
            reward = -50.0  # Large penalty for hitting the blocked side
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
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


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
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

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

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

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
                        eval_action, _, _, _ = eval_agent.get_action_and_value(eval_obs)
                    
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
                        eval_action, _, _, _ = eval_agent.get_action_and_value(eval_obs)
                    
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
