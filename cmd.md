python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1  \
    --command "python cleanrl/ppo.py --track --wandb-project-name cleanrltest" \
    --num-seeds 3 \
    --workers 9 

python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1  \
    --command "python cleanrl/credal_ppo2.py --track --wandb-project-name cleanrltest" \
    --num-seeds 3 \
    --workers 9 

python -m openrlbenchmark.rlops \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrltest&ceik=env_id&cen=exp_name&metric=evaluation/mean_reward_vs_steps' \
        'ppo?tag=pr-505' \
        'credal_ppo?tag=pr-511' \
    --env-ids CartPole-v1 Acrobot-v1 \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl10/ppo \
    --scan-history

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrltest&ceik=env_id&cen=exp_name&metrics=evaluation/mean_reward_vs_steps&metrics=losses/entropy&metrics=charts/episodic_return' \
        'ppo?tag=kr-505' \
        'ppo?tag=pr-505' \
        'credal_ppo2?tag=pr-919' \
    --env-ids Acrobot-v1 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl1e-5int/ppo \
    --scan-history

python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 \
    --command "python cleanrl/ppo_atari.py --track --wandb-project-name cleanrlatari" \
    --num-seeds 3 \
    --workers 9 

python -m cleanrl_utils.benchmark \
    --env-ids BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "python cleanrl/credal_ppo_atari.py --track --wandb-project-name cleanrlatari" \
    --num-seeds 3 \
    --workers 9 

python -m openrlbenchmark.rlops \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrlatari&ceik=env_id&cen=exp_name&metric=evaluation/mean_reward_vs_steps' \
        'ppo_atari?tag=pr-505' \
        'credal_ppo_atari?tag=pr-507' \
    --env-ids BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrlatari10/ppo \
    --scan-history

python -m cleanrl_utils.benchmark \
    --env-ids Pong-v5 \
    --command "python cleanrl/ppo_atari_envpool.py --track --wandb-project-name cleanrlatari" \
    --num-seeds 3 \
    --workers 9 

python -m cleanrl_utils.benchmark \
    --env-ids Pong-v5 \
    --command "python cleanrl/credal_ppo_atari_envpool2.py --track --wandb-project-name cleanrlatari" \
    --num-seeds 3 \
    --workers 9 

python -m openrlbenchmark.rlops \
   --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrlatari&ceik=env_id&cen=exp_name&metric=evaluation/mean_reward_vs_steps' \
        'ppo_atari_envpool?tag=pr-605' \
        'credal_ppo_atari_envpool?tag=pr-611' \
    --env-ids Pong-v5 \
    --no-check-empty-runs \
    --pc.ncols 1 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrlatari0.001/ppo_atari_envpool \
    --scan-history

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrlatari&ceik=env_id&cen=exp_name&metrics=evaluation/mean_reward_vs_steps&metrics=losses/entropy&metrics=charts/episodic_return' \
        'ppo_atari_envpool?tag=pr-605' \
        'ppo_atari_envpool?tag=kr-505' \
        'credal_ppo_atari_envpool2?tag=kr-523' \
    --env-ids Pong-v5 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrlatari1e-7noent/ppo_atari_envpool \
    --scan-history


python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrlatari&ceik=env_id&cen=exp_name&metrics=losses/value_loss&metrics=losses/entropy&metrics=losses/policy_loss' \
        'credal_ppo_atari_envpool?tag=pr-611' \
        'credal_ppo_atari_envpool?tag=pr-613' \
    --env-ids Pong-v5 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrlatarituning/ppo_atari_envpool \
    --scan-history

python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 \
    --command "torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --track --wandb-project-name cleanrlatari" \
    --num-seeds 3 \
    --workers 9 

python -m cleanrl_utils.benchmark \
    --env-ids Hopper-v4  \
    --command "python cleanrl/ppo_continuous_action.py --track --wandb-project-name cleanrlmujoco" \
    --num-seeds 3 \
    --workers 9 

python -m cleanrl_utils.benchmark \
    --env-ids Hopper-v4  \
    --command "python cleanrl/credal_ppo_continuous_action.py --track --wandb-project-name cleanrlmujoco" \
    --num-seeds 3 \
    --workers 9 

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrlmujoco&ceik=env_id&cen=exp_name&metrics=evaluation/mean_reward_vs_steps&metrics=losses/entropy&metrics=charts/episodic_return' \
        'ppo_continuous_action?tag=pr-705' \
        'credal_ppo_continuous_action?tag=pr-717' \
    --env-ids Hopper-v4 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrlmujoco1e-8/ppo_continuous_action \
    --scan-history

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrlrewrob&ceik=env_id&cen=exp_name&metrics=evaluation/mean_reward_vs_steps&metrics=losses/entropy&metrics=evaluation/reward_perturbation_correlation' \
        'ppo_rewrob?tag=kr-505' \
        'ppo_rewrob?tag=pr-505' \
        'credal_ppo2_rewrob?tag=pr-519' \
    --env-ids Acrobot-v1 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl0.01rewrobnoent/ppo \
    --scan-history

ppo_tranrob2, penalty -50.0
ppo_tranrob3, penalty -100.0

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrltranrob&ceik=env_id&cen=exp_name&metrics=evaluation/mean_reward_vs_steps&metrics=evaluation/target_reached_count&metrics=evaluation/side_violation_count' \
        'ppo_tranrob?tag=kr-505' \
        'ppo_tranrob?tag=pr-905' \
        'credal_ppo2_tranrob?tag=pr-915' \
    --env-ids Acrobot-v1 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl1.0tranrob3noentrun5/ppo \
    --scan-history

python -m cleanrl_utils.benchmark \
    --env-ids Pong-v5 \
    --command "python cleanrl/credal_ppo_atari_envpool2_rewrob.py --track --wandb-project-name cleanrlatarirew" \
    --num-seeds 3 \
    --workers 9


python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrlatarirew&ceik=env_id&cen=exp_name&metrics=evaluation/mean_reward_vs_steps&metrics=evaluation/mean_reward_double_player_vs_steps&metrics=evaluation/mean_reward_double_opponent_vs_steps' \
        'ppo_atari_envpool_rewrob?tag=pr-505' \
        'ppo_atari_envpool_rewrob?tag=kr-505' \
        'credal_ppo_atari_envpool2_rewrob?tag=kr-507' \
    --env-ids  Pong-v5 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrlatari1rewrobnoent5run/ppo_atari_envpool \
    --scan-history

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrlataritran&ceik=env_id&cen=exp_name&metrics=evaluation/mean_reward_vs_steps&metrics=evaluation/wall_collision_rate&metrics=evaluation/wall_center_y' \
        'ppo_atari_envpool_tranrob?tag=pr-505' \
        'ppo_atari_envpool_tranrob?tag=kr-505' \
        'credal_ppo_atari_envpool2_tranrob?tag=kr-521' \
    --env-ids  Pong-v5 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrlatari1e5tranrobnoent5run/ppo_atari_envpool \
    --scan-history
