
python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrlatari&ceik=env_id&cen=exp_name&metrics=evaluation/mean_reward_vs_steps&metrics=losses/entropy&metrics=charts/episodic_return' \
        'ppo_atari_envpool?tag=pr-605' \
        'ppo_atari_envpool?tag=kr-505' \
        'credal_ppo_atari_envpool2?tag=kr-519' \
    --env-ids Pong-v5 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrlatari1e-5noent5run/ppo_atari_envpool \
    --scan-history


python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrlatari&ceik=env_id&cen=exp_name&metrics=uncertainty/total_mean&metrics=uncertainty/lower_mean&metrics=uncertainty/epistemic_mean&metrics=evaluation/mean_reward_vs_steps&metrics=losses/entropy&metrics=charts/episodic_return' \
        'credal_ppo_atari_envpool2?tag=kr-509&cl="lambda=0.1"' \
        'credal_ppo_atari_envpool2?tag=kr-507&cl="lambda=0.01"' \
        'credal_ppo_atari_envpool2?tag=kr-515&cl="lambda=0.001"' \
        'credal_ppo_atari_envpool2?tag=kr-517&cl="lambda=1e-4"' \
        'credal_ppo_atari_envpool2?tag=kr-519&cl="lambda=1e-5"' \
    --env-ids  Pong-v5  \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 5 \
    --output-filename benchmark/aaprint/ppo_atari_envpool \
    --scan-history


python -m openrlbenchmark.rlops \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrlatari&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'ppo_atari_envpool?tag=kr-505&cl=PPO' \
        'ppo_atari_envpool?tag=pr-605&cl=Entropy Reg' \
        'credal_ppo_atari_envpool2?tag=kr-515&cl=CreRL' \
    --env-ids  Pong-v5 \
    --no-check-empty-runs \
    --pc.ncols 1 \
    --pc.ncols-legend 3 \
    --pc.ylabel 'Training Return' \
    --output-filename benchmark/aaprint/ppo_atari_envpool \
    --scan-history


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


python -m openrlbenchmark.rlops \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrltest&ceik=env_id&cen=exp_name&metric=losses/explained_variance' \
        'ppo?tag=kr-505&cl=PPO' \
        'ppo?tag=pr-505&cl=Entropy Reg' \
        'credal_ppo2?tag=pr-913&cl=CreRL' \
    --env-ids Acrobot-v1 \
    --no-check-empty-runs \
    --pc.ncols 1 \
    --pc.ncols-legend 3 \
    --pc.ylabel 'Training Return' \
    --output-filename benchmark/aaprint/ppo \
    --scan-history

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=lilkaskitc-ludwig-maximilian-university-of-munich&wpn=cleanrltest&ceik=env_id&cen=exp_name&metrics=uncertainty/total_mean&metrics=uncertainty/lower_mean&metrics=uncertainty/epistemic_mean&metrics=evaluation/mean_reward_vs_steps&metrics=losses/entropy&metrics=charts/episodic_return' \
       'credal_ppo2?tag=pr-907&cl="lambda=0.1"' \
       'credal_ppo2?tag=pr-915&cl="lambda=0.01"' \
       'credal_ppo2?tag=pr-913&cl="lambda=0.001"' \
       'credal_ppo2?tag=pr-917&cl="lambda=1e-4"' \
       'credal_ppo2?tag=pr-919&cl="lambda=1e-5"' \
    --env-ids Acrobot-v1 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 5 \
    --output-filename benchmark/aaprint/ppo \
    --scan-history