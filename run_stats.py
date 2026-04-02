import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
ENTITY = "lilkaskitc-ludwig-maximilian-university-of-munich"  # Update this!
PROJECT = "cleanrltest"
TAG = "v1.0.0-70-g004f8a0"
METRIC = "charts/episodic_return"
BIN_SIZE = 2500 # Adjust this to make the shade thicker/thinner

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"tags": TAG})

all_history = []
final_scores = []

print(f"Processing {len(runs)} seeds...")

for run in runs:
    if METRIC in run.summary:
        final_scores.append(run.summary[METRIC])
    
    # Fetch history
    history = run.scan_history(keys=["global_step", METRIC])
    df = pd.DataFrame(history)
    
    # --- Fix: Binning and Smoothing ---
    # 1. Round steps to the nearest BIN_SIZE so seeds align on the X-axis
    df["global_step"] = (df["global_step"] // BIN_SIZE) * BIN_SIZE
    
    # 2. Smooth the metric within this specific seed
    df[METRIC] = df[METRIC].rolling(window=10, min_periods=1).mean()
    
    df["seed"] = run.config.get("seed")
    all_history.append(df)

# --- 1. Final Stats ---
if final_scores:
    print(f"\nScore: {np.mean(final_scores):.2f} ± {np.std(final_scores):.2f}")

# --- 2. Plotting ---
if all_history:
    df_plot = pd.concat(all_history).dropna()
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Use 'ci="sd"' for Standard Deviation or 95 for Confidence Interval
    sns.lineplot(
        data=df_plot, 
        x="global_step", 
        y=METRIC, 
        errorbar="sd", 
        linewidth=2,
        color="#1f77b4"
    )

    plt.title(f"PPO: {PROJECT} (Mean & Std Dev)")
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Episodic Return")
    
    plt.savefig("clean_benchmark.png", dpi=300)
    print("Plot saved to: clean_benchmark.png")