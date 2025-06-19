import os
import time
import csv
import glob
import torch
import numpy as np
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment import STAR


def generate_log_filename(log_dir, prefix="log", M=4, N=4, K=4, D=4):
    parts = [prefix, f"M_{M}", f"N_{N}", f"K_{K}", f"D_{D}"]
    filename = "_".join(parts) + ".csv"
    return os.path.join(log_dir, filename)


def find_latest_model_by_params(model_dir, M, N, K, D):
    pattern = f"PPO_STAR_M_{M}_N_{N}_K_{K}_D_{D}_step_*.zip"
    full_path = os.path.join(model_dir, pattern)
    print(f"[DEBUG] Looking for model files at: {full_path}")

    model_files = glob.glob(full_path)
    print(f"[DEBUG] Found model files: {model_files}")
    if not model_files:
        return None, 0

    def extract_step(file_name):
        try:
            parts = file_name.split("_")
            idx = parts.index("step")
            return int(parts[idx + 1])
        except:
            return -1

    model_files.sort(key=lambda x: extract_step(x), reverse=True)
    latest_model = model_files[0]
    current_step = extract_step(latest_model)
    return latest_model, current_step


def save_model(model, model_dir, M, N, K, D, timestep):
    os.makedirs(model_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"PPO_STAR_M_{M}_N_{N}_K_{K}_D_{D}_step_{timestep}_{timestamp}.zip"
    model.save(os.path.join(model_dir, filename))
    print(f"[INFO] Model saved: {filename}")


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_count = 0

        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, mode="r") as f:
                    rows = list(csv.reader(f))
                    if len(rows) > 1:
                        last_line = rows[-1]
                        if last_line[0].isdigit():
                            self.episode_count = int(last_line[0])
            except Exception:
                print("[WARN] Could not resume episode count, starting from 0")
        else:
            with open(self.log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Reward"])

    def _on_step(self) -> bool:
        if self.locals.get("done", False):
            reward = self.locals["rewards"]
            self.episode_count += 1
            with open(self.log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.episode_count, reward])
        return True


def make_env(M, N, K, D):
    env = STAR(M, N, K, D)
    env = Monitor(env)
    return env


def train(M, N, K, D, resume=True, total_timesteps=int(5e5), save_freq=100000):
    log_dir = "logs"
    model_dir = "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    log_file = generate_log_filename(log_dir, prefix="log", M=M, N=N, K=K, D=D)
    env = make_env(M, N, K, D)

    if resume:
        print("[INFO] Attempting to resume training...")
        model_path, current_step = find_latest_model_by_params(model_dir, M, N, K, D)
        if model_path:
            print(f"[INFO] Resuming from: {model_path} at step {current_step}")
            model = PPO.load(model_path, env=env)
        else:
            print("[WARN] No matching model found, starting new training")
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
            current_step = 0
    else:
        print("[INFO] Starting new training session...")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        current_step = 0

    callback = RewardLoggerCallback(log_path=log_file)

    while current_step < total_timesteps:
        train_step = min(save_freq, total_timesteps - current_step)
        model.learn(total_timesteps=train_step, reset_num_timesteps=False, callback=callback)
        current_step += train_step
        save_model(model, model_dir, M, N, K, D, current_step)


def test(model_path=None, M=4, N=4, K=4, D=4):
    env = make_env(M, N, K, D)
    if model_path is None:
        model_path, _ = find_latest_model_by_params("models", M, N, K, D)
    if model_path is None:
        print("[ERROR] No model found to test.")
        return

    print(f"[TEST] Using model: {model_path}")
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    total_reward = 0
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"[TEST] Total reward: {total_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=4, help="Number of BS antennas")
    parser.add_argument("--N", type=int, default=4, help="Number of STAR-RIS elements")
    parser.add_argument("--K", type=int, default=4, help="Number of users")
    parser.add_argument("--D", type=int, default=4, help="Number of D2D pairs")
    parser.add_argument("--resume", action="store_true",default=True, help="Resume from last checkpoint")
    parser.add_argument("--timesteps", type=int, default=int(3e6), help="Total training timesteps")
    parser.add_argument("--save_freq", type=int, default=100000, help="Save frequency")

    args = parser.parse_args()

    train(
        M=args.M,
        N=args.N,
        K=args.K,
        D=args.D,
        resume=args.resume,
        total_timesteps=args.timesteps,
        save_freq=args.save_freq
    )

    # Uncomment to test:
    # test(M=args.M, N=args.N, K=args.K, D=args.D)
