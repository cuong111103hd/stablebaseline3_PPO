import os
import time
import csv
import glob
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from environment import STAR  # Ensure STAR class is correctly implemented with Gymnasium


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
                self.episode_count = 0
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


def make_env():
    env = STAR(num_antennas=4, num_star_elements=4, num_users=4, num_d2d_pairs=4)
    env = Monitor(env)
    return env


def find_latest_model(model_dir):
    model_files = glob.glob(os.path.join(model_dir, "PPO_STAR_*.zip"))
    if not model_files:
        return None
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model


def train(resume=True, total_timesteps=int(5e5), save_freq=100000):
    log_dir = "logs"
    model_dir = "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "log.csv")
    env = make_env()

    model_path = find_latest_model(model_dir) if resume else None

    if model_path:
        print(f"[INFO] Resuming from latest model: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("[INFO] Starting new training session...")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    callback = RewardLoggerCallback(log_path=log_file)

    timestep = 0
    while timestep < total_timesteps:
        model.learn(total_timesteps=save_freq, reset_num_timesteps=False, callback=callback)
        timestep += save_freq

        save_name = f"PPO_STAR_{timestep}_{int(time.time())}.zip"
        model.save(os.path.join(model_dir, save_name))
        print(f"[INFO] Model saved: {save_name}")


def test(model_path=None):
    env = make_env()
    if model_path is None:
        model_path = find_latest_model("models")
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
    train(
        resume=True,              # Set to False to start fresh
        total_timesteps=int(5e5), # Total training timesteps
        save_freq=100000          # Save model every N timesteps
    )

    # Uncomment to run a test after training:
    # test()
