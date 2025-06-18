import pandas as pd
import matplotlib.pyplot as plt

def smooth_rewards(rewards, window_size=100):
    """
    Trả về mảng reward đã được làm mượt bằng trung bình động.
    Với mỗi phần tử, tính trung bình của chính nó và (window_size - 1) phần tử trước.
    """
    return rewards.rolling(window=window_size, min_periods=1).mean()

# Đọc file log
df = pd.read_csv("logs/log_M_4_N_4_K_8_D_4.csv", header=0)
df.columns = ['episode', 'reward']
df['reward'] = df['reward'].astype(str).str.replace('[\[\]]', '', regex=True).astype(float)
df['episode'] = df['episode'].astype(int)

# Làm mượt reward
df['smoothed_reward'] = smooth_rewards(df['reward'], window_size=100)

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.plot(df['episode'], df['reward'], alpha=0.3, label='Raw Reward')
plt.plot(df['episode'], df['smoothed_reward'], color='blue', label='Smoothed Reward (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Reward with Moving Average')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
