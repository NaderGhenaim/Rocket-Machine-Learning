import gymnasium as gym
from stable_baselines3 import PPO

# create  environment
# env = gym.make("LunarLander-v3", render_mode="human")

# training without rendering environment (faster training)
env = gym.make("LunarLander-v3")

# initialize the PPO model (reinforcement learning) with a standard small neural network
model = PPO("MlpPolicy", env, verbose=1)

# train  model (1 million steps takes about 20 minutes on a standard CPU/Colab)
print("Training started")
model.learn(total_timesteps=1_000_000)

# 4. Save the trained model for later test
model.save("ppo_lunar_lander")
print("Training finished and model saved")
