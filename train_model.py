import os 
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


# Make logs directory if it doesn't exist

log_path = os.path.join('Training', 'Logs')
os.makedirs(log_path, exist_ok=True)

# Create the Fetch Slide environment

env = gym.make('FetchSlideDense-v2', render_mode='human')

# Initialize the PPO model with the environment
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# Train the model
model.learn(total_timesteps=1000000)

# Save the model into the models folder
model_path = os.path.join('Training', 'Models', 'PPO_FetchSlideDense-v2')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

model.save(model_path)
print(f"Model saved to {model_path}")