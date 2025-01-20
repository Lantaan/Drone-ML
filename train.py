from stable_baselines3 import PPO
from environment.setup_env import setup_env
from config import training_time, train_log_policy

env = setup_env(False)

model = PPO("MlpPolicy", env, verbose=train_log_policy)

model.learn(total_timesteps=training_time)
model.save('trained/new_agent')
