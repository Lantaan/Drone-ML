from stable_baselines3 import PPO
from environment.setup_env import setup_env

env = setup_env(False)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=60000)
model.save('trained/new_agent')
