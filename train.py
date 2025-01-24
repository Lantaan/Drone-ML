from stable_baselines3 import PPO
from environment.setup_env import setup_env


def train(config):
    env = setup_env(False, True, config)

    model = PPO("MlpPolicy", env, verbose=config["train_log_policy"])

    model.learn(total_timesteps=config["training_time"])
    model.save('trained/new_agent')
