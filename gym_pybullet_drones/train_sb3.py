import os
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.WaypointsAviary import WaypointsAviary

if __name__ == "__main__":
    train_env = make_vec_env(WaypointsAviary,
                       n_envs=16,
                       seed=0)
    
    eval_env = make_vec_env(WaypointsAviary)
    
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

       
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./trained_models/best_model/",
        log_path='./logs/',
        eval_freq=int(1000),
        deterministic=True,
        render=False
    )

    model = PPO('MlpPolicy',
                train_env,
                verbose=1)

    model.learn(total_timesteps=int(1.5e6), callback=eval_callback)
    model.save("./trained_models/testing_1.zip")
