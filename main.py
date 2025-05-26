import pybullet as p
import gym
import numpy as np
from stable_baselines3 import PPO
import pybullet_data
import gym
import numpy as np
from stable_baselines3 import PPO
from models.ppo_model import train_model,test_model
from env.robot_arm_env import RobotArmEnv
if __name__ == "__main__":
    # 训练模型
    train_model()