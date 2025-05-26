from stable_baselines3 import PPO
from env.robot_arm_env import RobotArmEnv
import sys
def train_model():
    env = RobotArmEnv(use_gui=False)  # 可以选择使用 GUI
    model = PPO(
    "MlpPolicy", 
    env, 
    batch_size=1024,  # Batch size
    n_steps=512,   # Time horizon
    learning_rate=0.0002,  # Learning rate
    ent_coef=0.001,  # β (Entropy regularization)
    clip_range=0.3,  # ε (Epsilon, acceptable difference range between old and new policies)
    gamma=0.99,      # γ (Discount factor)
    gae_lambda=0.99, # Lambda (GAE)
    n_epochs=3,      # Number of training epochs per update
    policy_kwargs=dict(net_arch=[512, 512]),  # Hidden layers, number of units
    verbose=1,
    device="cuda"
)
    # model = PPO.load("models/robot_arm_ppo_0.1_1",env=env)
    model.learn(total_timesteps=600000)
    model.save("models/robot_arm_ppo")
    env.save_data_to_files_train()
    env.plot_success_rate()
    env.plot_End_Effector_Distance_to_Target_Full()
    env.plot_End_Effector_Orientation_to_Target_Full()
    env.plot_End_Effector_Distance_to_Target()
    env.plot_End_Effector_Orientation_to_Target()
    env.plot_reward()
    env.plot_avgreward()

def test_model():
    env = RobotArmEnv(use_gui=False)  # 可以选择使用 GUI
    model = PPO.load("models/robot_arm_ppo")

    obs = env.reset()
    for _ in range(600000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            env.reset()
    env.save_data_to_files_test()
    env.plot_success_rate_test()
    env.success_step()
    env.plot_End_Effector_Distance_to_Target_test_Full()
    env.plot_End_Effector_Orientation_to_Target_test_Full()
    env.plot_End_Effector_Distance_to_Target_test()
    env.plot_End_Effector_Orientation_to_Target_test()


# from stable_baselines3 import DDPG
# from stable_baselines3.common.noise import NormalActionNoise
# from env.robot_arm_env import RobotArmEnv
# import numpy as np

# def train_model():
#     env = RobotArmEnv(use_gui=True)  # 可以选择使用 GUI

#     # 定义动作噪声（可选，用于探索）
#     n_actions = env.action_space.shape[-1]
#     action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#     # 创建 DDPG 模型
#     model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, learning_rate=0.001)
    
#     # 训练模型
#     model.learn(total_timesteps=100000)
    
# #     # 保存模型
#     model.save("models/robot_arm_ddpg")

# def test_model():
#     env = RobotArmEnv(use_gui=True)  # 可以选择使用 GUI
#     model = DDPG.load("models/robot_arm_ddpg")

#     obs = env.reset()
#     for _ in range(1000):
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         env.render()
#         if done:
#             obs = env.reset()

# from stable_baselines3 import SAC
# from stable_baselines3.common.noise import NormalActionNoise
# from env.robot_arm_env import RobotArmEnv
# import numpy as np

# def train_model():
#     env = RobotArmEnv(use_gui=False)  # 可以选择使用 GUI

#     # SAC 不需要动作噪声，所以不需要定义 NormalActionNoise

#     # 创建 SAC 模型
#     model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.001)
    
#     # 训练模型
#     model.learn(total_timesteps=100000)
    
#     # 保存模型
#     model.save("models/robot_arm_sac")

# def test_model():
#     env = RobotArmEnv(use_gui=True)  # 可以选择使用 GUI
#     model = SAC.load("models/robot_arm_sac")

#     obs = env.reset()
#     for _ in range(1000):
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         env.render()
#         if done:
#             obs = env.reset()
