import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import time 
import random
import matplotlib.pyplot as plt
# 定义动作空间和观察空间的参数
num_actions = 6  # 根据机械臂的关节数设置
num_observations = num_actions * 2  # 位置和速度
# 定义目标位置的随机生成范围
position_range_left = np.array([[0.1, 0.6], [-0.6, 0.6], [0.0, 0.6]])  # 左臂目标位置的范围
def generate_random_target_position(position_range, base_offset, max_radius):
    def is_target_in_workspace(target_pos, max_radius, base_offset):
        # 计算目标位置相对于机械臂原点的距离
        relative_pos = np.array(target_pos) - np.array(base_offset)
        distance = np.linalg.norm(relative_pos)
        return distance <= max_radius

    while True:
        # 随机生成目标位置
        position = np.array([
            np.random.uniform(position_range[0][0], position_range[0][1]),
            np.random.uniform(position_range[1][0], position_range[1][1]),
            np.random.uniform(position_range[2][0], position_range[2][1])
        ])
        # 检查目标位置是否在工作空间内
        if is_target_in_workspace(position, max_radius, base_offset):
            return position
        else:
            print(f"Target Position {position} is out of workspace, regenerating...")


def generate_random_orientation():
    # 随机生成一个四元数来表示目标姿态
    random_rotation = R.random().as_quat()
    return random_rotation
def normalize_angle_to_pi_range(angle):
    # 将角度归一化到[-pi, pi]范围
    integer_part = (angle + np.pi) // (2 * np.pi)  # 首先将角度值转换到[0, 2*pi]范围
    angle = angle - integer_part * (2 * np.pi)
    return angle

left_end_effector_link_index = 5  # 左臂末端的链接索引
class RobotArmEnv(gym.Env):
    def __init__(self, use_gui=False):
        super(RobotArmEnv, self).__init__()
        if p.isConnected():
            p.disconnect()
        self.total_tasks = []  # 用于记录总任务数
        self.success_rates = []  # 用于记录成功率
        self.final_dis_err_min = []
        self.final_ori_err_min = []
        self.final_total_reward = []
        self.final_avg_reward = []
        self.step_success = []
        self.total_success = []
        self.final_round_reward = []
        self.step_count = 0
        self.total_f = 0
        self.su = 0
        self.total_step_success = 0
        if use_gui:
            self.gui_client_id = p.connect(p.GUI)
        else:
            self.gui_client_id = p.connect(p.DIRECT)

        self.action_space = spaces.Box(low=-3.1415, high=3.1415, shape=(num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_observations,), dtype=np.float32)
        p.setGravity(0, 0, 0)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        urdf_path = "/root/MyArmDescription/myarm_rl/data/ur5_six_description/urdf/ur5_six.urdf"
        self.arm_id = p.loadURDF(urdf_path, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.arm_id)
        self.joint_limits = self._get_joint_limits_from_urdf(urdf_path)
        self.start_time = time.time()
        self.prev_positions = {
            'left': np.array([0.0, 0.0, 0.0])
        }
        self.prev_orientations = {
            'left': np.array([0.0, 0.0, 0.0, 1.0]) # 四元数形式
        }
        self.target_position_left = generate_random_target_position(position_range_left, [0.0,0.0,0.0], 0.6)
        self.target_orientation_left = generate_random_orientation()
        self.dis_Amin = 0.6  
        self.ori_Amin = 2  
        self.error_Amin = float('inf')
        self.dis_err_min = float('inf')
        self.ori_err_min = float('inf')
        self.repeat = 0
        self.position_threshold = 0.0001
        self.orientation_degree_threshold = 0.001
        orientation_ra = np.deg2rad(self.orientation_degree_threshold)
        self.orientation_threshold = 1- np.cos(orientation_ra/ 2)
        self.orientation_error = 1 + self.orientation_threshold
        self.error_threshold = self.position_threshold * self.orientation_error + self.orientation_error - 1
        self.joint_angles_history = []
        self.total_reward = 0
        self.round_reward = 0
        self.lowerLimits = [-3.14, -2.26, -2.96, -3.14, -3.14, -3.14]
        self.upperLimits = [ 3.14,  2.26,  2.96,  3.14,  3.14,  3.14]
        self.jointRanges  = [u - l for u, l in zip(self.upperLimits, self.lowerLimits)]

    def reset(self):
        # 重新设置环境
        self.target_position_left = generate_random_target_position(position_range_left, [0.0,0.0,0.0], 0.6)
        self.target_orientation_left = generate_random_orientation()
        self.dis_Amin = 0.6 
        self.ori_Amin = 2  
        self.error_Amin = float('inf')
        self.dis_err_min = float('inf')
        self.ori_err_min = float('inf')
        self.joint_angles_history = []
        self.repeat = 0
        self.round_reward = 0
        print(f"Generated Left Target Position: {self.target_position_left}")
        print(f"Generated Left Target Orientation: {self.target_orientation_left}") 
        self.step_count = 0 
        self._reset_arm()
        self.start_time = time.time()
        return self._get_observation()

    def step(self, action):
        # 执行动作
        self._apply_action(action)
        if self.step_count > 1000:
            print("Task False")
            self._print_status()
            self.final_dis_err_min.append(self.final_dis_err_min_num)
            self.final_ori_err_min.append(self.final_ori_err_min_num)
            self.total_f += 1
            self.total_tasks.append(self.su + self.total_f)
            pece = (self.su / (self.su + self.total_f)) * 100 if (self.su + self.total_f) > 0 else 0
            self.success_rates.append(pece)
            self.final_total_reward.append(self.total_reward)
            self.avg_reward = self.total_reward / (self.su + self.total_f)
            self.final_avg_reward.append(self.avg_reward)
            self.final_round_reward.append(self.round_reward)
            print(f"Repeat Num: {self.repeat}")
            print(f"False Num: {self.total_f}")
            print(f"Success Num {self.su}")
            print(f"Total Num: {self.su + self.total_f}")
            print(f"Success rate: {pece:.2f}%")
            self.repeat += 1
            self.reset()
            return self._get_observation(), -3, False, {}  # 返回失败状态，奖励为负值
        self.step_count += 1
        reward = self._calculate_reward()
        self.total_reward += reward
        self.round_reward += reward
        done = self._check_done()

        if done:
            print("任务完成")
            self._print_status_success()
            self.final_dis_err_min.append(self.final_dis_err_min_num)
            self.final_ori_err_min.append(self.final_ori_err_min_num)
            self.su += 1
            self.total_step_success += self.step_count
            self.step_success.append(self.total_step_success / self.su)
            self.total_success.append(self.su)
            self.total_tasks.append(self.su + self.total_f)
            pece = (self.su / (self.su + self.total_f)) * 100 if (self.su + self.total_f) > 0 else 0
            self.success_rates.append(pece)
            self.final_total_reward.append(self.total_reward)
            self.avg_reward = self.total_reward / (self.su + self.total_f)
            self.final_avg_reward.append(self.avg_reward)
            self.final_round_reward.append(self.round_reward)
            print(f"Repeat Num: {self.repeat}")
            print(f"False Num: {self.total_f}")
            print(f"Success Num {self.su}")
            print(f"Total Num: {self.su + self.total_f}")
            print(f"Success rate: {pece:.2f}%")
            print(f"step_success: {self.total_step_success / self.su}")
            # print(f"close_to_target_reward: {self.task_completion_reward}")
        obs = self._get_observation()

        return obs, reward, done, {}

    def render(self, mode='human'):
        # 渲染环境
        if mode == 'human':
            if not hasattr(self, 'gui_client_id'):
                self.gui_client_id = p.connect(p.GUI)
            
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=50,
                cameraPitch=-35,
                cameraTargetPosition=[0, 0, 0]
            )

    def close(self):
        # 关闭环境
        if hasattr(self, 'gui_client_id'):
            p.disconnect(self.gui_client_id)
        else:
            p.disconnect()

    def _apply_action(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if not self._check_done():
            # current_joint_angles_left = p.getJointStates(self.arm_id, [0,1,2,3,4,5])
            # current_joint_angles_left = [state[0] for state in current_joint_angles_left]
            # left_joint_angles_selected = np.array(current_joint_angles_left[0:7])
            left_joint_angles_new = action[0:6]
            p.setJointMotorControlArray(self.arm_id,[0,1,2,3,4,5],p.POSITION_CONTROL,targetPositions=left_joint_angles_new)
            for _ in range(150):
                p.stepSimulation()
            self._check_min()
        # else:
        #     print(f"RL")
        if not self._check_done():
            current_joint_angles_left = p.getJointStates(self.arm_id, [0,1,2,3,4,5])
            # current_joint_angles_left = [state[0] for state in current_joint_angles_left]
            left_joint_angles = p.calculateInverseKinematics(self.arm_id, 
                                                             left_end_effector_link_index, 
                                                             self.target_position_left, 
                                                             self.target_orientation_left,
                                                             maxNumIterations=300, 
                                                             residualThreshold=0,
                                                             lowerLimits=self.lowerLimits,
                                                             upperLimits=self.upperLimits,
                                                             jointRanges=self.jointRanges,
                                                             )
            # print("left_joint_angles:", left_joint_angles)
            # left_joint_angles_selected = np.array(left_joint_angles[0:6])
            # left_joint_angles_new = np.array(left_joint_angles_selected)
            # print("left_joint_angles_new:", left_joint_angles_new)
            p.setJointMotorControlArray(self.arm_id,[0,1,2,3,4,5],p.POSITION_CONTROL,targetPositions=left_joint_angles)
            for _ in range(150):
                p.stepSimulation()
            self._check_min()
        # # else:
        #     print(f"IK")
    def _calculate_reward(self):
        left_end_effector_position = self._get_end_effector_positions()
        left_end_effector_orientation= self._get_end_effector_orientations()
        distance_left = np.linalg.norm(np.array(left_end_effector_position) - np.array(self.target_position_left))
        left_end_effector_orientation = np.array(left_end_effector_orientation)
        target_orientation_left = np.array(self.target_orientation_left)
        orientation_error_left = 1- self.cosine_similarity(left_end_effector_orientation, target_orientation_left)
        ad_orientation_error_left = self.adjusted_cosine_distance(left_end_effector_orientation, target_orientation_left)
        error_left = distance_left * ad_orientation_error_left + ad_orientation_error_left - 1
        reward_left = 0
        reward_ori_left = 0
        if not self._check_done():
            if self.dis_Amin < distance_left:
                reward_left = 0.25*(self.dis_Amin - distance_left)
            elif self.dis_Amin > distance_left:
                reward_left = (self.dis_Amin - distance_left)
            else:
                reward_left = 0

            if self.ori_Amin < orientation_error_left:
                reward_ori_left = 0.25*(self.ori_Amin - orientation_error_left)
            elif self.ori_Amin > orientation_error_left:
                reward_ori_left = (self.ori_Amin - orientation_error_left)
            else:
                reward_ori_left = 0
        self.task_completion_reward = 3.0 + (self.error_threshold - error_left) * 1e6 if self._check_done() else -0.003
        reward =  self.task_completion_reward + (reward_left + reward_ori_left*0.3)/10
        return float(reward)  # 转换为浮点数，确保返回值是有效的数值

    def _check_done(self):
        left_end_effector_position = self._get_end_effector_positions()
        left_end_effector_orientation= self._get_end_effector_orientations()
        distance_left = np.linalg.norm(np.array(left_end_effector_position) - np.array(self.target_position_left))
        left_end_effector_orientation = np.array(left_end_effector_orientation)
        target_orientation_left = np.array(self.target_orientation_left)
        orientation_error_left = 1- self.cosine_similarity(left_end_effector_orientation, target_orientation_left)
        ad_orientation_error_left = self.adjusted_cosine_distance(left_end_effector_orientation, target_orientation_left)
        error_left = distance_left * ad_orientation_error_left + ad_orientation_error_left - 1
        done = (distance_left < self.position_threshold and orientation_error_left < self.orientation_threshold)
        return done

    def _check_min(self):
        left_end_effector_position = self._get_end_effector_positions()
        left_end_effector_orientation= self._get_end_effector_orientations()
        distance_left = np.linalg.norm(np.array(left_end_effector_position) - np.array(self.target_position_left))
        left_end_effector_orientation = np.array(left_end_effector_orientation)
        target_orientation_left = np.array(self.target_orientation_left)
        orientation_error_left = 1- self.cosine_similarity(left_end_effector_orientation, target_orientation_left)
        ad_orientation_error_left = self.adjusted_cosine_distance(left_end_effector_orientation, target_orientation_left)
        error_left = distance_left * ad_orientation_error_left + ad_orientation_error_left - 1
        if not hasattr(self, 'dis_Amin'):
            self.dis_Amin = distance_left
        self.dis_Amin = min(distance_left, self.dis_Amin)
        if not hasattr(self, 'ori_Amin'):
            self.ori_Amin = orientation_error_left
        self.ori_Amin = min(orientation_error_left, self.ori_Amin)
        if not hasattr(self, 'error_Amin'):
            self.error_Amin = error_left
        else :
            if error_left < self.error_Amin:
                self.min_left_end_effector_position = left_end_effector_position
                self.min_left_end_effector_orientation = left_end_effector_orientation
                self.error_Amin = error_left
                self.dis_err_min = distance_left
                self.ori_err_min = self.quaternion_angle_difference(left_end_effector_orientation,target_orientation_left)
                joint_angles = self.get_joint_angles()
                self.joint_angles_history = joint_angles

    def _get_observation(self):
        # 获取当前观测
        observation = []
        for joint_id in range(num_actions):
            joint_state = p.getJointState(self.arm_id, joint_id)
            observation.extend(joint_state[0:2])  # 位置和速度
        return np.array(observation)

    def _get_end_effector_positions(self):
        left_end_effector_state = p.getLinkState(self.arm_id, left_end_effector_link_index)
        left_end_effector_position = np.array(left_end_effector_state[0])  # 左臂末端位置

        return left_end_effector_position

    def _reset_arm(self):
        joint_indices = [0,1,2,3,4,5]
        for joint_id in joint_indices:
            random_value = random.uniform(-3.1415, -3.1415)  # Adjust the range as per your joint limits
            p.resetJointState(self.arm_id, joint_id, targetValue=random_value)


    def _get_joint_limits_from_urdf(self, urdf_path):
        # 解析 URDF 文件
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        joint_limits = {}

        # 遍历 URDF 中的所有关节
        for joint in root.findall('joint'):
            joint_name = joint.get('name')
            limit = joint.find('limit')
            
            if limit is not None:
                joint_limits[joint_name] = {
                    'lower': float(limit.get('lower', -float('inf'))),
                    'upper': float(limit.get('upper', float('inf')))
                }
        
        return joint_limits

    def _get_joint_name(self, joint_id):
        # 获取关节名称
        joint_info = p.getJointInfo(self.arm_id, joint_id)
        return joint_info[1].decode('utf-8')

    def _print_status(self):
        self.final_dis_err_min_num = self.dis_err_min
        self.final_ori_err_min_num = self.ori_err_min
        print(f"Left End Effector Position: {self.min_left_end_effector_position}")
        print(f"Left End Effector Distance to Target: {self.dis_err_min}")
        print(f"Left End Effector Orientation: {self.min_left_end_effector_orientation}")
        print(f"Left End Effector Orientation Error: {self.ori_err_min}")
        print(f"各关节角度: {self.joint_angles_history}")

    def _print_status_success(self):
        left_end_effector_position = self._get_end_effector_positions()
        left_end_effector_orientation= self._get_end_effector_orientations()
        distance_left = np.linalg.norm(np.array(left_end_effector_position) - np.array(self.target_position_left))
        left_end_effector_orientation = np.array(left_end_effector_orientation)
        target_orientation_left = np.array(self.target_orientation_left)
        orientation_err = self.quaternion_angle_difference(left_end_effector_orientation,target_orientation_left)
        self.final_dis_err_min_num = distance_left
        self.final_ori_err_min_num = orientation_err
        print(f"Left End Effector Position: {left_end_effector_position}")
        print(f"Left End Effector Distance to Target: {distance_left}")
        print(f"Left End Effector Orientation: {left_end_effector_orientation}")
        print(f"Left End Effector Orientation Error: {orientation_err}")
        joint_angles = self.get_joint_angles()
        print("各关节角度:", joint_angles)

    def _get_end_effector_orientations(self):
        left_orientation = p.getLinkState(self.arm_id, left_end_effector_link_index)[1]
        return left_orientation
    def get_joint_angles(self):
        # 获取机器臂的关节角度
        joint_angles = []
        for joint_index in range(self.num_joints):
            joint_state = p.getJointState(self.arm_id, joint_index)
            joint_angles.append(joint_state[0])  # 0表示关节位置（角度）
        return joint_angles

    def cosine_similarity(self,q1, q2):
        # Convert quaternions to rotation matrices
        r1 = R.from_quat(q1).as_matrix()
        r2 = R.from_quat(q2).as_matrix()

        # Compute the difference rotation matrix
        diff_r = np.dot(r1, r2.T)
        trace_diff_r = np.trace(diff_r)

        # Ensure the trace value is within the valid range for arccos
        trace_value = (trace_diff_r - 1) / 2
        trace_value = np.clip(trace_value, -1, 1)  # Clip the value to avoid domain errors

        # Calculate the angle in radians
        angle_r = np.arccos(trace_value)

        # Calculate the cosine of half the angle
        cos_theta_over_2 = np.cos(angle_r / 2)

        return cos_theta_over_2

    def adjusted_cosine_distance(self,v1, v2):
        similarity = self.cosine_similarity(v1, v2)
        return (1 - similarity) + 1  # 根据描述调整

    def quaternion_angle_difference(self,q1,q2):
        # 将四元数转换为旋转矩阵
        r1 = R.from_quat(q1).as_matrix()
        r2 = R.from_quat(q2).as_matrix()

        # 计算旋转矩阵的差异
        diff_r = np.dot(r1, r2.T)
        trace_diff_r = np.trace(diff_r)

        # 确保轨迹值在 arccos 的有效范围内
        trace_value = (trace_diff_r - 1) / 2
        trace_value = np.clip(trace_value, -1, 1)  # 防止数值问题导致超出[-1, 1]

        # 计算角度（弧度）
        angle_radians = np.arccos(trace_value)

        # 将弧度转换为度数
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees
    # package://dualarm_description/meshes







    def plot_success_rate(self):
            plt.figure(figsize=(10, 5))
            total_tasks_array = np.array(self.total_tasks)
            success_rates_array = np.array(self.success_rates, dtype=float)
            plt.plot(total_tasks_array, success_rates_array, marker='o')

            plt.xlabel('Total Tasks')
            plt.ylabel('Success Rate (%)')
            plt.title('Success Rate (Train)')
            plt.grid()
            plt.savefig("picture_train/Success_Rate_train.png")
            plt.close()  # 关闭图形以释放内存
    
    def plot_reward(self):
        plt.figure(figsize=(10, 5))
        total_tasks_array = np.array(self.total_tasks)
        final_total_reward = np.array(self.final_total_reward, dtype=float)
        plt.plot(total_tasks_array, final_total_reward, marker='o')

        plt.xlabel('Total Tasks')
        plt.ylabel('Reward')
        plt.title('Reward')
        plt.grid()
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.savefig("picture_train/Reward.png")
        plt.close()  # 关闭图形以释放内存

    def plot_avgreward(self):
        plt.figure(figsize=(10, 5))
        total_tasks_array = np.array(self.total_tasks)
        final_avg_reward = np.array(self.final_avg_reward, dtype=float)
        plt.plot(total_tasks_array, final_avg_reward, marker='o')

        plt.xlabel('Total Tasks')
        plt.ylabel('Average Reward')
        plt.title('Average Reward')
        plt.grid()
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.savefig("picture_train/Average Reward.png")
        plt.close()  # 关闭图形以释放内存

    def success_step(self):
        plt.figure(figsize=(10, 5))
        total_success_array = np.array(self.total_success)
        step_success_array = np.array(self.step_success, dtype=float)
        plt.plot(total_success_array, step_success_array, marker='o')
        plt.xlabel('Total Success')
        plt.ylabel('Average Success Steps')
        plt.title('Average Success Steps')
        plt.grid()
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.savefig("picture_test/Average Success Steps.png")
        plt.close()  # 关闭图形以释放内存train

    def plot_success_rate_test(self):
            plt.figure(figsize=(10, 5))
            total_tasks_array = np.array(self.total_tasks)
            success_rates_array = np.array(self.success_rates, dtype=float)
            plt.plot(total_tasks_array, success_rates_array, marker='o')
            plt.xlabel('Total Tasks')
            plt.ylabel('Success Rate (%)')
            plt.title('Success Rate (Test)')
            plt.grid()
            plt.savefig("picture_test/Success_Rate_test.png")
            plt.close()  # 关闭图形以释放内存

    def plot_End_Effector_Distance_to_Target(self):
        plt.figure(figsize=(10, 5))
        total_tasks_array = np.array(self.total_tasks)
        final_dis_err_min = np.array(self.final_dis_err_min, dtype=float)
        plt.plot(total_tasks_array, final_dis_err_min, marker='o')

        plt.xlabel('Total Tasks')
        plt.ylabel('dis_err_min')
        plt.title('End_Effector_Distance_to_Target (Train)')
        plt.grid()
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.axhline(y=self.position_threshold, color='red', linestyle='--', label='y=position_threshold')
        plt.ylim(0.6e-4, 1.2e-4)
        plt.savefig("picture_train/End_Effector_Distance_to_Target_train.png")
        plt.close()  # 关闭图形以释放内存

    def plot_End_Effector_Orientation_to_Target(self):
        plt.figure(figsize=(10, 5))
        total_tasks_array = np.array(self.total_tasks)
        final_ori_err_min = np.array(self.final_ori_err_min, dtype=float)
        plt.plot(total_tasks_array, final_ori_err_min, marker='o')
        plt.xlabel('Total Tasks')
        plt.ylabel('ori_err_min')
        plt.title('End_Effector_Orientation_to_Target (Train)')
        plt.grid()
        plt.axhline(y=self.orientation_degree_threshold, color='red', linestyle='--', label='y=orientation_degree_threshold')
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.ylim(0,2e-3)
        plt.savefig("picture_train/End_Effector_Orientation_to_Target_train.png")
        plt.close()  # 关闭图形以释放内存


    def plot_End_Effector_Distance_to_Target_test(self):
        plt.figure(figsize=(10, 5))
        total_tasks_array = np.array(self.total_tasks)
        final_dis_err_min = np.array(self.final_dis_err_min, dtype=float)
        plt.plot(total_tasks_array, final_dis_err_min, marker='o')
        plt.xlabel('Total Tasks')
        plt.ylabel('dis_err_min')
        plt.title('End_Effector_Distance_to_Target (Test)')
        plt.grid()
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.axhline(y=self.position_threshold, color='red', linestyle='--', label='y=position_threshold')
        plt.ylim(0.6e-4, 1.2e-4)
        plt.savefig("picture_test/End_Effector_Distance_to_Target_test.png")
        plt.close()  # 关闭图形以释放内存

    def plot_End_Effector_Orientation_to_Target_test(self):
        plt.figure(figsize=(10, 5))
        total_tasks_array = np.array(self.total_tasks)
        final_ori_err_min = np.array(self.final_ori_err_min, dtype=float)
        plt.plot(total_tasks_array, final_ori_err_min, marker='o')

        plt.xlabel('Total Tasks')
        plt.ylabel('ori_err_min')
        plt.title('End_Effector_Orientation_to_Target (Test)')
        plt.grid()
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.axhline(y=self.orientation_degree_threshold, color='red', linestyle='--', label='y=orientation_degree_threshold')
        plt.ylim(0,2e-3)
        plt.savefig("picture_test/End_Effector_Orientation_to_Target_test.png")
        plt.close()  # 关闭图形以释放内存
    
    def plot_End_Effector_Distance_to_Target_Full (self):
        plt.figure(figsize=(10, 5))
        total_tasks_array = np.array(self.total_tasks)
        final_dis_err_min = np.array(self.final_dis_err_min, dtype=float)
        plt.plot(total_tasks_array, final_dis_err_min, marker='o')

        plt.xlabel('Total Tasks')
        plt.ylabel('dis_err_min')
        plt.title('End_Effector_Distance_to_Target (Train_Full)')
        plt.grid()
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.axhline(y=self.position_threshold, color='red', linestyle='--', label='y=position_threshold')
        plt.savefig("picture_train/End_Effector_Distance_to_Target_train_Full.png")
        plt.close()  # 关闭图形以释放内存

    def plot_End_Effector_Orientation_to_Target_Full(self):
        plt.figure(figsize=(10, 5))
        total_tasks_array = np.array(self.total_tasks)
        final_ori_err_min = np.array(self.final_ori_err_min, dtype=float)
        plt.plot(total_tasks_array, final_ori_err_min, marker='o')
        plt.xlabel('Total Tasks')
        plt.ylabel('ori_err_min')
        plt.title('End_Effector_Orientation_to_Target (Train_Full)')
        plt.grid()
        plt.axhline(y=self.orientation_degree_threshold, color='red', linestyle='--', label='y=orientation_degree_threshold')
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.savefig("picture_train/End_Effector_Orientation_to_Target_train_Full.png")
        plt.close()  # 关闭图形以释放内存

    def plot_End_Effector_Distance_to_Target_test_Full(self):
        plt.figure(figsize=(10, 5))
        total_tasks_array = np.array(self.total_tasks)
        final_dis_err_min = np.array(self.final_dis_err_min, dtype=float)
        plt.plot(total_tasks_array, final_dis_err_min, marker='o')
        plt.xlabel('Total Tasks')
        plt.ylabel('dis_err_min')
        plt.title('End_Effector_Distance_to_Target (Test_Full)')
        plt.grid()
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.axhline(y=self.position_threshold, color='red', linestyle='--', label='y=position_threshold')
        plt.savefig("picture_test/End_Effector_Distance_to_Target_test_Full.png")
        plt.close()  # 关闭图形以释放内存

    def plot_End_Effector_Orientation_to_Target_test_Full(self):
        plt.figure(figsize=(10, 5))
        total_tasks_array = np.array(self.total_tasks)
        final_ori_err_min = np.array(self.final_ori_err_min, dtype=float)
        plt.plot(total_tasks_array, final_ori_err_min, marker='o')

        plt.xlabel('Total Tasks')
        plt.ylabel('ori_err_min')
        plt.title('End_Effector_Orientation_to_Target (Test_Full)')
        plt.grid()
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        plt.axhline(y=self.orientation_degree_threshold, color='red', linestyle='--', label='y=orientation_degree_threshold')
        plt.savefig("picture_test/End_Effector_Orientation_to_Target_test_Full.png")
        plt.close()  # 关闭图形以释放内存

    def save_data_to_files_test(self):
        # 保存总任务数到文件
        with open('data_test/total_tasks.txt', 'w') as f:
            for task in self.total_tasks:
                f.write(f"{task}\n")
        
        # 保存成功率到文件
        with open('data_test/success_rates.txt', 'w') as f:
            for rate in self.success_rates:
                f.write(f"{rate}\n")
        
        # 保存最小距离误差到文件
        with open('data_test/final_dis_err_min.txt', 'w') as f:
            for dis_err in self.final_dis_err_min:
                f.write(f"{dis_err}\n")
        
        # 保存最小姿态误差到文件
        with open('data_test/final_ori_err_min.txt', 'w') as f:
            for ori_err in self.final_ori_err_min:
                f.write(f"{ori_err}\n")
        
        # 保存每一步成功率到文件
        with open('data_test/step_success.txt', 'w') as f:
            for step_succ in self.step_success:
                f.write(f"{step_succ}\n")
        
        # 保存总成功次数到文件
        with open('data_test/total_success.txt', 'w') as f:
            for succ in self.total_success:
                f.write(f"{succ}\n")


    def save_data_to_files_train(self):
        # 保存总任务数到文件
        with open('data_train/total_tasks.txt', 'w') as f:
            for task in self.total_tasks:
                f.write(f"{task}\n")
        
        # 保存成功率到文件
        with open('data_train/success_rates.txt', 'w') as f:
            for rate in self.success_rates:
                f.write(f"{rate}\n")
        
        # 保存最小距离误差到文件
        with open('data_train/final_dis_err_min.txt', 'w') as f:
            for dis_err in self.final_dis_err_min:
                f.write(f"{dis_err}\n")
        
        # 保存最小姿态误差到文件
        with open('data_train/final_ori_err_min.txt', 'w') as f:
            for ori_err in self.final_ori_err_min:
                f.write(f"{ori_err}\n")
        
        # 保存总奖励到文件
        with open('data_train/final_total_reward.txt', 'w') as f:
            for reward in self.final_total_reward:
                f.write(f"{reward}\n")
        
        # 保存平均奖励到文件
        with open('data_train/final_avg_reward.txt', 'w') as f:
            for avg_reward in self.final_avg_reward:
                f.write(f"{avg_reward}\n")
        
        with open('data_train/final_round_reward.txt', 'w') as f:
            for round_reward in self.final_round_reward:
                f.write(f"{round_reward}\n")
        # 保存每一步成功率到文件
        with open('data_train/step_success.txt', 'w') as f:
            for step_succ in self.step_success:
                f.write(f"{step_succ}\n")
        
        # 保存总成功次数到文件
        with open('data_train/total_success.txt', 'w') as f:
            for succ in self.total_success:
                f.write(f"{succ}\n")


    def accurateCalculateInverseKinematics(bodyId, endEffectorIndex, targetPos, threshold=1e-4, maxIter=100):
        numJoints = p.getNumJoints(bodyId)
        closeEnough = False
        iterCount = 0
        while not closeEnough and iterCount < maxIter:
            jointPoses = p.calculateInverseKinematics(bodyId, endEffectorIndex, targetPos)
            for i in range(numJoints):
                p.resetJointState(bodyId, i, jointPoses[i])
            actualPos = p.getLinkState(bodyId, endEffectorIndex)[4]
            diff = [a - b for a, b in zip(targetPos, actualPos)]
            dist2 = sum(d * d for d in diff)
            closeEnough = (dist2 < threshold)
            iterCount += 1
        return jointPoses
