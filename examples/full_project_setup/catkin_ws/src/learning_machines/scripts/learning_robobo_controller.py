#!/usr/bin/env python3
import sys
from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions
import numpy as np
import time
# import tensorflow as tf
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
from data_files import RESULT_DIR, FIGRURES_DIR
from robobo_interface.datatypes import (
    Emotion,
    LedColor,
    LedId,
    Acceleration,
    Position,
    Orientation,
    WheelPosition,
    SoundEmotion,
)
import cv2

if __name__ == "__main__":
    # You can do better argument parsing than this!
    print(sys.argv[1], "----------------------")
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware or simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    print("start-------")
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    #
    # def find_largest_green_object(image, i):
    #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     lower_green = np.array([45, 70, 70])
    #     upper_green = np.array([85, 255, 255])
    #     mask = cv2.inRange(hsv, lower_green, upper_green)
    #
    #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     if contours:
    #         largest_contour = max(contours, key=cv2.contourArea)
    #         largest_area = cv2.contourArea(largest_contour)
    #         x, y, w, h = cv2.boundingRect(largest_contour)
    #         center_x = x + w // 2
    #         center_y = y + h // 2
    #         cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)  #
    #         cv2.imwrite(FIGRURES_DIR / f'output_image_infunction_{i}.png', image)
    #         return center_x, center_y, largest_area
    #     return 0, 0, 0
    #
    #
    # rob.set_phone_tilt_blocking(96, 100)
    # time.sleep(2)
    #
    # for i in range(50):
    #     image = rob.get_image_front()
    #     image = cv2.flip(image, 0)
    #     center_x, center_y, area = find_largest_green_object(image, i)
    #     print(i, center_x, center_y, area)
    #     rob.move_blocking(50,50,100)
    # time.sleep(10)




    # Define DQN network
    class DQN(nn.Module):
        def __init__(self, state_size, action_size):
            super(DQN, self).__init__()
            self.fc1 = nn.Linear(state_size, 24)
            self.fc2 = nn.Linear(24, 24)
            self.fc3 = nn.Linear(24, action_size)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # DQN Agent
    class DQNAgent:
        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=50000)
            self.gamma = 0.95
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.learning_rate = 0.001
            self.model = DQN(state_size, action_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss()

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def act(self, state):
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            state = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state)
            return torch.argmax(act_values[0]).item()

        def replay(self, batch_size):
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    next_state = torch.FloatTensor(next_state).unsqueeze(0)
                    target = (reward + self.gamma * torch.max(self.model(next_state)[0]).item())
                state = torch.FloatTensor(state).unsqueeze(0)
                target_f = self.model(state)
                target_f[0][action] = target
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(state), target_f)
                loss.backward()
                self.optimizer.step()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        def act_test(self, state):
            # self.model.eval()  # 将模型设置为评估模式
            state = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state)
            return torch.argmax(act_values[0]).item()

    class RobotEnv:
        def __init__(self, state_size):
            self.state_size = state_size
            self.action_size = 4
            self.max_steps = 200
            self.current_step = 0
            self.reset()
            self.pre_nr_food_collected = 0
            self.last_action = 0
            self.last_last_action = 0
            self.last_reward = 0
            self.last_reward_sum = 0
            self.move_forward_flag = False

        def reset(self):
            # Initialize sensor data and steps
            self.state = np.zeros(self.state_size)
            self.current_step = 0
            return self.state

        def step(self, action, state, total_reward):
            # Execute action and update state
            food_collected = rob.nr_food_collected()

            speed = 50
            time = 120
            if action == 0:
                rob.move_blocking(speed, speed, 3.5*time)
            elif action == 1:
                rob.move_blocking(speed, -speed, time)
            elif action == 2:
                rob.move_blocking(-speed, speed, time)
            elif action == 3:
                rob.move_blocking(-speed, -speed, 2*time)
            self.current_step += 1
            food_collected_after = rob.nr_food_collected()
            done = False
            reward = 0
            image = rob.get_image_front()
            image = cv2.flip(image, 0)
            center_x, center_y, area = find_largest_green_object(image)

            # Get sensor values
            self.irs = rob.read_irs()
            # front_irs = [self.irs[7], self.irs[2], self.irs[4], self.irs[3], self.irs[5]]
            self.state = np.array(self.irs)

            # Reward function
            if food_collected_after > food_collected:
                reward = 50
                self.move_forward_flag = False


            elif self.move_forward_flag and action == 0 and 0 < abs(center_y - state[21]) < 25:
                reward = self.last_reward + 2
                self.move_forward_flag = True

            elif action == 0 and 0 < abs(center_y - state[21]) < 25:
                reward = 2
                self.move_forward_flag = True
                print("move forward")

            elif self._is_collision():
                reward = -10
                self.move_forward_flag = False

                # done = True

            elif food_collected_after == 7:
                reward = 200
                done = True

            else:
                self.move_forward_flag = False
                reward = -3

            if total_reward < -200:
                done = True

            # Check if max steps reached
            if self.current_step >= self.max_steps:
                done = True
            self.last_last_action = self.last_action
            self.last_action = action
            self.last_reward = reward
            if self.move_forward_flag:
                self.last_reward_sum = reward
            if food_collected_after == 7:
                done = True
            print(reward)

            return self.state, reward, done, area, center_x, center_y, {}

        def step_test(self, action):
            speed = 50
            time = 120
            if action == 0:
                rob.move_blocking(speed, speed, 2 * time)
            elif action == 1:
                rob.move_blocking(speed, -speed, time)
            elif action == 2:
                rob.move_blocking(-speed, speed, time)
            elif action == 3:
                rob.move_blocking(-speed, -speed, 2 * time)

            image = rob.get_image_front()
            image = cv2.flip(image, 0)
            center_x, center_y, area = find_largest_green_object(image)

            # Get sensor values
            self.irs = rob.read_irs()
            # front_irs = [self.irs[7], self.irs[2], self.irs[4], self.irs[3], self.irs[5]]
            self.state = np.array(self.irs)
            return self.state, area, center_x, center_y, {}

        def _is_collision(self):
            # Check collision
            collision_threshold = 800
            return any(sensor > collision_threshold for sensor in self.irs)

    class Node:
        def __init__(self):
            self.state = None
            self.reward = 0  # 每个节点有一个奖励值
            self.children = []  # 子节点列表
            self.parent = None  # 父节点
            self.action = None  # 与此节点关联的动作（如果有的话）

        def add_child(self, child):
            """此方法用于添加一个子节点到当前节点"""
            self.children.append(child)  # 把新的子节点加入到子节点列表中
            child.parent = self  # 设置新子节点的父节点为当前节点


    def train_agent(state_size, action_size, episodes, batch_size, env, agent, rob, RESULT_DIR, test_number):
        items = []
        total_rewards = []
        origin_position = rob.get_position()
        origin_orientation = rob.get_orientation()

        # Uncomment this line if you want to load a pre-trained model
        # agent.model.load_state_dict(torch.load(RESULT_DIR / 'dqn_model.pth'))

        # Main training loop
        for e in range(episodes):
            # Uncomment these lines if using a simulation environment
            if isinstance(rob, SimulationRobobo):
                rob.stop_simulation()
                rob.play_simulation()
            random_number = random.uniform(0, 0.1)
            random_number_1 = random.uniform(0, 6.5)

            # new_position = Position(origin_position.x, origin_position.y, origin_position.z)
            # new_position.x += random_number
            # new_position.y += random_number
            # new_orientation = Orientation(origin_orientation.yaw, origin_orientation.pitch, origin_orientation.roll)
            # new_orientation.pitch += random_number_1
            # rob.set_position(new_position, new_orientation)
            # rob.set_phone_tilt_blocking(96, 100)

            state = env.reset()
            root = Node()
            current_node = root
            half_length = len(state) // 2
            total_reward = 0
            for time in range(500):
                # print(time)
                current_node.state = state[half_length:]
                # print(state)
                new_node = Node()
                current_node.add_child(new_node)

                action = agent.act_test(state)
                # print("action start")
                next_state_sensors, reward, done, area, center_x, center_y, _ = env.step(action, state, total_reward)
                new_node.state = np.append(next_state_sensors, [area, center_x, center_y])
                # print(new_node.state)
                next_state = np.concatenate((current_node.state, new_node.state))
                reward = reward if not done else -10
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                print("total_reward", total_reward)

                if done:
                    items.append(time)
                    current_average = sum(items) / len(items)
                    total_rewards.append(total_reward)
                    food_collected = rob.nr_food_collected()
                    print(
                        f"episode: {e}/{episodes}, score: {time}, reward: {total_reward} e: {agent.epsilon:.2}, food_number: {food_collected}")
                    print()
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

        torch.save(agent.model, RESULT_DIR / ('dqn_model_' + test_number + '.pkl'))
        with open(RESULT_DIR / ('items_' + test_number + '.txt'), 'a') as file:
            for item in items:
                file.write(f"{item}\n")

        with open(RESULT_DIR / ('total_rewards_' + test_number + '.txt'), 'a') as file:
            for total_rewards in total_rewards:
                file.write(f"{total_rewards}\n")


    def test_agent(agent, env, RESULT_DIR, test_number):
        # Load the trained model
        agent.model = torch.load(RESULT_DIR / ('dqn_model_' + test_number + '.pkl'))
        print("model loaded")
        # Reset the environment to get the initial state
        state = env.reset()
        done = False
        total_reward = 0
        time = 0
        root = Node()
        current_node = root
        # rob.set_phone_tilt_blocking(100, 100)
        half_length = len(state) // 2
        while not done:

            current_node.state = state[half_length:]
            # print(state)
            new_node = Node()
            current_node.add_child(new_node)

            action = agent.act_test(state)
            print(action)
            # print("action start")
            next_state_sensors, area, center_x, center_y, _ = env.step_test(action)
            new_node.state = np.append(next_state_sensors, [area, center_x, center_y])
            # print(new_node.state)
            next_state = np.concatenate((current_node.state, new_node.state))

            state = next_state

            time = time + 1
            # print(time)


        return total_reward


    def find_largest_green_object(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([45, 70, 70])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)  #
            # cv2.imwrite(FIGRURES_DIR / 'output_image_infunction.png', image)
            return center_x, center_y, largest_area
        return 0, 0, 0


    state_size = 22
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    episodes = 80
    batch_size = 32
    env = RobotEnv(state_size)
    test_number = "11"

    # Uncomment this line to train the agent
    # train_agent(state_size, action_size, episodes, batch_size, env, agent, rob, RESULT_DIR, test_number)

    # Test the agent

    total_reward = test_agent(agent, env, RESULT_DIR, test_number)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

