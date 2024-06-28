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
            self.epsilon_1 = 1.0
            self.epsilon_2 = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay_1 = 0.995
            self.epsilon_decay_2 = 0.997
            self.learning_rate = 0.001
            self.model = DQN(state_size, action_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss()

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def act(self, state, current_task):
            print("current_task", current_task)
            if current_task == 1:
                if np.random.rand() <= self.epsilon_1:
                    return random.randrange(self.action_size)
            elif current_task == 2:
                if np.random.rand() <= self.epsilon_2:
                    return random.randrange(self.action_size)
            state = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state)
            return torch.argmax(act_values[0]).item()

        def replay(self, batch_size, current_task):
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
            if current_task == 1:
                if self.epsilon_1 > self.epsilon_min:
                    self.epsilon_1 *= self.epsilon_decay_1
            elif current_task == 2:
                if self.epsilon_2 > self.epsilon_min:
                    self.epsilon_2 *= self.epsilon_decay_2

        def act_test(self, state):
            self.model.eval()  # 将模型设置为评估模式
            state = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state)
            return torch.argmax(act_values[0]).item()

    class RobotEnv:
        def __init__(self, state_size, collision_threshold):
            self.state_size = state_size
            self.max_steps = 100
            self.current_step = 0
            self.reset()
            self.pre_nr_food_collected = 0
            self.last_action = 0
            self.last_last_action = 0
            self.last_reward = 0
            self.last_reward_sum = 0
            self.move_forward_flag = False
            self.collision_threshold = collision_threshold
            self.food_flag = False

        def reset(self):
            # Initialize sensor data and steps
            self.state = np.zeros(self.state_size)
            self.current_step = 0
            return self.state

        def step(self, action, state, current_task, time_):
            # Execute action and update state
            speed = 50
            move_time = 140
            if current_task == 1:
                if action == 0:
                    rob.move_blocking(speed, speed, 3*move_time)
                elif action == 1:
                    rob.move_blocking(speed, -speed, move_time)
                elif action == 2:
                    rob.move_blocking(-speed, speed, move_time)
            elif current_task == 2:
                if action == 0:
                    rob.move_blocking(speed, speed, 3*move_time)
                elif action == 1:
                    rob.move_blocking(speed, 0, 400)
                elif action == 2:
                    rob.move_blocking(0, speed, 400)
            self.current_step += 1
            done = False
            image = rob.get_image_front()
            if current_task == 1:
                center_x, center_y, area = find_red_object(image)
            elif current_task == 2:
                center_x, center_y, area = find_largest_green_object(image)

            # Get sensor values
            self.irs = rob.read_irs()
            irs = self.irs
            front_irs = [self.irs[7], self.irs[2], self.irs[4], self.irs[3], self.irs[5]]
            self.irs = [self.normalization(irs) for irs in self.irs]  # normalization
            self.state = np.array(self.irs)

            # Reward function
            if current_task == 1:
                if self.are_three_sensors_greater_than_180(irs):
                    reward = -30
                    done = True
                elif irs[4] > 2000 and self.is_only_one_greater_than_2000(irs):
                    reward = 30
                    current_task = 2

                elif action == 0 and 126 < center_y < 386:
                    reward = area * 100 / (512*512)
                    if reward > 10:
                        reward = 10

                elif action == 0 and self.irs[4] > 0.15:
                    reward = self.irs[4] * 10
                    if reward > 10:
                        reward = 10
                else:
                    reward = -2
            elif current_task == 2:
                # print(center_x, center_y, area, area/(512*512))
                # cv2.imwrite(FIGRURES_DIR / f'output_image_infunction_{time_}.png', image)

                if self.are_three_sensors_greater_than_180(irs):
                    reward = -40
                    done = True
                elif center_y > 460 and 106 < center_x < 406:
                    reward = 50
                    done = True
                    rob.move_blocking(speed, speed, 3 * move_time)

                elif action == 0:
                    reward = area * 200/ (512 * 512)
                    if reward > 10:
                        reward = 10

                else:
                    reward = -3

            # Check if max steps reached
            if self.current_step >= self.max_steps:
                # reward = -100
                done = True
            self.last_last_action = self.last_action
            self.last_action = action
            self.last_reward = reward
            if self.move_forward_flag:
                self.last_reward_sum = reward
            if done:
                current_task = 1

            print("reward", reward)

            return self.state, reward/10, done, area/(512*512), center_x/512, center_y/512, current_task, {}

        def step_test(self, action, state, current_task, steps):
            speed = 50
            move_time = 150
            if current_task == 1:
                if action == 0:
                    rob.move_blocking(speed, speed, 3 * move_time)
                elif action == 1:
                    rob.move_blocking(speed, -speed, move_time)
                elif action == 2:
                    rob.move_blocking(-speed, speed, move_time)
            elif current_task == 2:
                if action == 0:
                    rob.move_blocking(speed, speed, 3 * move_time)
                elif action == 1:
                    rob.move_blocking(speed, 0, 400)
                elif action == 2:
                    rob.move_blocking(0, speed, 400)
            self.current_step += 1
            time.sleep(0.5)
            done = False
            image = rob.get_image_front()
            x, y, w, h = 0, 0, 480, 512
            # 裁剪图像
            image = image[y:y + h, x:x + w]
            if current_task == 1:
                center_x, center_y, area = find_red_object(image)
            elif current_task == 2:
                center_x, center_y, area = find_largest_green_object(image)

            # Get sensor values
            self.irs = rob.read_irs()
            irs = self.irs
            front_irs = [self.irs[7], self.irs[2], self.irs[4], self.irs[3], self.irs[5]]
            self.irs = [self.normalization(irs) for irs in self.irs]  # normalization
            self.state = np.array(self.irs)

            # Reward function
            print(current_task)
            print(center_x, center_y, area)
            print(front_irs)
            if current_task == 1:
                if 400> center_x > 260 and center_y > 450:
                    rob.move_blocking(50, 50, 350)
                    print("move forward")
                    time.sleep(2)
                if self.are_three_sensors_greater_than_180(irs):
                    reward = -30
                    # done = True
                elif irs[4] > 2000 and self.is_only_one_greater_than_2000(irs):

                    reward = 30
                    current_task = 2

                elif action == 0 and 126 < center_y < 386:
                    reward = area * 100 / (512 * 512)
                    if reward > 10:
                        reward = 10

                elif action == 0 and self.irs[4] > 0.15:
                    reward = self.irs[4] * 10
                    if reward > 10:
                        reward = 10
                else:
                    reward = -2
            elif current_task == 2:
                if self.are_three_sensors_greater_than_180(irs):
                    reward = -40
                    # done = True
                elif center_y > 460 and 106 < center_x < 406:
                    reward = 50
                    done = True
                    rob.move_blocking(speed, speed, 3 * move_time)

                elif action == 0:
                    reward = area * 200 / (512 * 512)
                    if reward > 10:
                        reward = 10

                else:
                    reward = -3

            if self.current_step >= 300:
                done = True
            self.last_last_action = self.last_action
            self.last_action = action
            self.last_reward = reward
            if self.move_forward_flag:
                self.last_reward_sum = reward
            if done:
                current_task = 1

            print("reward", reward)

            return self.state, reward / 10, done, area / (512 * 512), center_x / 512, center_y / 512, current_task, {}

        def step_test_hardware(self, action, state, current_task, steps):
            speed = 50
            move_time = 180
            if current_task == 1:
                if action == 0:
                    rob.move_blocking(speed, speed, 2 * move_time)
                elif action == 1:
                    rob.move_blocking(speed, -speed, move_time)
                elif action == 2:
                    rob.move_blocking(-speed, speed, move_time)
            elif current_task == 2:
                if action == 0:
                    rob.move_blocking(speed, speed, 2 * move_time)
                elif action == 1:
                    rob.move_blocking(speed, 0, 600)
                elif action == 2:
                    rob.move_blocking(0, speed, 600)
            self.current_step += 1
            time.sleep(0.5)
            done = False
            image = rob.get_image_front()
            x, y, w, h = 0, 0, 480, 512
            # 裁剪图像
            image = image[y:y + h, x:x + w]
            if current_task == 1:
                center_x, center_y, area = find_red_object_save_image(image, steps)
            elif current_task == 2:
                center_x, center_y, area = find_largest_green_object_save_image(image, steps)

            # Get sensor values
            self.irs = rob.read_irs()
            irs = self.irs
            front_irs = [self.irs[7], self.irs[2], self.irs[4], self.irs[3], self.irs[5]]
            self.irs = [self.normalization(irs) for irs in self.irs]  # normalization
            self.state = np.array(self.irs)

            # Reward function
            print("current task:", current_task)
            print(center_x, center_y, area)
            if current_task == 1:
                if 400> center_x > 260 and center_y > 450:
                    rob.move_blocking(50, 50, 350)
                if self.are_three_sensors_greater_than_180(irs):
                    reward = -30
                    # done = True
                elif irs[4] > 2000 and self.is_only_one_greater_than_2000(irs):

                    reward = 30
                    current_task = 2

                elif action == 0 and 126 < center_y < 386:
                    reward = area * 100 / (480 * 512)
                    if reward > 10:
                        reward = 10

                elif action == 0 and self.irs[4] > 0.15:
                    reward = self.irs[4] * 10
                    if reward > 10:
                        reward = 10
                else:
                    reward = -2
            elif current_task == 2:
                if self.are_three_sensors_greater_than_180(irs):
                    reward = -40
                    # done = True
                elif center_y > 460 and 106 < center_x < 406:
                    reward = 50
                    done = True
                    rob.move_blocking(speed, speed, 3 * move_time)

                elif action == 0:
                    reward = area * 200 / (480 * 512)
                    if reward > 10:
                        reward = 10

                else:
                    reward = -3

            if self.current_step >= 300:
                done = True
            self.last_last_action = self.last_action
            self.last_action = action
            self.last_reward = reward
            if self.move_forward_flag:
                self.last_reward_sum = reward
            if done:
                current_task = 1

            print("reward", reward)

            return self.state, reward / 10, done, area / (480 * 512), center_x / 480, center_y / 512, current_task, {}

        def _is_collision(self):
            # Check collision

            return any(sensor == 1 for sensor in self.irs)

        def normalization(self, data):

            if data < self.collision_threshold:
                data = data / self.collision_threshold

            else:
                data = 1

            return data

        def is_only_one_greater_than_2000(self, lst):
            count = 0
            for num in lst:
                if num > 2000:
                    count += 1
                if count > 1:
                    return False
            return count == 1

        def are_three_sensors_greater_than_180(self, irs):
            count = 0
            for sensor in irs:
                if sensor > 180:
                    count += 1
                if count >= 3:
                    return True
            return False

        def are_four_sensors_greater_than_180(self, irs):
            count = 0
            for sensor in irs:
                if sensor > 180:
                    count += 1
                if count >= 4:
                    return True
            return False


    class Node:
        def __init__(self):
            self.state = None
            self.reward = 0  # Each node has a reward value
            self.children = []  # List of child nodes
            self.parent = None  # Parent node
            self.action = None  # Action associated with this node (if any)

        def add_child(self, child):
            """This method is used to add a child node to the current node"""
            self.children.append(child)  # Add the new child node to the list of child nodes
            child.parent = self  # Set the new child node's parent to the current node


    def train_agent(state_size, action_size, episodes, batch_size, env, agent, rob, RESULT_DIR, test_number):
        current_task = 1
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
            random_number = random.uniform(-0.1, 0.1)
            random_number_1 = random.uniform(-0.3, 0.3)
            #
            new_position = Position(origin_position.x, origin_position.y, origin_position.z)
            new_position.x += random_number
            new_position.y += random_number
            new_orientation = Orientation(origin_orientation.yaw, origin_orientation.pitch, origin_orientation.roll)
            new_orientation.pitch += random_number_1
            rob.set_position(new_position, new_orientation)
            rob.set_phone_tilt_blocking(106, 100)
            time.sleep(0.5)

            state = env.reset()
            root = Node()
            current_node = root
            half_length = len(state) // 2
            total_reward = 0
            for time_ in range(500):
                # print(time)
                current_node.state = state[half_length:]
                # print(state)
                new_node = Node()
                current_node.add_child(new_node)

                action = agent.act(state, current_task)
                # print("action start")
                next_state_sensors, reward, done, area, center_x, center_y, current_task, _ = env.step(action, state, current_task, time_)

                new_node.state = np.append(next_state_sensors, [area, center_x, center_y, current_task])
                next_state = np.concatenate((current_node.state, new_node.state))

                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    items.append(time_)
                    total_rewards.append(total_reward)
                    if current_task == 1:
                        print(
                            f"episode: {e}/{episodes}, time: {time_}, reward: {total_reward}, e1: {agent.epsilon_1:.2}, e2: {agent.epsilon_2:.2}")
                        print()
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size, current_task)

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
        rob.set_phone_tilt_blocking(100, 100)

        time.sleep(0.5)
        # Reset the environment to get the initial state
        current_task = 1
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        root = Node()
        current_node = root
        # rob.set_phone_tilt_blocking(96, 100)
        half_length = len(state) // 2
        while not done:

            current_node.state = state[half_length:]
            # print(state)
            new_node = Node()
            current_node.add_child(new_node)

            action = agent.act_test(state)
            # print("action start")
            next_state_sensors, reward, done, area, center_x, center_y, current_task, _ = env.step_test(action, state,
                                                                                                   current_task, steps)

            new_node.state = np.append(next_state_sensors, [area, center_x, center_y, current_task])
            # print(new_node.state)
            next_state = np.concatenate((current_node.state, new_node.state))

            state = next_state

            steps = steps + 1
            total_reward += reward
            # print(time)
        print("total reward", total_reward, "steps",  steps)


        return total_reward

    def test_agent_hardware(agent, env, RESULT_DIR, test_number):
        # Load the trained model
        agent.model = torch.load(RESULT_DIR / ('dqn_model_' + test_number + '.pkl'))
        print("model loaded")
        rob.set_phone_tilt_blocking(106, 100)

        time.sleep(0.5)
        # Reset the environment to get the initial state
        current_task = 1
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        root = Node()
        current_node = root
        # rob.set_phone_tilt_blocking(96, 100)
        half_length = len(state) // 2
        while not done:

            current_node.state = state[half_length:]
            # print(state)
            new_node = Node()
            current_node.add_child(new_node)

            action = agent.act_test(state)
            # print("action start")
            next_state_sensors, reward, done, area, center_x, center_y, current_task, _ = env.step_test_hardware(action, state,
                                                                                                   current_task, steps)

            new_node.state = np.append(next_state_sensors, [area, center_x, center_y, current_task])
            # print(new_node.state)
            next_state = np.concatenate((current_node.state, new_node.state))

            state = next_state

            steps = steps + 1
            total_reward += reward
            # print(time)
        print("total reward", total_reward, "steps",  steps)


        return total_reward
    def find_red_object(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # 根据阈值构建掩模
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(image, largest_contour, -1, (0, 255, 0), 3)
            largest_area = cv2.contourArea(largest_contour)
            if largest_area > 50:

                x, y, w, h = cv2.boundingRect(contours[0])
                center_x = x + w // 2
                center_y = y + h // 2
                return center_x, center_y, largest_area
            else:
                return 0, 0, 0
        return 0, 0, 0

    def find_red_object_save_image(image, steps):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # 根据阈值构建掩模
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(image, largest_contour, -1, (0, 255, 0), 3)
            largest_area = cv2.contourArea(largest_contour)
            if largest_area > 50:

                x, y, w, h = cv2.boundingRect(contours[0])
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)
                cv2.imwrite(FIGRURES_DIR / f'output_image_hardware_{steps}.png', image)
                return center_x, center_y, largest_area
            else:
                return 0, 0, 0
        return 0, 0, 0

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
            # cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)  #
            # cv2.imwrite(FIGRURES_DIR / f'output_image_infunction.png', image)
            return center_x, center_y, largest_area
        return 0, 0, 0

    def find_largest_green_object_save_image(image, steps):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([45, 70, 70])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(image, largest_contour, -1, (0, 255, 0), 3)
            largest_area = cv2.contourArea(largest_contour)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)  #
            cv2.imwrite(FIGRURES_DIR / f'output_image_hardware_{steps}.png', image)
            return center_x, center_y, largest_area
        return 0, 0, 0

    def retrain_agent(state_size, action_size, episodes, batch_size, env, agent, rob, RESULT_DIR, test_number):
        current_task = 1
        items = []
        total_rewards = []
        agent.model = torch.load(RESULT_DIR / ('dqn_model_' + test_number + '.pkl'))

        # Main training loop
        for e in range(episodes):
            # Uncomment these lines if using a simulation environment
            if isinstance(rob, SimulationRobobo):
                rob.stop_simulation()
                rob.play_simulation()
            rob.set_phone_tilt_blocking(106, 100)
            time.sleep(0.5)

            state = env.reset()
            root = Node()
            current_node = root
            half_length = len(state) // 2
            total_reward = 0
            for time_ in range(500):
                # print(time)
                current_node.state = state[half_length:]
                # print(state)
                new_node = Node()
                current_node.add_child(new_node)

                action = agent.act(state, current_task)
                # print("action start")
                next_state_sensors, reward, done, area, center_x, center_y, current_task, _ = env.step(action, state, current_task, time_)

                new_node.state = np.append(next_state_sensors, [area, center_x, center_y, current_task])
                next_state = np.concatenate((current_node.state, new_node.state))

                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    items.append(time)
                    total_rewards.append(total_reward)
                    if current_task == 1:
                        print(
                            f"episode: {e}/{episodes}, time: {time_}, reward: {total_reward}, e1: {agent.epsilon_1:.2}, e2: {agent.epsilon_2:.2}")
                        print()
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size, current_task)

        torch.save(agent.model, RESULT_DIR / ('dqn_model_' + test_number + '.pkl'))
        with open(RESULT_DIR / ('time_' + test_number + '.txt'), 'a') as file:
            for item in items:
                file.write(f"{item}\n")

        with open(RESULT_DIR / ('total_rewards_' + test_number + '.txt'), 'a') as file:
            for total_rewards in total_rewards:
                file.write(f"{total_rewards}\n")


    state_size = 24
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    episodes = 100

    batch_size = 64
    collision_threshold = 1000
    env = RobotEnv(state_size, collision_threshold)
    test_number = "Task3_4_hardware"

    # Uncomment this line to train the agent
    # train_agent(state_size, action_size, episodes, batch_size, env, agent, rob, RESULT_DIR, test_number)

    # Test the agent

    total_reward = test_agent(agent, env, RESULT_DIR, test_number)

    # total_reward = test_agent_hardware(agent, env, RESULT_DIR, test_number)

    # retrain_agent(state_size, action_size, episodes, batch_size, env, agent, rob, RESULT_DIR, test_number)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

