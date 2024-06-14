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

    class RobotEnv:
        def __init__(self):
            self.state_size = 8
            self.action_size = 4
            self.max_steps = 200
            self.current_step = 0
            self.reset()
            self.pre_nr_food_collected = 0
            self.last_action = 0
            self.last_last_action = 0

        def reset(self):
            # Initialize sensor data and steps
            self.state = np.zeros(self.state_size)
            self.current_step = 0
            return self.state

        def step(self, action):
            # Execute action and update state
            if action == 0:
                rob.move_blocking(100, 100, 200)
            elif action == 1:
                rob.move_blocking(100, -100, 160)
            elif action == 2:
                rob.move_blocking(-100, 100, 160)
            elif action == 3:
                rob.move_blocking(-100, -100, 200)
            self.current_step += 1
            done = False
            reward = 0

            # Get sensor values
            irs = rob.read_irs()
            self.state = np.array(irs)

            # Reward function
            if self._is_collision():
                reward = -2
                done = True
            elif self._is_too_close():
                reward = -1
            else:
                if action != 0 and self.last_action != 0 and self.last_last_action != 0:
                    reward = -1
                elif action == 0 and self.last_action == 0:
                    reward = 2
                else:
                    reward = 1

            # Check if max steps reached
            if self.current_step >= self.max_steps:
                done = True
            self.last_last_action = self.last_action
            self.last_action = action

            return self.state, reward, done, {}

        def _is_collision(self):
            # Check collision
            collision_threshold = 4000
            return any(sensor > collision_threshold for sensor in self.state)

        def _is_too_close(self):
            # Check proximity
            collision_threshold = 200
            return any(sensor > collision_threshold for sensor in self.state)

    def train_agent(state_size, action_size, episodes, batch_size, env, agent, rob, RESULT_DIR, test_number):
        items = []
        averages = []
        origin_position = rob.get_position()
        origin_orientation = rob.get_orientation()

        # Uncomment this line if you want to load a pre-trained model
        # agent.model.load_state_dict(torch.load(RESULT_DIR / 'dqn_model.pth'))

        # Main training loop
        for e in range(episodes):
            # Uncomment these lines if using a simulation environment
            # if isinstance(rob, SimulationRobobo):
            #     rob.stop_simulation()
            #     rob.play_simulation()
            random_number = random.uniform(0, 0.1)
            random_number_1 = random.uniform(0, 6.5)

            new_position = Position(origin_position.x, origin_position.y, origin_position.z)
            new_position.x += random_number
            new_position.y += random_number
            new_orientation = Orientation(origin_orientation.yaw, origin_orientation.pitch, origin_orientation.roll)
            new_orientation.pitch += random_number_1
            rob.set_position(new_position, new_orientation)

            state = env.reset()

            for time in range(500):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    items.append(time)
                    current_average = sum(items) / len(items)
                    averages.append(current_average)
                    print(
                        f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}, average_score: {current_average:.2}")
                    print()
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

        torch.save(agent.model.state_dict(), RESULT_DIR / ('dqn_model_' + test_number + '.pth'))
        with open(RESULT_DIR / ('items_' + test_number + '.txt'), 'a') as file:
            for item in items:
                file.write(f"{item}\n")

        with open(RESULT_DIR / ('averages_' + test_number + '.txt'), 'a') as file:
            for average in averages:
                file.write(f"{average}\n")


    def test_agent(agent, env, RESULT_DIR, test_number):
        # Load the trained model
        agent.model.load_state_dict(torch.load(RESULT_DIR / ('dqn_model_' + test_number + '.pth')))

        # Reset the environment to get the initial state
        state = env.reset()
        done = False
        total_reward = 0
        count = 0

        while not done:
            # Select an action
            action = agent.act(state)

            # Perform the action
            next_state, reward, done, _ = env.step(action)

            # Update the state
            state = next_state

            # Accumulate the reward
            total_reward += reward
            if done:
                count += 1
                done = False

            if count > 10:
                done = True

        return total_reward

    state_size = 8
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    episodes = 500
    batch_size = 32
    env = RobotEnv()
    test_number = "2"

    # Uncomment this line to train the agent
    # train_agent(state_size, action_size, episodes, batch_size, env, agent, rob, RESULT_DIR, test_number)

    # Test the agent
    total_reward = test_agent
