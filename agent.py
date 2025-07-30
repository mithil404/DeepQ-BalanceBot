import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from game import Game
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 32
LR = 0.001

class Agent:

    def __init__(self):
        self.episodes = 0
        self.epsilon = 1.0  # Start with high exploration
        self.gamma = 0.99   # Increased from 0.95 for better long-term planning
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(4, 24, 9)
        self.target_model = Linear_QNet(4, 24, 9)  # Create target network
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize with same weights
        self.target_update_frequency = 100  # Update target every 100 episodes
        
        # load model
        self.model.load_state_dict(torch.load('./model/model.pth'))
        self.model.eval()

        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Decay epsilon from 1.0 to 0.1 over training
        self.epsilon = max(0.01, 1.0 - 0.9 * self.episodes / 10000)  # Assuming 10000 total episodes
        
        # random moves: tradeoff exploration / exploitation
        if random.random() < self.epsilon:
            # Exploration: choose random action
            move = random.randint(0, 8)
            final_move[move] = 1
        else:
            # Exploitation: choose best action based on model prediction
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()
    while True:
        # get old state
        state_old = game.get_state()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.episodes += 1
            agent.train_long_memory()
            
            # Update target network periodically
            if agent.episodes % agent.target_update_frequency == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
                print(f"Target network updated at episode {agent.episodes}")
                
            if score > record:
                record = score
                agent.model.save()
            print(f'Episode: {agent.episodes}, Score: {score}, Record: {record}, Epsilon: {agent.epsilon:.2f}')

            plot_scores.append(score)
            total_score = total_score + score
            mean_score = total_score / agent.episodes 
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
    