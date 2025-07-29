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
        self.epsilon = 0.1
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(4, 16, 9)
        # load model
        self.model.load_state_dict(torch.load('./model/model.pth'))
        self.model.eval()

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

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
        # random moves: tradeoff exploration / exploitation
        
        if random.random() < self.epsilon:
            move = random.randint(0, 8)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()  # Prediction
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
            if score > record:
                record = score
                agent.model.save()
            print(f'Episode: {agent.episodes}, Record: {record}')

            plot_scores.append(score)
            total_score = total_score + score
            mean_score = total_score / agent.episodes 
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
    