import numpy as np
import random, os, time, shutil
from tqdm import tqdm
import cv2

from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from game import Game


class Bot:
    def __init__(self, score_requirement=300, initial_games=10):
        self.game = Game()
        self.score_requirement = score_requirement
        self.initial_games = initial_games
        self.transform_map = {'w': [1, 0, 0], 's': [0, 1, 0], 'n': [0, 0, 1]}

    def initial_population(self):
        training_data = []

        for _ in tqdm(range(self.initial_games)):
            score = 0
            game_memory = []
            # Get the first observation
            prev_observation = self.game.reset()

            # Play the game
            while 1:
                action = input('')
                if action != '': action = action[0]
                else: action = np.random.choice(['n', 's'], p=[0.9, 0.1])
                if action != 's' and action != 'w': action = 'n'

                # Get params from next frame and update game_memory and score
                observation, done, reward = self.game.step(action)
                game_memory.append([prev_observation, action])
                score += reward

                prev_observation = observation
                # Break if game is over
                if done:
                    break

            # Check if score of the game is good enough and make training_data entry
            if score >= self.score_requirement:
                game_memory = game_memory[:-30]
                for data in game_memory:
                    # Make labels (one hot)
                    output = self.transform_map[data[1]]

                    training_data.append([data[0], output])
        # Save training_data
        training_data_save = np.array(training_data)
        np.save('data/saved_easy.npy', training_data_save)

        return training_data

    @staticmethod
    def neural_network_model():
        # 3 fully-connected hidden layers with 32 units and output with 3
        model = Sequential()

        model.add(Dense(1024, input_shape=(6, ), activation='relu'))

        model.add(Dense(1024, activation='relu'))

        model.add(Dense(1024, activation='relu'))

        model.add(Dense(3, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(0.00005), metrics=['categorical_accuracy'])

        return model

    def train_model(self, training_data, model=False, batch_size=32, epochs=20):
        # Make features and labels for training
        X = np.array([i[0] for i in training_data]).reshape(len(training_data), 6)
        y = np.array([i[1] for i in training_data])

        # Load model if not passed to the function
        if not model:
            model = self.neural_network_model()

        tb_callback = TensorBoard(log_dir='logs/easy', histogram_freq=0,
                                  batch_size=batch_size, write_graph=True,
                                  write_grads=True, write_images=True)

        mc_callback = ModelCheckpoint('models/model_easy.h5', save_best_only=True)

        es_callback = EarlyStopping(monitor='val_loss', patience=7)

        rp_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

        model.fit(X, y, batch_size=batch_size, validation_split=0.3, epochs=epochs,
                  callbacks=[tb_callback, mc_callback, es_callback, rp_callback])

        return model

    def play_games(self, score_req, model):
        training_data = []
        for _ in tqdm(range(100)):
            score = 0
            game_memory = []
            prev_observation = self.game.reset()

            while 1:
                # Predict the next action given a model
                action = np.random.choice(['w', 's', 'n'], p=model.predict(prev_observation.reshape((1, 6)))[0])

                observation, done, reward = self.game.step(action)
                game_memory.append([prev_observation, action])
                prev_observation = observation
                score += reward
                if done:
                    break

            # Check if score of the game is good enough and make training_data entry
            if score >= score_req:
                game_memory = game_memory[:-30]
                for data in game_memory:
                    output = self.transform_map[data[1]]

                    training_data.append([data[0], output])

        return training_data


def run(init=False, has_model=False):
    bot = Bot()

    # If there is no initial population saved, make it
    if not init:
        training_data = bot.initial_population()
    else:
        training_data = np.load('data/saved_easy.npy')

    # If there is no model saved, make it
    if not has_model:
        model = bot.train_model(training_data, batch_size=32, epochs=20)

        # Loop that trains the model until perfection
        for i in range(4):
            training_data = bot.play_games((i + 1) * 1000, model)
            model = bot.train_model(training_data, model, batch_size=128, epochs=20)
    else:
        model = load_model('models/model_easy.h5')

    # See the results
    for i in range(10):
        prev_observation = bot.game.reset()
        while 1:
            time.sleep(0.01)
            action = np.argmax(model.predict(prev_observation.reshape(1, 6)))
            if action == 0: action = 'w'
            elif action == 1: action = 's'
            else: action = 'n'
            current_observation, done, _ = bot.game.step(action)
            prev_observation = current_observation
            if done:
                break
