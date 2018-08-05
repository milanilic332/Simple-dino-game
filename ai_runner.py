"""
Bot class that plays the game and learns how to play it from given params
    initial_population - makes initial population from random actions with probabilities (0.1, 0.45, 0.45)
    weighted_categorical_crossentropy - probability of jump is 0.1 and for other actions 0.45 so loss must be weighted
    neural_network_model - makes the neural network and compiles it
    train_model - trains the model with training data
    play_games - plays 100 games given a model to predict and returns new training_data
"""
import numpy as np
import random
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from keras import backend as K
from chrome_game.game import Game


class Bot:
    def __init__(self, score_requirement=500, initial_games=10000):
        self.game = Game()
        self.score_requirement = score_requirement
        self.initial_games = initial_games

    @staticmethod
    def weighted_categorical_crossentropy(weights):
        weights = K.variable(weights)

        def loss(y_true, y_pred):
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss

        return loss

    def initial_population(self):
        training_data = []
        accepted_scores = []

        for it in range(self.initial_games):
            print(it)
            score = 0
            game_memory = []
            # Get the first observation
            prev_observation = self.game.reset()

            # Play the game
            while 1:
                # Get random action
                k = random.uniform(0, 1)
                if k < 0.1:
                    action = 'w'
                elif k < 0.55:
                    action = 's'
                else:
                    action = 'n'
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
                print(score)
                accepted_scores.append(score)
                for data in game_memory:
                    # Make labels (one hot)
                    if data[1] == 'w':
                        output = [1, 0, 0]
                    elif data[1] == 's':
                        output = [0, 1, 0]
                    else:
                        output = [0, 0, 1]

                    training_data.append([data[0], output])
        # Save training_data
        training_data_save = np.array(training_data)
        np.save('data/saved_easy.npy', training_data_save)

        return training_data

    def neural_network_model(self):
        # 3 fully-connected hidden layers with 32 units and output with 3
        model = Sequential()

        model.add(Dense(32, input_shape=(6, ), activation='relu'))

        model.add(Dense(32, activation='relu'))

        model.add(Dense(32, activation='relu'))

        model.add(Dense(3, activation='softmax'))

        # Make custom loss function
        loss = self.weighted_categorical_crossentropy(np.array([4.5, 1.0, 1.0]))

        model.compile(loss=loss, optimizer='adam', metrics=['categorical_accuracy'])

        # Save model architecture as json (can't load it if you save it other way)
        with open('data/model_easy.json', 'w') as f:
            f.write(model.to_json())

        return model

    def train_model(self, training_data, model=False):
        # Make features and labels for training
        X = np.array([i[0] for i in training_data]).reshape(len(training_data), 6)
        y = np.array([i[1] for i in training_data])

        # Load model if not passed to the function
        if not model:
            model = self.neural_network_model()

        # Fit data and save weights
        model.fit(X, y, batch_size=128, epochs=3)
        model.save_weights('data/model_weights_easy.h5')

        return model

    def play_games(self, score_req, model):
        training_data = []
        accepted_scores = []

        for it in range(100):
            print(it)
            score = 0
            game_memory = []
            prev_observation = self.game.reset()

            while 1:
                # Predict the next action given a model
                action = model.predict(prev_observation.reshape((1, 6)))
                action = np.argmax(action[0])
                if action == 0:
                    action = 'w'
                elif action == 1:
                    action = 's'
                else:
                    action = 'n'
                observation, done, reward = self.game.step(action)
                game_memory.append([prev_observation, action])
                prev_observation = observation
                score += reward
                if done:
                    break

            # Check if score of the game is good enough and make training_data entry
            if score >= score_req:
                print(score)
                accepted_scores.append(score)
                for data in game_memory:
                    if data[1] == 'w':
                        output = [1, 0, 0]
                    elif data[1] == 's':
                        output = [0, 1, 0]
                    else:
                        output = [0, 0, 1]

                    training_data.append([data[0], output])

        return training_data


def main(init=False, model=False):
    bot = Bot()

    # If there is no initial population saved, make it
    if not init:
        training_data = bot.initial_population()
    else:
        training_data = np.load('data/saved_easy.npy')

    # If there is no model saved, make it
    if not model:
        model = bot.train_model(training_data)

        # Loop that trains the model until perfection
        for i in range(4):
            training_data = bot.play_games(bot.score_requirement + (i + 1) * 1000, model)
            model = bot.train_model(training_data, model)
    else:
        model = model_from_json(open('data/model_easy.json').read())
        model.load_weights('data/model_weights_easy.h5')
        loss = bot.weighted_categorical_crossentropy(np.array([4.5, 1.0, 1.0]))
        model.compile(loss=loss, optimizer='adam', metrics=['categorical_accuracy'])

    # See the results
    for i in range(10):
        prev_observation = bot.game.reset()
        while 1:
            action = np.argmax(model.predict(prev_observation.reshape(1, 6)))
            if action == 0:
                action = 'w'
            elif action == 1:
                action = 's'
            else:
                action = 'n'
            current_observation, done, _ = bot.game.step(action)
            prev_observation = current_observation
            if done:
                break


if __name__ == '__main__':
    main(init=True, model=True)
