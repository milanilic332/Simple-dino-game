import numpy as np
import random, time, os, shutil
from tqdm import tqdm
import cv2

from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten, concatenate
from keras.optimizers import Adam
from keras.models import Sequential, load_model, Input, Model
from keras import backend as K
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from game import Game


class Bot:
    def __init__(self, score_requirement=800, initial_games=20):
        self.game = Game()
        self.score_requirement = score_requirement
        self.initial_games = initial_games

    def initial_population(self):
        for it in tqdm(range(self.initial_games)):
            score = 0
            game_memory = []

            # Get the first observation
            prev_observation = self.game.reset('hard')
            prev_observation = cv2.resize(prev_observation, (400, 175))
            # Play the game
            while 1:
                action = input('')
                if action != '': action = action[0]
                else: action = np.random.choice(['n', 's'], p=[0.9, 0.1])
                if action != 's' and action != 'w': action = 'n'

                # Get params from next frame and update game_memory and score
                observation, done, reward = self.game.step(action, 'hard')
                observation = cv2.resize(observation, (400, 175))
                frame = np.array(observation.astype(np.float) - prev_observation.astype(np.float))
                frame[frame > 0] = 255
                frame[frame == 0] = 127
                frame[frame < 0] = 0
                game_memory.append([frame, action])
                score += reward

                prev_observation = observation
                # Break if game is over
                if done:
                    break

            # Check if score of the game is good enough and make training_data entry
            if score >= self.score_requirement:
                # Removing frames that cause an error
                game_memory = game_memory[:-30]
                for i, (a, b) in enumerate(game_memory):
                    cv2.imwrite('data/images/initial/' + b + '/' + str(it) + '_' + str(i) + '.jpg', a)

    @staticmethod
    def neural_network_hard():
        # 3 fully-connected hidden layers with 32 units and output with 3
        input = Input(shape=(175, 400, 1))

        tower1 = Conv2D(16, (1, 1), activation='relu', padding='same')(input)
        tower1 = Conv2D(16, (3, 3), activation='relu', padding='same')(tower1)

        tower2 = Conv2D(16, (1, 1), activation='relu', padding='same')(input)
        tower2 = Conv2D(16, (5, 5), activation='relu', padding='same')(tower2)

        tower3 = Conv2D(16, (1, 1), activation='relu', padding='same')(input)
        tower3 = Conv2D(16, (7, 7), activation='relu', padding='same')(tower3)

        x = concatenate([tower1, tower2, tower3], axis=-1)

        x = MaxPool2D((3, 3))(x)

        tower1 = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
        tower1 = Conv2D(32, (3, 3), activation='relu', padding='same')(tower1)

        tower2 = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
        tower2 = Conv2D(32, (5, 5), activation='relu', padding='same')(tower2)

        x = concatenate([tower1, tower2], axis=-1)

        x = MaxPool2D((3, 3))(x)

        x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)

        x = MaxPool2D((3, 3))(x)

        x = Flatten()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.4)(x)

        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.4)(x)

        x = Dense(3, activation='softmax')(x)

        model = Model(inputs=input, outputs=x)

        optimizer = Adam(0.0001)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

        return model

    def train_model(self, batch_size=16, epochs=10, model=False, dir='initial'):
        # Load model if not passed to the function
        if not model:
            model = self.neural_network_hard()

        tb_callback = TensorBoard(log_dir='logs/hard', histogram_freq=0,
                                  batch_size=32, write_graph=True,
                                  write_grads=True, write_images=True)

        mc_callback = ModelCheckpoint('models/saved_model_hard.h5', save_best_only=True)

        es_callback = EarlyStopping(monitor='val_loss', patience=7)

        rp_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

        # ImageDataGenerator for train images (using data augmentation)
        datagen= ImageDataGenerator(rescale=1./255, validation_split=0.2, brightness_range=(0.8, 1.2))

        generator_train = datagen.flow_from_directory(directory='data/images/' + dir,
                                                      target_size=(175, 400),
                                                      color_mode='grayscale',
                                                      classes=['w', 'n', 's'],
                                                      class_mode='categorical',
                                                      batch_size=batch_size,
                                                      subset='training')

        generator_valid = datagen.flow_from_directory(directory='data/images/' + dir,
                                                      target_size=(175, 400),
                                                      color_mode='grayscale',
                                                      classes=['w', 'n', 's'],
                                                      class_mode='categorical',
                                                      batch_size=batch_size,
                                                      subset='validation')

        # Fit data and save weights
        model.fit_generator(generator_train,
                            steps_per_epoch=generator_train.samples // batch_size,
                            validation_data=generator_valid,
                            validation_steps=generator_valid.samples // batch_size,
                            epochs=epochs,
                            callbacks=[tb_callback, mc_callback, es_callback, rp_callback])

        return model

    def play_games(self, score_req, model):
        shutil.rmtree('data/images/after/w')
        shutil.rmtree('data/images/after/s')
        shutil.rmtree('data/images/after/n')
        os.mkdir('data/images/after/w')
        os.mkdir('data/images/after/s')
        os.mkdir('data/images/after/n')
        for it in tqdm(range(50)):
            score = 0
            game_memory = []
            prev_observation = self.game.reset('hard')
            prev_observation = cv2.resize(prev_observation, (400, 175))
            frame = prev_observation - prev_observation
            while 1:
                # Predict the next action given a model
                action = np.argmax(model.predict(frame.reshape((1, 175, 400, 1)))[0])
                if action == 0: action = 'w'
                elif action == 1: action = 's'
                else: action = 'n'

                game_memory.append([frame, action])

                observation, done, reward = self.game.step(action, 'hard')

                observation = cv2.resize(observation, (400, 175))
                frame = np.array(observation.astype(np.float) - prev_observation.astype(np.float))
                frame[frame > 0] = 255
                frame[frame == 0] = 127
                frame[frame < 0] = 0

                prev_observation = observation
                score += reward

                if done:
                    break

            # Check if score of the game is good enough and make training_data entry
            if score >= score_req:
                game_memory = game_memory[:-30]

                for i, (a, b) in enumerate(game_memory):
                    cv2.imwrite('data/images/after/' + b + '/' + str(it) + '_' + str(i) + '.jpg', a)


def run(init=False, has_model=False):
    bot = Bot()

    if not init:
        bot.initial_population()

    # If there is no model saved, make it
    if not has_model:
        model = bot.train_model(batch_size=32, epochs=10, dir='initial')

        # Loop that tr  ains the model until perfection
        for i in range(2):
            bot.play_games((i + 1) * 2000, model)
            model = bot.train_model(batch_size=32, epochs=3, model=model, dir='after')
    else:
        model = load_model('models/saved_model_hard.h5')

    # See the results
    for i in range(10):
        prev_observation = bot.game.reset('hard')
        prev_observation = cv2.resize(prev_observation, (400, 175))
        frame = prev_observation - prev_observation
        while 1:
            time.sleep(0.01)
            action = np.argmax(model.predict(frame.reshape(1, 175, 400, 1)))

            if action == 0: action = 'w'
            elif action == 1: action = 's'
            else: action = 'n'

            observation, done, _ = bot.game.step(action, 'hard')
            observation = cv2.resize(observation, (400, 175))
            frame = np.array(observation.astype(np.float) - prev_observation.astype(np.float))
            frame[frame > 0] = 255
            frame[frame == 0] = 127
            frame[frame < 0] = 0
            prev_observation = observation

            if done:
                break
