"""
Game class
    next_frame - returns the next frame and param that tells if the game is over
    reset - resets the game and returns the start frame
    step - returns the next frame params, param if the game is over and reward
"""
from tkinter import Canvas, Tk
import numpy as np
from PIL import Image, ImageDraw
import random
import time
from objects import Player, Floor, Obstacle


class Game:
    def __init__(self):
        self.tk = Tk()
        self.tk.title('Chrome')
        self.tk.resizable(0, 0)
        self.canvas = Canvas(self.tk, width=800, height=200, bd=0, highlightthickness=0)
        self.canvas.config(bg='black')
        self.canvas.pack()

        self.player = Player(self.canvas)
        self.floor = Floor(self.canvas)
        self.obstacles = []
        self.current_speed = 5
        self.frame_number = 0

    def next_frame(self, action):
        done = False
        # Do we have to remove the first obstacle from obstacles
        elim = False

        # Creating image and drawing with PIL
        img = Image.new('L', (800, 200), 0)
        drawing = ImageDraw.Draw(img)

        # Creating new random obstacle and change speed of the game
        if self.frame_number % 40 == 0:
            if self.current_speed < 25:
                self.current_speed += 0.5
            else:
                print('Game beat!')
                done = True
            self.obstacles.append(Obstacle(self.canvas, random.choice([1, 2, 3, 4]), self.current_speed))

        # Checking collision
        for i in range(len(self.obstacles)):
            obs_pos = self.obstacles[i].get_coords()
            player_pos = self.player.get_coords()

            # Getting coords of corners of player and obstacles
            obs_corners = [(obs_pos[0], obs_pos[1]), (obs_pos[2], obs_pos[1]),
                           (obs_pos[0], obs_pos[3]), (obs_pos[2], obs_pos[3])]
            player_corners = [(player_pos[0], player_pos[1]), (player_pos[2], player_pos[1]),
                              (player_pos[0], player_pos[3]), (player_pos[2], player_pos[3])]
            for corner in obs_corners:
                if player_pos[0] <= corner[0] <= player_pos[2] and player_pos[1] <= corner[1] <= player_pos[3]:
                    done = True
            for corner in player_corners:
                if obs_pos[0] <= corner[0] <= obs_pos[2] and obs_pos[1] <= corner[1] <= obs_pos[3]:
                    done = True

            # Removing obstacles or drawing them
            if obs_pos[2] < -1:
                self.obstacles[i].remove()
                elim = True
            else:
                self.obstacles[i].draw(drawing, self.current_speed)

        # Drawing a player
        self.player.draw(action, drawing)
        # Updating screen
        self.tk.update_idletasks()
        self.tk.update()

        # Remove obstacle if needed
        if elim:
            self.obstacles = self.obstacles[1:]

        return np.array(img)[:175, :], done

    def reset(self, mode='easy'):
        # Reseting player and params
        self.player.reset()
        self.frame_number = 0
        self.current_speed = 13

        # Removing obstacles
        for i in range(len(self.obstacles)):
            self.obstacles[i].remove()
        self.obstacles = []

        # Getting start frame (first action is jump)
        start_frame, _, _ = self.step('n', mode)
        return start_frame

    def step(self, action, mode='easy'):
        # Get params of the next frame
        cur_x, done = self.next_frame(action)

        # Calculating reward
        if done:
            reward = -500/self.frame_number
        else:
            reward = 0.01*self.frame_number

        # Increment frame_number (score)
        self.frame_number += 1

        if mode == 'hard':
            return cur_x, done, reward

        # Players current position
        pos_player = self.player.get_coords()

        player_y = (pos_player[1] - 85)/90
        # Ensure that there are no errors
        obstacle_distance = -1
        obstacle_x = -1
        obstacle_y = -1
        min_y = -1
        try:
            # Get params and normalize them
            if self.obstacles[0].get_coords()[2] - pos_player[0] > 0:
                pos_obstacle = self.obstacles[0].get_coords()

                obstacle_distance = (pos_player[2] - pos_obstacle[0])/765
                obstacle_x = (pos_obstacle[2] - pos_obstacle[0])/40
                obstacle_y = (pos_obstacle[3] - pos_obstacle[1])/40
                min_y = (pos_obstacle[1] - 80)/65
            elif len(self.obstacles) > 1:
                pos_obstacle = self.obstacles[1].get_coords()

                obstacle_distance = (pos_player[2] - pos_obstacle[0])/765
                obstacle_x = (pos_obstacle[2] - pos_obstacle()[0])/40
                obstacle_y = (pos_obstacle[3] - pos_obstacle[1])/40
                min_y = (pos_obstacle[1] - 80)/65
        except:
            pass

        speed = (self.current_speed - 5)/20

        return np.array([player_y, obstacle_distance, obstacle_x, obstacle_y, min_y, speed]), done, reward
