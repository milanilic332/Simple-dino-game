"""
Objects used for game.
Player - rectangle that can jump, duck and do nothing
Floor - static rectangle at the bottom of the screen
Obstacle - one random chosen rectangle out of four
"""

import random


class Player:
    def __init__(self, canvas):
        self.canvas = canvas
        self.id = self.canvas.create_rectangle((10, 125, 35, 175), fill='yellow')
        self.y = 0
        # Hardcoded jumping values for efficiency (~sin)
        self.jumping = [-11, -10, -11, -10, -9, -8, -7, -7, -5,  -5,  -3,  -3, -1,
                        0, 1, 3, 3, 5, 5, 7, 7, 8, 9, 10, 11, 10, 11]
        self.jumping_iter = 0
        self.jumping_param = False
        self.ducking_param = False

    # Resets the game to start over
    def reset(self):
        pos = self.canvas.coords(self.id)
        self.canvas.move(self.id, 0, 125 - pos[1])
        self.jumping_param = False
        self.ducking_param = False
        self.jumping_iter = 0

    # Returns coordinates of players current position
    def get_coords(self):
        return self.canvas.coords(self.id)

    # Checks if a player can duck
    def check_duck(self):
        if not self.jumping_param:
            pos = self.canvas.coords(self.id)
            self.canvas.move(self.id, 0, 125 - pos[1])
            self.ducking_param = True

    # Duck for one frame
    def duck(self):
        if self.ducking_param:
            pos = self.canvas.coords(self.id)
            self.canvas.move(self.id, 0, 150 - pos[1])
            self.ducking_param = False

    # Checks if a player can jump
    def check_jump(self):
        if not self.jumping_param:
            pos = self.canvas.coords(self.id)
            self.canvas.move(self.id, 0, 125 - pos[1])
            self.jumping_param = True

    # Jump
    def jump(self):
        if self.jumping_param and not self.ducking_param:
            if self.jumping_iter != len(self.jumping):
                self.y = self.jumping[self.jumping_iter]
                self.jumping_iter += 1
            else:
                self.jumping_iter = 0
                self.jumping_param = False

    # Moves the player for the next frame
    def draw(self, action, drawing):
        pos = self.canvas.coords(self.id)
        drawing.rectangle((pos[0], pos[1], pos[2], pos[3]), fill=255)
        if action == 's':
            self.check_duck()
        elif action == 'w':
            self.check_jump()
        self.duck()
        self.jump()
        self.canvas.move(self.id, 0, self.y)
        self.y = 0


class Floor:
    def __init__(self, canvas):
        self.canvas = canvas
        self.id = self.canvas.create_rectangle((0, 175, 800, 200), fill='brown')


class Obstacle:
    def __init__(self, canvas, subclass, speed=None):
        self.canvas = canvas
        self.id = None
        # Picks the start x coordinate for obstacle
        k = random.randint(800, 820) + (0 if speed is None else (speed - 5))
        # Makes the obstacle
        if subclass == 1:
            self.id = self.canvas.create_rectangle((k, 120, k + 40, 140), fill='red')
        elif subclass == 2:
            self.id = self.canvas.create_rectangle((k, 140, k + 30, 175), fill='red')
        elif subclass == 3:
            self.id = self.canvas.create_rectangle((k, 145, k + 40, 175), fill='red')
        else:
            self.id = self.canvas.create_rectangle((k, 80, k + 40, 110), fill='red')

    # Returns coordinates of obstacle
    def get_coords(self):
        return self.canvas.coords(self.id)

    # Moves obstacle
    def draw(self, drawing, movement_speed):
        pos = self.canvas.coords(self.id)
        drawing.rectangle((pos[0], pos[1], pos[2], pos[3]), fill=255)
        self.canvas.move(self.id, -movement_speed, 0)

    # Removes the obstacle
    def remove(self):
        self.canvas.delete(self.id)
