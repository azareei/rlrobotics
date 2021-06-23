from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Importing the Kivy packages
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.graphics import Color, Line
from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.vector import Vector
from kivy.core.window import Window

# Importing the Dqn object from our AI in ai.py
from dqn import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Window.size = (1200, 800)

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
goal_reached_nb = 0


# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
# x, y, yaw
def load_data():
    df = pd.read_pickle(f'{Path(__file__).resolve().parent}/_all_sequences.pkl')
    # round close to zero values to zero
    df['x'] = df['x'].where(abs(df['x']) > 1e-3, 0)
    df['y'] = df['y'].where(abs(df['y']) > 1e-3, 0)
    df['yaw'] = df['yaw'].where(abs(df['yaw']) > 1e-2, 0)

    # Remove duplicates
    df = df.drop_duplicates(subset=['x', 'y', 'yaw'])

    # Rescale
    df['yaw'] = np.degrees(df['yaw'])
    df['x'] = np.multiply(df['x'], 100)
    df['y'] = np.multiply(df['y'], 100)
    return df.values.tolist()


action2rotation = load_data()
# action2rotation = [
#     [6, 0, 0],
#     [0, 0, 20],
#     [0, 0, -20]
# ]
print(f'Number of actions : {len(action2rotation)}')
brain = Dqn(8, len(action2rotation), 0.9)
last_reward = 0
scores = []

# Initializing the map
first_update = True

last_distance = 0


def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((width, height))
    goal_x = 30
    goal_y = height - 30
    first_update = False


class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    sensor4_x = NumericProperty(0)
    sensor4_y = NumericProperty(0)
    sensor4 = ReferenceListProperty(sensor4_x, sensor4_y)
    sensor5_x = NumericProperty(0)
    sensor5_y = NumericProperty(0)
    sensor5 = ReferenceListProperty(sensor5_x, sensor5_y)
    sensor6_x = NumericProperty(0)
    sensor6_y = NumericProperty(0)
    sensor6 = ReferenceListProperty(sensor6_x, sensor6_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)
    signal4 = NumericProperty(0)
    signal5 = NumericProperty(0)
    signal6 = NumericProperty(0)

    def move(self, displacement):
        self.pos = Vector(displacement[3], displacement[4]).rotate(self.angle) + self.pos
        self.rotation = displacement[5]
        # self.pos = Vector(displacement[0], displacement[1]).rotate(self.angle) + self.pos
        # self.rotation = displacement[2]
        self.angle = self.angle + self.rotation

        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30) % 360) + self.pos

        self.sensor4 = Vector(-30, 0).rotate(self.angle) + self.pos
        self.sensor5 = Vector(-30, 0).rotate((self.angle-30) % 360) + self.pos
        self.sensor6 = Vector(-30, 0).rotate((self.angle+30) % 360) + self.pos

        self.signal1 = int(
            np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10,
                        int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(
            np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10,
                        int(self.sensor2_y)-10:int(self.sensor2_y) + 10])) / 400.
        self.signal3 = int(
            np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10,
                        int(self.sensor3_y)-10:int(self.sensor3_y) + 10])) / 400.

        self.signal4 = int(
            np.sum(sand[int(self.sensor4_x)-10:int(self.sensor4_x)+10,
                        int(self.sensor4_y)-10:int(self.sensor4_y)+10]))/400.
        self.signal5 = int(
            np.sum(sand[int(self.sensor5_x)-10:int(self.sensor5_x)+10,
                        int(self.sensor5_y)-10:int(self.sensor5_y) + 10])) / 400.
        self.signal6 = int(
            np.sum(sand[int(self.sensor6_x)-10:int(self.sensor6_x)+10,
                        int(self.sensor6_y)-10:int(self.sensor6_y) + 10])) / 400.

        if self.sensor1_x > width-10 or self.sensor1_x < 10 or self.sensor1_y > height-10 or self.sensor1_y < 10:
            self.signal1 = 1.
        if self.sensor2_x > width-10 or self.sensor2_x < 10 or self.sensor2_y > height-10 or self.sensor2_y < 10:
            self.signal2 = 1.
        if self.sensor3_x > width-10 or self.sensor3_x < 10 or self.sensor3_y > height-10 or self.sensor3_y < 10:
            self.signal3 = 1.
        if self.sensor4_x > width-10 or self.sensor4_x < 10 or self.sensor4_y > height-10 or self.sensor4_y < 10:
            self.signal4 = 1.
        if self.sensor5_x > width-10 or self.sensor5_x < 10 or self.sensor5_y > height-10 or self.sensor5_y < 10:
            self.signal5 = 1.
        if self.sensor6_x > width-10 or self.sensor6_x < 10 or self.sensor6_y > height-10 or self.sensor6_y < 10:
            self.signal6 = 1.


class Goal(Widget):
    pass


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


class Ball4(Widget):
    pass


class Ball5(Widget):
    pass


class Ball6(Widget):
    pass

# Creating the game class


class Game(Widget):
    car = Car()
    ball1 = Ball1()
    ball2 = Ball2()
    ball3 = Ball3()
    ball4 = Ball4()
    ball5 = Ball5()
    ball6 = Ball6()
    goal = Goal()

    def serve_car(self):
        self.car.center = self.center
        self.car.angle = 0
        self.car.velocity = Vector(1, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global width
        global height

        width = self.width
        height = self.height
        if first_update:
            self.steps = 0
            self.last_steps = 0
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy))/180.

        last_signal = [
            self.car.signal1, self.car.signal2, self.car.signal3,
            self.car.signal4, self.car.signal5, self.car.signal6,
            orientation, -orientation
        ]

        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        displacement = action2rotation[action]
        self.car.move(displacement)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        self.ball4.pos = self.car.sensor4
        self.ball5.pos = self.car.sensor5
        self.ball6.pos = self.car.sensor6
        self.goal.pos = Vector(goal_x, goal_y)

        self.steps += 1

        # print(f'{distance} {scores[-1]}')
        if sand[int(self.car.x), int(self.car.y)] > 0:
            last_reward = -100  # sand reward
            self.car.velocity = Vector(0.01, 0).rotate(self.car.angle)
        else:  # otherwise
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1 * abs(last_distance - distance) / 6
            if distance < last_distance:
                last_reward = 1 * abs(last_distance - distance) / 6
        # score based also on orientation
        last_reward += (1-abs(orientation)) * 0.9

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -10  # too close to edges of the wall reward
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -10
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -10
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -10

        if distance < 50:
            global goal_reached_nb
            goal_reached_nb += 1
            if goal_reached_nb < 10:
                goal_x = self.width-goal_x
                goal_y = self.height-goal_y
            else:
                goal_x = random.randint(10, width-10)
                goal_y = random.randint(10, height-10)
            last_reward = self.last_steps - self.steps  # reward for reaching the objective faster than last round
            self.last_steps = self.steps
            self.steps = 0
        last_distance = distance

# Adding the painting tools


class MyPaintWidget(Widget):
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10:int(touch.x) + 10, int(touch.y) - 10:int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/120.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        savebtn = Button(text='save', pos=(parent.width, 0))
        loadbtn = Button(text='load', pos=(2 * parent.width, 0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((width, height))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()


# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
