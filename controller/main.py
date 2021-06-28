import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.vector import Vector

from controller_widgets import (Ball1, Ball2, Ball3, Ball4, Ball5, Ball6, Robot,
                                Goal)
from dqn import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Window.size = (1280, 720)

n_points = 0
length = 0
goal_reached_nb = 0


def load_data():
    df = pd.read_pickle(f'{Path(__file__).resolve().parent}/AB_sequences.pkl')
    # round close to zero values to zero
    df['x'] = df['x'].where(abs(df['x']) > 1e-2, 0)
    df['y'] = df['y'].where(abs(df['y']) > 1e-3, 0)
    df['yaw'] = df['yaw'].where(abs(df['yaw']) > 1e-2, 0)

    # Remove duplicates
    df = df.drop_duplicates(subset=['x', 'y', 'yaw'])

    # Rescale
    df['yaw'] = np.degrees(df['yaw'])
    df['x'] = np.multiply(df['x'], 100)
    df['y'] = np.multiply(df['y'], 100)

    # Separate the actuation phase between 0 and 180 and keep only reverse to false (remove symmetry)
    df = df[df["actuation"] == 0]

    df = df[df['reverse']]
    # df = df[~df['reverse']]
    return df.values.tolist()


list_actions = load_data()
print(f'Number of actions : {len(list_actions)}')
model = Dqn(8, len(list_actions), 0.9)
last_reward = 0
last_action = 0
last_distance = 0
last_orientation = 0
last_nb_steps = 1e5
cum_rewards = 0
scores = []

first_update = True


class Game(Widget):
    robot = Robot()
    ball1 = Ball1()
    ball2 = Ball2()
    ball3 = Ball3()
    ball4 = Ball4()
    ball5 = Ball5()
    ball6 = Ball6()
    goal = Goal()

    def serve_robot(self):
        self.robot.center = self.center
        self.robot.angle = 0
        self.robot.velocity = Vector(1, 0)

    def update(self):

        global model
        global last_reward
        global cum_rewards
        global scores
        global last_distance
        global last_orientation
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

        xx = goal_x - self.robot.x
        yy = goal_y - self.robot.y
        orientation = Vector(*self.robot.velocity).angle((xx, yy))/180.

        last_signal = [
            self.robot.signal1, self.robot.signal2, self.robot.signal3,
            self.robot.signal4, self.robot.signal5, self.robot.signal6,
            orientation, -orientation
        ]

        action = model.update(last_reward, last_signal)
        scores.append(model.score())
        displacement = list_actions[action]
        self.robot.move(displacement, width, height)
        distance = np.sqrt((self.robot.x - goal_x)**2 + (self.robot.y - goal_y)**2)
        self.ball1.pos = self.robot.sensor1
        self.ball2.pos = self.robot.sensor2
        self.ball3.pos = self.robot.sensor3
        self.ball4.pos = self.robot.sensor4
        self.ball5.pos = self.robot.sensor5
        self.ball6.pos = self.robot.sensor6
        self.goal.pos = Vector(goal_x, goal_y)

        self.steps += 1

        self.robot.velocity = Vector(1, 0).rotate(self.robot.angle)
        last_reward = (last_distance - distance) / 6

        # score based also on orientation
        #  last_reward += (1-abs(orientation)) * 0.9
        if abs(last_orientation) < abs(orientation):
            last_reward -= 0.2
        elif abs(last_orientation) == abs(orientation):
            last_reward += 0.
        else:
            last_reward += 0.2
        # Score also based on sequence change
        global last_action
        if list_actions[last_action][0] == list_actions[action][0]:
            last_reward += 0.02
        last_action = action

        if self.robot.x < 10:
            self.robot.x = 10
            last_reward = -10  # too close to edges of the wall reward
        if self.robot.x > self.width - 10:
            self.robot.x = self.width - 10
            last_reward = -10
        if self.robot.y < 10:
            self.robot.y = 10
            last_reward = -10
        if self.robot.y > self.height - 10:
            self.robot.y = self.height - 10
            last_reward = -10

        if distance < 50:
            global goal_reached_nb
            goal_reached_nb += 1
            if goal_reached_nb < 100:
                goal_x = self.width-goal_x
                goal_y = self.height-goal_y
                last_reward = self.last_steps - self.steps  # reward for reaching the objective faster than last round
            else:
                goal_x = random.randint(10, width-10)
                goal_y = random.randint(10, height-10)

            self.last_steps = self.steps
            global last_nb_steps
            last_nb_steps = self.steps
            self.steps = 0
            cum_rewards = 0

        cum_rewards += last_reward
        last_distance = distance
        global scorelabel
        scorelabel.text = 'Last run steps : {:.0f}\nReward : {:.1f}\nSequence : {}\nGoals completed : {}'.format(
            last_nb_steps,
            cum_rewards,
            list_actions[action][0],
            goal_reached_nb
        )


def init():
    global goal_x
    global goal_y
    global first_update
    global scorelabel
    goal_x = 50
    goal_y = height - 50
    first_update = False


class RobotApp(App):

    def build(self):
        parent = Game()
        parent.serve_robot()
        Clock.schedule_interval(parent.update, 1.0/120.0)
        savebtn = Button(text='save', pos=(parent.width, 0))
        loadbtn = Button(text='load', pos=(2 * parent.width, 0))
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)

        global scorelabel
        scorelabel = Label(
            text="Score",
            pos=(0.3 * parent.width, 2 * parent.height),
            halign="left",
            size_hint=(1.0, 1.0)
        )
        scorelabel.bind(size=scorelabel.setter('text_size'))
        parent.add_widget(scorelabel)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def save(self, obj):
        model.save()
        print("Saved model")
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        model.load()
        print("Loaded model")


# Running the whole thing
if __name__ == '__main__':
    RobotApp().run()
