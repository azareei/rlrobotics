from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.uix.widget import Widget
from kivy.vector import Vector
import numpy as np


class Robot(Widget):
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

    def move(self, displacement, sand, width, height):
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

        if self.sensor1_x > width-10 or self.sensor1_x < 10 or \
                self.sensor1_y > height-10 or self.sensor1_y < 10:
            self.signal1 = 1.
        if self.sensor2_x > width-10 or self.sensor2_x < 10 or \
                self.sensor2_y > height-10 or self.sensor2_y < 10:
            self.signal2 = 1.
        if self.sensor3_x > width-10 or self.sensor3_x < 10 or \
                self.sensor3_y > height-10 or self.sensor3_y < 10:
            self.signal3 = 1.
        if self.sensor4_x > width-10 or self.sensor4_x < 10 or \
                self.sensor4_y > height-10 or self.sensor4_y < 10:
            self.signal4 = 1.
        if self.sensor5_x > width-10 or self.sensor5_x < 10 or \
                self.sensor5_y > height-10 or self.sensor5_y < 10:
            self.signal5 = 1.
        if self.sensor6_x > width-10 or self.sensor6_x < 10 or \
                self.sensor6_y > height-10 or self.sensor6_y < 10:
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
