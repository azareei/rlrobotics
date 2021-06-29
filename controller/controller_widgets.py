from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.uix.widget import Widget
from kivy.vector import Vector


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

    def sensor_value(self, sensorx, sensory, width, height):
        if sensorx > width-10 or sensorx < 10 or sensory > height-10 or sensory < 10:
            return 1.
        return 0.

    def move(self, displacement, width, height):
        self.pos = Vector(displacement[3], displacement[4]).rotate(self.angle) + self.pos
        self.rotation = displacement[5]
        # self.pos = Vector(displacement[0], displacement[1]).rotate(self.angle) + self.pos
        # self.rotation = displacement[2]
        self.angle += self.rotation

        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30) % 360) + self.pos

        self.sensor4 = Vector(-30, 0).rotate(self.angle) + self.pos
        self.sensor5 = Vector(-30, 0).rotate((self.angle-30) % 360) + self.pos
        self.sensor6 = Vector(-30, 0).rotate((self.angle+30) % 360) + self.pos

        # Only gives signal if in the direction of motion
        # velocity_angle = Vector(*self.velocity).angle(Vector(1, 0))
        # print(f'{velocity_angle} {(self.angle+180) % 360 - 180}')
        self.signal1 = self.sensor_value(self.sensor1_x, self.sensor1_y, width, height)
        self.signal2 = self.sensor_value(self.sensor2_x, self.sensor2_y, width, height)
        self.signal3 = self.sensor_value(self.sensor3_x, self.sensor3_y, width, height)
        self.signal4 = self.sensor_value(self.sensor4_x, self.sensor4_y, width, height)
        self.signal5 = self.sensor_value(self.sensor5_x, self.sensor5_y, width, height)
        self.signal6 = self.sensor_value(self.sensor6_x, self.sensor6_y, width, height)


class Goal(Widget):
    pass


class SignalFront(Widget):
    pass


class SignalBack(Widget):
    pass
