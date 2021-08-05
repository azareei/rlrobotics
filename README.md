# rlrobotics
Reinforcement Learning for Robotics multistable based legged robots.

# How to run the code ?

There is 3 main categories.

1. Controller
    Place the mapping table in the controller folder as `_all_sequences.pkl`
    install requirements `pip install -r requirements.txt`
    Run `main.py` to start the Deep Q learning process or use Visual studio code debug mode and select `Run AI`
2. Energy
    Run `energy.py` to produce the plot
3. robot
    install requirements `pip install -r requirements.txt`
    Run the selected simulation you want or the mapping table generator with debug monde on Visual Studio code.

# Code documentation

    All the documentation is inside the code, especially for the Robot section where a full sphinx style documentation 
    is created.
    

# Reinforcement learning:
The following video shows how the robot learns to maneuver and reaches the selected targets using only two sequences of A and B.


https://user-images.githubusercontent.com/6335541/128372029-25700f9b-fdf0-4664-a8fc-7a0123efefe9.mp4

