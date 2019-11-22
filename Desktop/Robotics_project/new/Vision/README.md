ECE 470 Introduction to Robotics

- Setup (You need):
  1. V-REP
  2. Python
  3. pygame
  4. matplotlib
  5. numpy

- To run the human vs simple agent sumo match:
  1. First open sumo.ttt and run the simulation
  2. "cd python"
  3. "python main.py"
  4. In the opened pygame window, use w, a, s, d to control the robot

- To run the proximity sensor test code:
  1. First open prox_test.ttt and run the simulation
  2. "cd python"
  3. "python prox_test.py"
  4. In the opened pygame window, use w, a, s, d to control the robot
  5. Watch the visualized sensor reading in matplotlib window

- Where to find everything:
  1. Most of our scripts are under userscripts folder.
  2. DR12.py wraps V-REP robot
  3. agent.py controls the robot
  4. sumo.py implement the rules of sumo competition
  5. main.py runs the sumo competition
  6. prox_test.py runs the proximity sensor simulation

- Notes:
  1. There may be some sudo issues on Linux workstations. (Personal machines run fine.)
  2. We wrapped a lot of the V-REP commands into Python classes. You can find them in DR12.py file.
