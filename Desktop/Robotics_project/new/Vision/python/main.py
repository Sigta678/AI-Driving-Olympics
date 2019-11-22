import vrep
import numpy as np
import sys
import time
from userscripts.DR12 import DR12_LF, DR12_VISION
from userscripts.agent import StayOnAgent
from userscripts.agent import HumanAgent
from userscripts.agent import VisualAgent
from userscripts.agent import Agent
from userscripts.sumo import Sumo

# http://fid.cl/courses/ai-robotics/vrep-tut/pythonBubbleRob.pdf
if __name__ == '__main__':
    # Finish any pending simulation
    vrep.simxFinish(-1)
    # Start the client
    clientID = vrep.simxStart('127.0.0.1', 470, True, True, 5000, 5)
    # Check if successful
    if clientID != -1:
        # Connected
        print("Connected")
    else:
        # Failed and exit
        print("Failed")
        sys.exit("Could not connect")

    # Wrap robot 1
    robot_1 = DR12_LF(clientID)
    # Attach agent to robot
    agent_1 = HumanAgent(robot_1)
    # Wrap robot 2
    robot_2 = DR12_VISION(clientID,
                      body = 'dr12_body_#0',
                      leftJoint = 'dr12_leftJoint_#0',
                      rightJoint = 'dr12_rightJoint_#0',
                      leftSensor = 'LeftSensor#0',
                      rightSensor = 'RightSensor#0',
                      visionSensor = "Vision_sensor")
    # Attach agent to robot
    agent_2 = StayOnAgent(robot_2)
    #agent_2 = VisualAgent(robot_2)
    # Start sumo match
    sumo = Sumo(agent_1, agent_2)
    # If reaches max time, quit
    sumo_maxtime = 120
    update_freq = 0.05
    max_update_count = int(sumo_maxtime / update_freq)

    winner = -1

    # Run match
    for count in range(max_update_count):
        # Update all agents
        sumo.update()
        # Check winner
        winner = sumo.getWinner()
        if winner != -1:
            if winner == 1:
                print("Robot 1 wins")
            elif winner == 2:
                print("Robot 2 wins")
            break
        # Wait for program to update
        time.sleep(update_freq)

    # Tie
    if winner == -1:
        print("Tie")

    # Stop simulation
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    vrep.simxFinish(-1)
