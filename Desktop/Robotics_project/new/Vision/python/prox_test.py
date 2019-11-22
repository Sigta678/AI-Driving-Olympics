import vrep
import sys
import matplotlib.pyplot as plt
import time
import numpy as np
from userscripts.DR12 import DR12_PROX
from userscripts.agent import HumanAgent

vrep.simxFinish(-1) # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP

if clientID != -1:
    print("Connected to remote API Server")
else:
    print("Connection not successful")
    sys.exit("Could not connect")

robot = DR12_PROX(clientID)
agent = HumanAgent(robot)

plt.show()

try:
    while(True):
        detectionState, detectedPoint = robot.getProximitySensor()
        detectedPoint = np.asarray([-detectedPoint[0], detectedPoint[2]])
        origin = np.asarray([0, 0])
        plt.clf()
        plt.xlim(-0.3, 0.3)
        plt.ylim(-0.3, 0.3)
        if detectionState and (detectedPoint[0]) != origin[0] and (detectedPoint[1] != origin[1]):
            plt.arrow(origin[0], origin[0], detectedPoint[0], detectedPoint[1])
        plt.draw()
        plt.pause(1e-15)
        agent.update()
        time.sleep(0.05)
except KeyboardInterrupt:
    # Stop simulation
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    vrep.simxFinish(-1)
    sys.exit(0)

# # Get Motor Handles
# errorCode, leftJointHandle = vrep.simxGetObjectHandle(clientID,"dr12_leftJoint_",vrep.simx_opmode_blocking)
# errorCode, rightJointHandle = vrep.simxGetObjectHandle(clientID,"dr12_rightJoint_",vrep.simx_opmode_blocking)
#
# """
# # Move Robot
# vel = 1
# errorCode = vrep.simxSetJointTargetVelocity(clientID,leftJointHandle,vel,vrep.simx_opmode_streaming)
# errorCode = vrep.simxSetJointTargetVelocity(clientID,rightJointHandle,vel,vrep.simx_opmode_streaming)
# """
#
# # Get Sensor Handle
# errorCode, proxHandle = vrep.simxGetObjectHandle(clientID,"Proximity_sensor",vrep.simx_opmode_blocking)
#
# # Get Obstacle Handle
# errorCode, boxHandle = vrep.simxGetObjectHandle(clientID,"Cuboid",vrep.simx_opmode_blocking)
#
# # Read Prox Sensor
# returnCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector = vrep.simxReadProximitySensor(clientID,proxHandle,vrep.simx_opmode_streaming)

#returnCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector = vrep.simxReadProximitySensor(clientID,proxHandle,vrep.simx_opmode_buffer)
