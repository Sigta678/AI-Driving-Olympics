import vrep
import numpy as np
import sys

import matplotlib.pyplot as mpl
import time

# http://fid.cl/courses/ai-robotics/vrep-tut/pythonBubbleRob.pdf
class DR12:

    def __init__(self, clientID):
        self.clientID = clientID
        err, self.motor_left = vrep.simxGetObjectHandle(self.clientID,"dr12_leftJoint_", vrep.simx_opmode_blocking)
        err, self.motor_right = vrep.simxGetObjectHandle(self.clientID,"dr12_rightJoint_", vrep.simx_opmode_blocking)
        errorCode, self.visionSensorHandle = vrep.simxGetObjectHandle(self.clientID, 'Vision_sensor', vrep.simx_opmode_oneshot_wait)

    def setVelocity(self, left_v, right_v):
        err = vrep.simxSetJointTargetVelocity(self.clientID, self.motor_left, left_v, vrep.simx_opmode_streaming)
        err = vrep.simxSetJointTargetVelocity(self.clientID, self.motor_right, right_v, vrep.simx_opmode_streaming)

    def getVisionSensor(self):
        # Get the image of vision sensor
        errprCode, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.visionSensorHandle, 0,
                                                                     vrep.simx_opmode_streaming)
        time.sleep(1)
        errprCode, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.visionSensorHandle, 0,
                                                                     vrep.simx_opmode_buffer)


        # Process the image to the format (64,64,3)
        sensorImage = np.array(image, dtype=np.uint8)
        sensorImage.resize([resolution[0], resolution[1], 3])

        # Use matplotlib.imshow to show the image
        mpl.imshow(sensorImage, origin='lower')
        mpl.show()

if __name__ == '__main__':

    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

    if clientID != -1:
        print("Connected")
    else:
        print("Failed")
        sys.exit("Could not connect")

    robot = DR12(clientID)





    while(True):
        robot.setVelocity(1.0, 1.0)
        robot.getVisionSensor()
