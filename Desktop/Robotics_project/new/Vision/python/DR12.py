import vrep
import vrepConst
import numpy as np
import time
import matplotlib.pyplot as mpl

# http://fid.cl/courses/ai-robotics/vrep-tut/pythonBubbleRob.pdf
class DR12:

    def __init__(self, clientID, body = "dr12_body_", leftJoint = "dr12_leftJoint_", rightJoint = "dr12_rightJoint_"):
        # Save parameters
        self.clientID = clientID
        self.leftJoint = leftJoint
        self.rightJoint = rightJoint
        # Get handle of joints and sensors
        err, self.body_handle = vrep.simxGetObjectHandle(self.clientID, body, vrep.simx_opmode_blocking)
        err, self.motor_left = vrep.simxGetObjectHandle(self.clientID, leftJoint, vrep.simx_opmode_blocking)
        err, self.motor_right = vrep.simxGetObjectHandle(self.clientID, rightJoint, vrep.simx_opmode_blocking)

    def wheelFK(self, left_w, right_w):
        # Reference: http://www.cs.columbia.edu/~allen/F17/NOTES/icckinematics.pdf
        # Forward Kinematics
        R_wheel = 86 # [mm]
        L_between_wheels = 164 # [mm]
        R_of_rotation = 0.5*L_between_wheels*(left_w + right_w) / (right_w - left_w)
        ICC_vector = [R_of_rotation,0,0]
        w = [0,0,R_wheel*(right_w - left_w) / L_between_wheels]
        v = np.cross(w,ICC_vector)

        return (v, w)

    def wheelIK(self, v_body, w_body):
        left_w = 0
        right_w = 0
        # TO-DO
        # Inverse Kinematics
        return (left_w, right_w)

    def setWheelVelocity(self, left_w, right_w):
        err = vrep.simxSetJointTargetVelocity(self.clientID, self.motor_left, left_w, vrep.simx_opmode_streaming)
        err = vrep.simxSetJointTargetVelocity(self.clientID, self.motor_right, right_w, vrep.simx_opmode_streaming)

    def getWheelVelocity(self):
        err, left_w = vrep.simxGetObjectFloatParameter(self.clientID, self.motor_left,
                                                       vrepConst.sim_jointfloatparam_velocity, vrep.simx_opmode_blocking)
        err, right_w = vrep.simxGetObjectFloatParameter(self.clientID, self.motor_right,
                                                        vrepConst.sim_jointfloatparam_velocity, vrep.simx_opmode_blocking)

        return (left_w, right_w)

    def setVelocity(self, v_body, w_body):
        left_w, right_w = self.wheelIK(v_body, w_body)
        self.setWheelVelocity(left_w, right_w)

    def getVelocity(self):
        left_w, right_w = self.getWheelVelocity()
        v, w = self.wheelFK(left_w, right_w)
        return (v, w)

    def getVelocityVREP(self):
        err, v, w = vrep.simxGetObjectVelocity(self.clientID, self.body_handle, vrep.simx_opmode_blocking)
        return (v, w)

    def getPosition(self):
        return vrep.simxGetObjectPosition(self.clientID, self.body_handle, -1, vrep.simx_opmode_blocking)[1]


# Line follower DR12
class DR12_LF(DR12):

    def __init__(self, clientID, body = "dr12_body_", leftJoint = "dr12_leftJoint_", rightJoint = "dr12_rightJoint_",
                 leftSensor = "LeftSensor", rightSensor = "RightSensor"):
        DR12.__init__(self, clientID, body, leftJoint, rightJoint)
        # Save parameters
        self.leftSensor = leftSensor
        self.rightSensor = rightSensor
        # Get handle of extra sensors
        err, self.sensor_left = vrep.simxGetObjectHandle(self.clientID, leftSensor, vrep.simx_opmode_blocking)
        err, self.sensor_right = vrep.simxGetObjectHandle(self.clientID, rightSensor, vrep.simx_opmode_blocking)

    def getLineSensors(self):
        leftReading = vrep.simxReadVisionSensor(self.clientID, self.sensor_left, vrep.simx_opmode_blocking)[1]
        rightReading = vrep.simxReadVisionSensor(self.clientID, self.sensor_right, vrep.simx_opmode_blocking)[1]
        return (leftReading, rightReading)

class DR12_PROX(DR12_LF):

    def __init__(self, clientID, body = "dr12_body_", leftJoint = "dr12_leftJoint_", rightJoint = "dr12_rightJoint_",
                 leftSensor = "LeftSensor", rightSensor = "RightSensor", proximitySensor = "Proximity_sensor"):
        DR12_LF.__init__(self, clientID, body, leftJoint, rightJoint, leftSensor, rightSensor)
        self.proximitySensor = proximitySensor
        self.firstTime = 0
        # Get handle of proximity sensor
        errorCode, self.proxHandle = vrep.simxGetObjectHandle(clientID, proximitySensor, vrep.simx_opmode_blocking)


    def getProximitySensor(self):
        if self.firstTime == 0:
            returnCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector = vrep.simxReadProximitySensor(self.clientID,self.proxHandle,vrep.simx_opmode_streaming)
            self.firstTime += 1
        else:
            returnCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector = vrep.simxReadProximitySensor(self.clientID,self.proxHandle,vrep.simx_opmode_buffer)
        return (detectionState, detectedPoint)

class DR12_VISION(DR12_LF):
    def __init__(self, clientID, body = "dr12_body_", leftJoint = "dr12_leftJoint_", rightJoint = "dr12_rightJoint_",
                 leftSensor = "LeftSensor", rightSensor = "RightSensor", visionSensor = "Vision_sensor"):
        DR12_LF.__init__(self, clientID, body, leftJoint, rightJoint, leftSensor, rightSensor)
        self.visionSensor = visionSensor
        err, self.visionSensor = vrep.simxGetObjectHandle(self.clientID, visionSensor, vrep.simx_opmode_blocking)

    def getVisionSensor(self):
        _, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.visionSensor, 0, vrep.simx_opmode_streaming)
        time.sleep(0.1)
        _, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.visionSensor, 0, vrep.simx_opmode_buffer)

        sensorImage = np.array(image, dtype=np.uint8)
        image_1 = sensorImage.flatten()
        sensorImage.resize([resolution[0], resolution[1], 3])

        mpl.imshow(sensorImage, origin='lower')
        return image_1, sensorImage