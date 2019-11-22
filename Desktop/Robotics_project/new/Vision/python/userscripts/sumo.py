import vrep

class Sumo():

    def __init__(self, agent1, agent2, falling_z = 0.0):
        self.agent1 = agent1
        self.agent2 = agent2
        self.falling_z = falling_z   

    # -1 if still not done yet, 1 if robot 1 wins, 2 if robot 2 wins
    def getWinner(self):
        robot1_z = self.agent1.getRobotPosition()[2]
        robot2_z = self.agent2.getRobotPosition()[2]
        robot1_fall = (robot1_z <= self.falling_z)
        robot2_fall = (robot2_z <= self.falling_z)
        if robot1_fall and robot2_fall:
            if robot1_z < robot2_z:
                return 2
            else:
                return 1
        elif robot1_fall:
            return 2
        elif robot2_fall:
            return 1
        else:
            return -1
        
    def update(self):
        self.agent1.update()
        self.agent2.update(self)
