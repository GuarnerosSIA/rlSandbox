from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
import numpy as np

time.sleep(1)

alpha =[]
alphaDot =[]
xDot = []
d =[]
gammaAngle = []
gammaPoint =[]

r = 0.1
l = 0.15
xd = 3*(np.random.rand()-0.5)
yd = 3*(np.random.rand()-0.5)

client = RemoteAPIClient()
sim = client.require('sim')
sim.setStepping(True)
sim.startSimulation()

rightmotor = sim.getObject('/PioneerP3DX/rightMotor')
leftMotor = sim.getObject('./PioneerP3DX/leftMotor')
robot = sim.getObject('./PioneerP3DX')
    
sim.setObjectPosition( robot,(0.5,-0.5,0.5),-1)
(x,y,x) = sim.getObjectPosition(robot,-1)

while (t := sim.getSimulationTime()) < 20:
    # print(f'Simulation time: {t:.2f} [s]')
    (x,y,z) = sim.getObjectPosition(robot,-1)
    (dummy,beta,theta) = sim.getObjectOrientation(robot,-1)
    dx = xd-x
    dy = yd-y
    distance =math.sqrt(dx*dx + dy*dy)
    (x,y,z) = sim.getObjectPosition(robot,-1)
    angle = -math.degrees(theta)+math.degrees(math.atan2(dy,dx))
    vel = distance*0.05
    angVel = angle*0.1
    print(vel,angVel)
    if distance > 0.1:
        rl = (2*vel+angVel*l)/(2*r)
        ll = (2*vel-angVel*l)/(2*r)
    else:
        rl = 0
        ll = 0    
    sim.setJointTargetVelocity(rightmotor,rl)
    sim.setJointTargetVelocity(leftMotor,ll)
    sim.step()
sim.stopSimulation()



