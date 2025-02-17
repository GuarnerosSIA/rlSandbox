from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
import numpy as np
import matplotlib.pyplot as plt

time.sleep(5)

V = []
VK = []
g = 9.81
l = 0.5
theta0 = 90*(np.random.rand()-0.5)

client = RemoteAPIClient()
sim = client.require('sim')
sim.setStepping(True)
sim.startSimulation()

rightmotor = sim.getObject('/Base/theta1')
mass = sim.getObject('./Base/theta1/mass')
Base = sim.getObject('./Base')
    
# sim.setJointTargetPosition(rightmotor,45)
x1_a = sim.getJointTargetPosition(rightmotor)
x2_a = sim.getJointTargetVelocity(rightmotor)
# (x,y,x) = sim.getObjectPosition(robot,-1)
dt = 0.05
gammaS_a =-45
gammaS = 0
while (t := sim.getSimulationTime()) < 10:
    # print(f'Simulation time: {t:.2f} [s]')
    (alpha,beta,gamma) = np.rad2deg(sim.getObjectOrientation(rightmotor,mass))
    dotGamma = sim.getJointVelocity(rightmotor)
    # ddotGamma = -g/l*np.rad2deg(np.sin(gamma))
    Vf = 0.005*dotGamma**2 + 0.005*gamma**2
    VfK = (g/l)*(1-np.cos(np.deg2rad(gamma)))+0.57*(dotGamma**2)
    print(dotGamma)
    V.append(Vf)
    VK.append(VfK)
    sim.step()
sim.stopSimulation()



plt.plot(V)
plt.plot(VK)
plt.show()