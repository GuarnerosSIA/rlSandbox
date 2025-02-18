from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
import numpy as np
import json

time.sleep(1)


client = RemoteAPIClient(port=23001)
sim = client.require('sim')
sim.setStepping(True)
sim.startSimulation()

joint_handles = [
    sim.getObject('/stewartPlatform/motor1'),
    sim.getObject('/stewartPlatform/motor2'),
    sim.getObject('/stewartPlatform/motor3'),
    sim.getObject('/stewartPlatform/motor4'),
    sim.getObject('/stewartPlatform/motor5'),
    sim.getObject('/stewartPlatform/motor6')
]


while (t := sim.getSimulationTime()) < 20:
    print(f'Simulation time: {t:.2f} [s]')
    joint_positions = [sim.getJointPosition(joint) for joint in joint_handles]
    print(f'Joint positions: {joint_positions}')
    value = int(np.sin(t)*500+500)
    sim.setStringSignal("data_1", str(value))
    value = int(np.cos(t)*500+500)
    sim.setStringSignal("data_2", str(value))
    value = int(np.sin(t)*500+500)
    sim.setStringSignal("data_3", str(value))
    value = int(np.cos(t)*500+500)
    sim.setStringSignal("data_4", str(value))
    value = int(np.sin(t)*500+500)
    sim.setStringSignal("data_5", str(value))
    value = int(np.cos(t)*500+500)
    sim.setStringSignal("data_6", str(value))
    sim.step()
sim.stopSimulation()