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

    
while (t := sim.getSimulationTime()) < 20:
    print(f'Simulation time: {t:.2f} [s]')
    data_to_send = {
        "joint_position":[0.1,0.2,0.3,0.4,0.5,0.6],
        "messager": "Update joint positions"
    }
    data_string = json.dumps(data_to_send)
    sim.setStringSignal("data_from_python", data_string)
 
    sim.step()
sim.stopSimulation()