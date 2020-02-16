import ai2thor.controller
# import ai2thor.robot_controller
import random
import time
import numpy as np
from pprint import pprint

thor_scene = ["FloorPlan311"]
robot_scene = ["FloorPlan_Train1_1", "FloorPlan_Train1_3"]

runs = [
    {'id': 'thor', 'port':8200, 'controller': ai2thor.controller.Controller},
    {'id': 'robot', 'port':9200,  'controller': ai2thor.controller.Controller}
]

for run_config in runs:
    id = run_config['id']
    port = run_config['port']
    if id=='thor':
        controller = run_config['controller'](port=port,scene=thor_scene[0], gridSize=0.25, rotateStepDegrees=1) # if not specify the scene argument when using thor, scene in robot_scene wiil be used by default.
    else:
        controller = run_config['controller'](port=port, agentMode='bot', agentType='stochastic', gridSize=0.25, applyActionNoise=True, rotateStepDegrees=1)

    for scene in thor_scene:
        print(scene)
        event = controller.reset(scene)# not necessarily
        for i in range(10):
            # event = controller.step(dict(action='Initialize', gridSize=0.25, fieldOfView=90, renderObjectImage=True))
            # event = controller.step(dict(action='InitialRandomSpawn', forceVisible=True, maxNumRepeats=10, randomSeed=1))
            # event = controller.step(dict(action='MoveAhead', noise=0.02))
            event = controller.step(dict(action='RotateLeft'))
            # print("event for '{}':".format(run_config['id']))
            # pprint(event.metadata)
            time.sleep(1)