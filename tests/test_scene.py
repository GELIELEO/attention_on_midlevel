import ai2thor.controller
import numpy as np

robot_scene = ["FloorPlan_Train1_2"]


controller = ai2thor.controller.Controller(agentMode='bot')

event = controller.reset(robot_scene[0])

all_visible_objects = [obj['objectType'] for obj in event.metadata['objects']]

print(len(all_visible_objects))
#
for obj in controller.last_event.metadata['objects']:
    if obj['objectType'] == 'BasketBall':
        print(obj)

'''
{'name': 'TVStand_8463b5a2', 'position': {'x': 9.559441, 'y': 0.0003528595, 'z': -1.48073769}, 
'rotation': {'x': 0.0, 'y': 270.0, 'z': 0.0}, 'cameraHorizon': 0.0, 'visible': False, 'receptacle': True, 
'toggleable': False, 'isToggled': False, 'breakable': False, 'isBroken': False, 'canFillWithLiquid': False, 
'isFilledWithLiquid': False, 'dirtyable': False, 'isDirty': False, 'canBeUsedUp': False, 'isUsedUp': False, 
'cookable': False, 'isCooked': False, 'ObjectTemperature': 'RoomTemp', 'canChangeTempToHot': False, 
'canChangeTempToCold': False, 'sliceable': False, 'isSliced': False, 'openable': False, 'isOpen': False, 
'pickupable': False, 'isPickedUp': False, 'mass': 0.0, 'salientMaterials': None, 
'receptacleObjectIds': ['AlarmClock|+09.58|+00.80|-01.78', 'Candle|+09.60|+00.80|-01.14'], 
'distance': 6.62101173, 'objectType': 'TVStand', 'objectId': 'TVStand|+09.56|+00.00|-01.48', 
'parentReceptacle': None, 'parentReceptacles': None, 'currentTime': 0.0, 'isMoving': False, 'objectBounds': None}
'''