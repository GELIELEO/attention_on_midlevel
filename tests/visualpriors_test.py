from PIL import Image

import torch
import torchvision.transforms.functional as TF
import visualpriors
import subprocess
import cv2
import numpy as np

mode = ['autoencoding', 'depth_euclidean']# 'reshading', 'keypoints2d', 'edge_occlusion','curvature', 'edge_texture', 'keypoints3d', 'segment_unsup2d', 'segment_unsup25d','normal','segment_semantic', 'denoising' , 'inpainting',
    #    'class_object',
    #    'jigsaw', 'room_layout','class_scene', 'egomotion', 'nonfixated_pose','fixated_pose', 'point_matching', 'vanishing_point']
tool = 'cv'
#not impletemented: 'jigsaw', 'room_layout','class_scene', 'egomotion', 'nonfixated_pose','fixated_pose', 'point_matching', 'vanishing_point'
# mismatch: 'class_object'
# 
#'colorization',
#
#
#   
# Download a test image
# subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)

# Load image and rescale/resize to [-1,1] and 3x256x256
if tool=='pil':
    image = Image.open('./test/test.png')
    print(type(image))
    x = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
    print(x.dtype)
    print(type(x))
else:
    image=cv2.imread('./test/test.png')
    x=torch.from_numpy(image)
    print(x.dtype)
    x = TF.to_tensor(image) * 2 - 1
    # x = x.permute(2,0,1).float()* 2 - 1
    print(x.dtype)
x = x.unsqueeze_(0)

for i, m in enumerate(mode):

    try:
        representation = visualpriors.representation_transform(x, m, device='cpu')
        print(representation.shape)# torch.Size([1, 8, 16, 16])
    except:
        print(m)

    # Transform to normals feature and then visualize the readout
    pred = visualpriors.feature_readout(x, m, device='cpu')

    # Save it
    TF.to_pil_image(pred[0] / 2. + 0.5).save('test_{}_readout.png'.format(m))