import numpy as np
import torch
import torchvision.transforms.functional as TF
import visualpriors


def visual_priors(x, mode:list, cat=False)->list:
    # x: np.array or PIL img (3 dims)
    x = np.ascontiguousarray(x)
    
    # use method to_tensor in torch can avoid encoding error
    x = torch.from_numpy(x) * 2 - 1
    # x = x * 2 - 1

    x = x.unsqueeze_(0)
    print('The shape of input to visual_prior ===============', x.shape) #batch, channel, height, width
    # print(type(x))
    representations = []
    for m in mode:
        representation = visualpriors.representation_transform(x, m, device='cpu').numpy()
        representations.append(representation)
        
        '''
        # Transform to normals feature and then visualize the readout
        try:
            pred = visualpriors.feature_readout(x, m, device='cpu')
            TF.to_pil_image(pred[0] / 2. + 0.5).save('./img/test_{}_readout.png'.format(m))
        except Exception as e:
            print(e)
        '''

        # print(representation.shape)#1,8,19,19
        # print(pred.shape)#1,3/1,304,304
    # print(representation)
    # return torch.cat(representations, dim=0) if cat else representations
    return np.concatenate(representations, axis=0) if cat else representations #(4 dims)