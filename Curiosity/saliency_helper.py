import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize 

##Some of the code have been Shamelessly taken from https://github.com/greydanus/visualize_atari/blob/master/saliency.py. 
def get_mask(center, size, r):
        y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
        keep = x*x + y*y <= 1
        mask = np.zeros(size) ; 
        mask[keep] = 1 
        mask = gaussian_filter(mask, sigma=r) 
        return mask/mask.max()

def get_perturbed_image(image, mask):
    return image*(1-mask) + gaussian_filter(image, sigma = 5)*mask

def get_policy(model, image, p_image):
    image = [image]

    image = torch.Tensor(image).float().cuda()
    p_image = [p_image]

    p_image = torch.FloatTensor(p_image).cuda()

    policy_pImage, _ = model(p_image)
    policy_image, _ = model(image)

    return policy_pImage, policy_image

def get_value(model, image, p_image):
    image = [image]

    image = torch.Tensor(image).float().cuda()
    p_image = [p_image]
    p_image = torch.FloatTensor(p_image).cuda()
    
    _, value_pImage = model(p_image)
    _, value_image = model(image)

    return value_pImage, value_image

def saliency_score(model, image, d, mode ='actor'):
    scores = np.zeros((int(84/d)+1,int(84/d)+1))
    p_image = image.copy()
    for i in range(0,84,d):
        for j in range(0,84,d):
            mask = get_mask([i,j], [84,84], 3)
            p_image[0,:,:] = get_perturbed_image(image[0], mask)
            p_image[1,:,:] = get_perturbed_image(image[1], mask)
            p_image[2,:,:] = get_perturbed_image(image[2], mask)
            p_image[3,:,:] = get_perturbed_image(image[3], mask)
            if mode == 'actor':
                L,l = get_policy(model,image, p_image)
            elif mode == 'critic':
                L,l = get_value(model, image, p_image)

            scores[int(i/d),int(j/d)] = (L-l).pow(2).sum().mul_(.5)

    pmax = scores.max()
    scores = imresize(scores, size=[84,84], interp='bilinear').astype(np.float32)
    return pmax*scores / scores.max()

def saliency_on_frame(saliency, atari_frame, fudge_factor, channel, sigma):

    saliency = imresize(saliency, size=[160,160], interp='bilinear').astype(np.float32)
    p_a = saliency.max()

    saliency = gaussian_filter(saliency, sigma)
    saliency -= saliency.min()  
    saliency = fudge_factor* p_a * saliency / saliency.max()
    frame = atari_frame.astype('uint16')
    frame[35:195,:,channel] += saliency.astype('uint16')
    frame = frame.clip(1,255).astype('uint8')

    return frame





