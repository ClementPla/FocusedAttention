
from utils.tools import minmax_norm, save_cv2_img, set_text
import cv2
import numpy as np
import os
import tqdm
import torch


def run_attribution(dataset, model, 
                    capt_att, 
                    foldersave,
                    cuda=True, 
                    save_raw_map=True,
                    save_colormap=True,
                    save_img=False, 
                    include_prediction=True,
                    inverse=False,
                    interpolation='nearest',
                    labels=None,
                    **kwargs):
    model.eval()
    if cuda:
        model = model.cuda()
    for i, sample in enumerate(tqdm.tqdm(dataset, total=len(dataset))):
        model.zero_grad()
        img = sample['image'].unsqueeze(0)
        b, c, h, w = img.shape
        roi = sample.get('roi', torch.ones((1, h, w)))
        if cuda: 
            img = img.cuda()
        
        pred = model(img)
        if labels is None:
            labels = {i:i for i in range(pred.shape[1])}

        pred = pred.argmax()
        img.requires_grad_()
        attributions = capt_att.attribute(img, target=pred, **kwargs)
        
        attributions = torch.nn.functional.interpolate(attributions, 
                                                      size=img.shape[2:], 
                                                      mode=interpolation)
        
        attributions = attributions.detach().cpu().numpy()
        if attributions.shape[1] > 1:
            attributions = attributions.max(1)
        roi = roi.detach().cpu().numpy()
        attributions = minmax_norm(attributions)
        if inverse:
            attributions = 1-attributions
        attributions = np.squeeze(attributions)
        attributions = np.uint8(255 * attributions)
        if save_colormap:
            heatmap = cv2.applyColorMap(attributions, cv2.COLORMAP_JET)            
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2RGBA)
            heatmap[:, :, 3] = roi*255
            if include_prediction:
                heatmap = set_text(heatmap, "Prediction: %s"%labels[int(pred)])
            save_cv2_img(heatmap, os.path.join(foldersave, 'Heatmap'), 'sample_%i.png'%i)
        if save_raw_map:
            save_cv2_img(attributions, os.path.join(foldersave, 'Mask'), 'sample_%i.png'%i)      
        if save_img:
            img = minmax_norm(img.squeeze()).detach().cpu().numpy()
            img = np.uint8(255 * img)
            save_cv2_img(img, os.path.join(foldersave, 'Image'), 'sample_%i.png'%i)

