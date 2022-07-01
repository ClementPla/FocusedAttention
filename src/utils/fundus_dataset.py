import nntools.dataset as D
import albumentations as A
import os
import cv2
import numpy as np
import nntools

DR_score = {
0: "No DR",
1: "Mild",
2: "Moderate",
3: "Severe",
4: "Proliferative DR"
}


@D.nntools_wrapper
def autocrop(image, mask=None):
    blur = 5
    threshold = 20
    threshold_img = cv2.blur(image, (blur, blur), borderType=cv2.BORDER_REPLICATE)
    if threshold_img.ndim == 3:
        threshold_img = np.mean(threshold_img, axis=2)

    roi = threshold_img > threshold
    not_null_pixels = np.nonzero(roi)

    x_range = (np.min(not_null_pixels[1]), np.max(not_null_pixels[1]))
    y_range = (np.min(not_null_pixels[0]), np.max(not_null_pixels[0]))
    d = {'image': image[y_range[0]:y_range[1], x_range[0]:x_range[1]],
         'roi': roi[y_range[0]:y_range[1], x_range[0]:x_range[1]].astype(np.uint8)}
    if mask is not None:
        d['mask'] = mask[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    return d

@D.nntools_wrapper
def process_masks(Exudates=None, Microaneurysms=None, Hemorrhages=None, Cotton_Wool_Spot=None):
    stacks = []
    def add_to_stack(*args):
        for arg in args:
            if arg is not None:
                stacks.append(arg > 0)
        return np.stack(tuple(stacks), 2)
    masks = {'mask': add_to_stack(Cotton_Wool_Spot, Exudates, Hemorrhages, Microaneurysms).astype(np.uint8)} 
    return masks

def sort_func_idrid(x):
    return '_'.join(x.split('.')[0].split('_')[:2])

def get_masks_paths(ex, ctw, he, ma):
    masks = {}
    masks['Exudates'] = ex
    masks['Cotton_Wool_Spot'] = ctw
    masks['Hemorrhages'] = he
    masks['Microaneurysms'] = ma
    return masks


def get_idrid_dataset(imgs_path, mask_path, shape,
                      mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    ma = os.path.join(mask_path, '1. Microaneurysms/')
    he = os.path.join(mask_path, '2. Haemorrhages/')
    ex = os.path.join(mask_path, '3. Hard Exudates/')
    ctw = os.path.join(mask_path, '4. Soft Exudates/')
        
    masks = get_masks_paths(ex, ctw, he, ma)

    segmentation_dataset = D.SegmentationDataset(imgs_path, masks, shape, keep_size_ratio=True, 
                                                filling_strategy=nntools.NN_FILL_UPSAMPLE,
                                                extract_image_id_function=sort_func_idrid)
    composer = D.Composition()

    preprocess = A.Compose([A.LongestMaxSize(max(shape), always_apply=True),
                            A.PadIfNeeded(*shape, value=[0, 0, 0],
                                          border_mode=cv2.BORDER_CONSTANT, always_apply=True,
                                          mask_value=0)], additional_targets={'roi':'mask'})
    normalization = A.Normalize(mean=mean, std=std,
                                    always_apply=True)
    
    composer << process_masks << autocrop << preprocess << normalization
    segmentation_dataset.set_composition(composer)
    return segmentation_dataset
    