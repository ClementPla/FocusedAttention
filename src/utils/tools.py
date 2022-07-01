import cv2
import numpy as np
import os
    


def show_interpretability_on_image(img, mask, alpha):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap*alpha + (1-alpha)*np.float32(img)
    return cam

def save_cv2_img(img, folder, filename):
    if img.ndim==3:
        if img.shape[0]==3:
            img = img.transpose((1,2,0))
        else:
            img = np.squeeze(img)
        if img.shape[2]==4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    filepath = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(filepath, img)



def set_text(img, text, fontScale=0.5, thickness=1):# setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get boundary of this text
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    # get coords based on boundary
    textX = (img.shape[1] - textsize[0]) // 2
    textY = (textsize[1])
    # add text centered on image
    return cv2.putText(img, text, (textX, textY ), font, fontScale, (255, 255, 255, 255), thickness)


def minmax_norm(vec):
    return (vec-vec.min())/(vec.max()-vec.min())

