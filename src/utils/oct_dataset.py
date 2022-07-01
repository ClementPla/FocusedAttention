import albumentations as A
import nntools.dataset as D
import cv2

KERMANY_score = {
    0: 'CNV',
    1: 'DME',
    2: 'Drusen',
    3: 'Normal'
}
@D.nntools_wrapper
def to_rgb(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return {'image': image}


def get_kermany_dataset(root, shape, mean=(0.5,0.5,0.5),
                        std=(0.5,0.5,0.5)):

    dataset = D.ClassificationDataset(img_url=root, shape=shape)

    composer = D.Composition()
    composer << to_rgb << A.Normalize(mean=mean,
                                      std=std,
                                      always_apply=True)

    dataset.composer = composer

    return dataset
