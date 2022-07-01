import timm
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, model_urls, Bottleneck
from torchvision.models.vgg import vgg19_bn
import torch.nn as nn
from nntools.nnet import nnt_format


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')

        model.load_state_dict(state_dict, strict=False)
    return model


def get_network(architecture, n_classes, **kwargs):
    config = {'architecture': architecture, 'n_classes':n_classes,
              'pretrained': False}

    if config['architecture'] == 'vit_base_patch32_384':
        from timm.models.vision_transformer import vit_base_patch32_384
        network = vit_base_patch32_384(pretrained=config['pretrained'], num_classes=config['n_classes'])

    if config['architecture'] == 'efficientnet_b7_ns':
        from timm.models.efficientnet import tf_efficientnet_b7_ns
        network = tf_efficientnet_b7_ns(pretrained=config['pretrained'], num_classes=config['n_classes'])

    if config['architecture'] == 'vit_base_patch16_384':
        from timm.models.vision_transformer import vit_base_patch16_384
        network = vit_base_patch16_384(pretrained=config['pretrained'], num_classes=config['n_classes'])

    if config['architecture'] == 'vit_large_patch32_384':
        from timm.models.vision_transformer import vit_large_patch32_384
        network = vit_large_patch32_384(pretrained=config['pretrained'], num_classes=config['n_classes'])

    if config['architecture'] == 'vit_large_patch16_384':
        from timm.models.vision_transformer import vit_large_patch16_384
        network = vit_large_patch16_384(pretrained=config['pretrained'], num_classes=config['n_classes'])

    if config['architecture'] == 'Wide ResNet-101-2':
        network = _resnet('wide_resnet101_2',
                          Bottleneck, [3, 4, 23, 3],
                          pretrained=config['pretrained'],
                          progress=True,
                          num_classes=config['n_classes'], width_per_group=64 * 2)

    if config['architecture'] == 'inception_v3':
        from timm.models.inception_v3 import inception_v3
        network = inception_v3(pretrained=config['pretrained'], num_classes=config['n_classes'])

    if config['architecture'] == 'vgg19':
        network = vgg19_bn(pretrained=True)
        classifier = network.classifier
        removed = list(classifier.children())[:-1]
        network.classifier = nn.Sequential(*removed, nn.Linear(4096, config['n_classes'], bias=True))

    if config['architecture'] == 'ResNet152':
        network = _resnet('resnet152',
                          Bottleneck, [3, 8, 36, 3],
                          pretrained=config['pretrained'],
                          progress=True,
                          num_classes=config['n_classes'])
    return nnt_format(network)
