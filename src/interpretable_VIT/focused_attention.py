import torch
import torch.nn.functional as F
from copy import deepcopy
import math
from icecream import ic


def resample_positional_embeddings(network, new_stride,
                                   original_stride=16,
                                   original_size=384):
    if new_stride == original_stride:
        return original_size
    padding = new_stride // 2 # wheter or not we include tokens from the border of the image
    padding = 0
    network.patch_embed.proj.padding = padding, padding
    network.patch_embed.proj.stride = new_stride, new_stride

    new_size = (original_size - original_stride + 2 * padding) // (new_stride) + 1

    pos_embed = network.pos_embed.transpose(2, 1)  # BxFxN
    _, f, N = pos_embed.shape
    cls = pos_embed[:, :, 0]
    cls = torch.unsqueeze(cls, 2)
    pos_embed = pos_embed[:, :, 1:]
    pos_embed = pos_embed.reshape(1, f, int(N ** 0.5), int(N ** 0.5))
    pos_embed = F.interpolate(pos_embed, size=(new_size, new_size)).flatten(2)
    pos_embed = torch.cat((cls, pos_embed), 2).transpose(2, 1)
    network.pos_embed = torch.nn.Parameter(pos_embed)
    return new_size ** 2


def normalize(vec):
    # return torch.sigmoid(vec)
    with torch.no_grad():
        return (vec-vec.min())/(vec.max()-vec.min())


def get_transformer_attribution(attribution_generator, image, index, method='transformer_attribution'):
    transformer_attribution, prediction = attribution_generator.generate_LRP(image, method=method, index=index)
    # transformer_attribution = normalize(transformer_attribution.detach())
    return transformer_attribution


def get_focused_attention(attribution_generator,
                      image,
                      class_index,
                      iterations=4,
                      max_sampling_number=256,
                      use_cuda=True,
                      method='transformer_attribution',
                      threshold=0,
                      smoothing=0.5,
                      ):

    network = attribution_generator.model
    original_stride = network.patch_embed.proj.stride[0]
    original_pos_embedded = deepcopy(network.pos_embed)
    network.filter_index_module.filter_index = None
    factors = [2 ** (i) for i in range(iterations)]
    for i, f in enumerate(factors):
        network.pos_embed = original_pos_embedded
        new_stride = original_stride // f
        ic('Current stride %i' % new_stride)
        embedded_size = resample_positional_embeddings(network, new_stride, original_stride=original_stride)
        if i == 0:
            attribution = get_transformer_attribution(attribution_generator, image, class_index, method=method)
        else:
            attr_shape = attribution.shape[1]
            attribution = attribution.reshape(1, 1, int(attr_shape ** 0.5), int(attr_shape ** 0.5))
            attribution = torch.nn.functional.interpolate(attribution,
                                                          size=(int(embedded_size ** 0.5), int(embedded_size ** 0.5)),
                                                          mode='bilinear')
            attribution = attribution.flatten()
            attribution = torch.clamp(attribution, 0, 1)
            distibution = attribution / attribution.sum()

            R = int(1.5 ** i)
            cdf = distibution.cumsum(0)
            for k in range(R):
                r = torch.rand(size=(max_sampling_number,))
                if use_cuda:
                    r = r.cuda()
                filtered_index = torch.searchsorted(cdf, r)

                neibouring_filter = [-1]
                for f in filtered_index:
                    neibouring_filter += [max(0, f - math.sqrt(embedded_size)), min(f + 1, embedded_size - 1),
                                          max(f - 1, 0),
                                          min(embedded_size - 1, f + math.sqrt(embedded_size))]

                neigbors = torch.Tensor(neibouring_filter).long()
                if use_cuda:
                    neigbors = neigbors.cuda()
                filtered_index = torch.cat([neigbors, filtered_index]) + 1
                network.filter_index_module.filter_index = filtered_index.detach()
                new_attribution = get_transformer_attribution(attribution_generator, image, class_index)
                new_attribution = normalize(new_attribution)
                attribution = attribution.squeeze()
                if threshold:
                    new_attribution[new_attribution < threshold] = 0

                attribution[filtered_index[1:] - 1] = (new_attribution.squeeze()*smoothing + (1-smoothing)*attribution[
                    filtered_index[1:] - 1])
            attribution.unsqueeze_(0)

    attribution[attribution < threshold] = 0
    network.pos_embed = original_pos_embedded
    network.patch_embed.proj.padding = 0, 0
    network.patch_embed.proj.stride = original_stride, original_stride
    network.filter_index_module.filter_index = None
    return attribution.clamp(0, 1.0)


class FocusedAttention:
    def __init__(self, attribution_generator):
        self.attribution_generator = attribution_generator

    def attribute(self, img, target, **kwargs):
        attribution = get_focused_attention(self.attribution_generator, img, class_index=target,
                                            **kwargs)
        attribution = attribution.reshape(1, 1, int(attribution.shape[1] ** 0.5), int(attribution.shape[1] ** 0.5))

        return attribution
