import gc

from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable

from gan.net2 import Net


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)),
                         Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch


def apply_style_transfer(style_model: Net, main_photo: str, style_photo: str, im_size: int)->Image:
    content_image = tensor_load_rgbimage(main_photo, size=im_size,
                                         keep_asp=True).unsqueeze(0)
    style = tensor_load_rgbimage(style_photo, size=im_size).unsqueeze(0)
    style = preprocess_batch(style)

    style_v = Variable(style)
    content_image = Variable(preprocess_batch(content_image))
    style_model.setTarget(style_v)

    output = style_model(content_image)

    del content_image, style, style_v

    torch.cuda.empty_cache()
    gc.collect()

    tensor = output.data[0]
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))

    if torch.cuda.is_available():
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()

    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)

    return img
