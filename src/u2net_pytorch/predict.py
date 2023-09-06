import glob
import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import PIL
from PIL import Image
import wandb
from tqdm import tqdm

IMG_SZ = 320
RGB_MEAN = [0.5, 0.5, 0.5]
RGB_STD = [0.5, 0.5, 0.5]

## added now
if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image


## function not used
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# function not used

# dont remember right now, but not being used
def preprocessImage(img):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    kernel = np.ones((7, 7), np.uint8)
    img_erosion = cv2.erode(blur, kernel, iterations=1)
    # img_dilation = cv2.dilate(blur, kernel, iterations=1)

    mask_maker = img_erosion

    edges = cv2.Canny(mask_maker, 50, 70)
    kernel = np.ones((5, 5), np.uint8)
    img_edges = cv2.dilate(edges, kernel, iterations=1)
    return img_edges


# taken from og code itself, just normalizing for more confidence.
# opacity range is 0-255, if model predicts in range 10-240, we stretch it out and make as 0-255, 240: 255
def normPREDNumpy(d):
    ma = np.max(d)
    mi = np.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def get_mask_for_image(og_img, ort_session):
    # needs to be opencv's rgb image (color)
    #     print(og_img.shape)
    h, w, _ = og_img.shape
    if h > w:
        img = image_resize(og_img, height=IMG_SZ)
    else:
        img = image_resize(og_img, width=IMG_SZ)

    x = img
    x = x / 255.0
    x[..., 0] -= RGB_MEAN[0]
    x[..., 1] -= RGB_MEAN[1]
    x[..., 2] -= RGB_MEAN[2]
    x[..., 0] /= RGB_STD[0]
    x[..., 1] /= RGB_STD[1]
    x[..., 2] /= RGB_STD[2]

    pad_x = int(IMG_SZ - x.shape[0])
    pad_y = int(IMG_SZ - x.shape[1])
    x = np.pad(x, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant")

    inp = np.array([x]).astype("float32")
    inp = np.transpose(inp, (0, 3, 1, 2))
    inp.shape

    ort_inputs = {ort_session.get_inputs()[0].name: inp}
    ort_outs = ort_session.run(None, ort_inputs)[0]
    ort_outs.shape

    res = np.squeeze(ort_outs)
    res = normPREDNumpy(res)
    res.shape

    if pad_x > 0:
        res = res[:-pad_x]
    if pad_y > 0:
        res = res[:, :-pad_y]
    res = res * 255

    bg_mask = cv2.resize(res, (og_img.shape[1], og_img.shape[0]))
    #     print(bg_mask.shape)

    return bg_mask


def apply_mask(og_img, bg_mask):
    # TODO: replace with numpy op

    bg_removed = cv2.cvtColor(og_img, cv2.COLOR_BGR2BGRA)
    dist = []
    for i in range(bg_removed.shape[0]):
        for j in range(bg_removed.shape[1]):
            mask_val = int(bg_mask[i][j])
            # if mask_val > 80: mask_val = 255
            bg_removed[i][j][3] = mask_val

    return bg_removed


def add_fill(og_img, bg_removed, bg_mask, fill_col=[255, 255, 255]):
    height, width, _ = og_img.shape
    background = np.zeros((height, width, 4), np.uint8)
    background[:, 0:width] = (fill_col[0], fill_col[1], fill_col[2], 255)  # (R, G, B, alpha)
    foreground = bg_removed

    background = cv2.cvtColor(background, cv2.COLOR_RGBA2RGB)
    foreground = cv2.cvtColor(foreground, cv2.COLOR_RGBA2RGB)
    alpha = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2RGB)

    foreground = foreground.astype(float)
    background = background.astype(float)
    alpha = alpha.astype(float) / 255
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    outImage = cv2.add(foreground, background)
    outImage /= 255
    outImage = np.array(outImage, dtype=np.float32)

    return outImage


def predict(args):
    model_path = args.model_path
    input_img = args.input_image
    input_dir = args.input_dir
    out_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y_%m_%d_%H_%M"))
    fill_col = [255, 0, 0]

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    u2net = model_path
    ort_session = onnxruntime.InferenceSession(u2net)

    input_files = []
    if input_img:
        input_files.append(input_img)
    if input_dir:

        input_files += [
            f
            for f in glob.glob(input_dir + "/*.*")
            if ".jpg" in f.lower() or ".jpeg" in f.lower() or ".png" in f.lower()
        ]

    for f in tqdm(input_files):
        og_img = cv2.imread(f)
        og_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2RGB)
        bg_mask = get_mask_for_image(og_img, ort_session)
        bg_removed = apply_mask(og_img, bg_mask)
        bg_removed = add_fill(og_img, bg_removed, bg_mask, fill_col)
        # bg_removed = cv2.cvtColor(bg_removed, cv2.COLOR_RGBA2BGRA)
        bg_removed = cv2.cvtColor(bg_removed, cv2.COLOR_RGB2BGR) * 255
        alpha = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2RGB)
        dest_fpath = os.path.join(
            out_dir, ".".join(f.split("/")[-1].split(".")[:-1]) + ".png"
        )  # TODO: replace with Path
        cv2.imwrite(dest_fpath, bg_removed)
        dest_fpath = os.path.join(
            out_dir, ".".join(f.split("/")[-1].split(".")[:-1]) + "_alpha.png"
        ) 
        
        # TODO: replace with Path
        cv2.imwrite(dest_fpath, alpha)


        ## logging images
        wandb.init(project="Experimenting",name = "Expt1")
      
        wandb.log({"inputimage": wandb.Image(og_img)})
        wandb.log({"predicted_mask": wandb.Image(alpha)})
        wandb.log({"Predicted_BGRemoved_mask": wandb.Image(bg_removed)})
