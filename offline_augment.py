import os
import random
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import numpy as np
import cv2
import shutil

# add the root
SRC_ROOT = "dataset_more_none"
DST_ROOT = "dataset_aug"
AUG_PER_IMAGE = 2

def random_perspective(img, max_shift=0.05):
    arr = np.array(img)
    h, w = arr.shape[:2]

    # the original 4 angels
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # shifting the 4 angles (modelling different camera angels)]
    shift_x = w * random.uniform(-max_shift, max_shift)
    shift_y = h * random.uniform(-max_shift, max_shift)
    dst_pts = np.float32([
        [0 + shift_x, 0 + shift_y],
        [w - shift_x, 0 + shift_y],
        [w - shift_x, h - shift_y],
        [0 + shift_x, h - shift_y]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    arr_warped = cv2.warpPerspective(arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(arr_warped)

def augment_image(img: Image.Image):
    # changing the angel of the aug to 3-7 degree (modeling shooting from the sides)
    if random.random() < 0.5:
        img = random_perspective(img, max_shift=0.05)

    # blurring or sharpening the photo
    if random.random() < 0.4:
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.5)))
        else:
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150))

    # lighting and white balance adjustment
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.2))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))

    # Gamma shift
    if random.random() < 0.4:
        gamma = random.uniform(0.9, 1.3)
        table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in range(256)]).astype("uint8")
        img = Image.fromarray(cv2.LUT(np.array(img), table))

    # compress
    if random.random() < 0.4:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(40, 90)]
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        _, enc_img = cv2.imencode(".jpg", img_cv, encode_param)
        img_cv_compressed = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        img = Image.fromarray(cv2.cvtColor(img_cv_compressed, cv2.COLOR_BGR2RGB))

    # noise
    if random.random() < 0.3:
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, 0.02, arr.shape)
        arr = np.clip(arr + noise, 0, 1)
        img = Image.fromarray((arr*255).astype(np.uint8))

    return img

def augment_dataset():
    # make dst
    os.makedirs(DST_ROOT, exist_ok=True)

    # read left/none/right
    classes = [c for c in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, c))]
    for cls in classes:
        src_cls_dir = os.path.join(SRC_ROOT, cls)
        dst_cls_dir = os.path.join(DST_ROOT, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        print(f"Category: {cls}")
        files = [f for f in os.listdir(src_cls_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]

        for f in tqdm(files):
            src_path = os.path.join(src_cls_dir, f)

            # copy image to dst
            shutil.copy(src_path, os.path.join(dst_cls_dir, f))

            # read image
            img = Image.open(src_path).convert("RGB")
            fname, ext = os.path.splitext(f)

            # make augmented image
            for i in range(AUG_PER_IMAGE):
                aug_img = augment_image(img.copy())
                aug_name = f"{fname}_aug{i+1}{ext}"
                aug_img.save(os.path.join(dst_cls_dir, aug_name))

    print(f"Done, all images are saved into: {DST_ROOT}/(left|none|right)")

if __name__ == "__main__":
    augment_dataset()