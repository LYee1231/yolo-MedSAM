import os
import cv2


def mask_to_yolo(mask_folder, label_folder, classes):

    os.makedirs(label_folder, exist_ok=True)

    for mask_file in os.listdir(mask_folder):
        if not mask_file.lower().endswith((".png", ".jpg")):
            continue

        mask_path = os.path.join(mask_folder, mask_file)
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
        h, w = img.shape


        ys, xs = (img > 0).nonzero()
        if len(xs) == 0 or len(ys) == 0:
            continue


        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()


        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        box_width = (xmax - xmin) / w
        box_height = (ymax - ymin) / h


        txt_file = os.path.join(label_folder, mask_file.rsplit("_", 1)[0] + ".txt")
        with open(txt_file, "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

        print(f"Processed {mask_file} -> {txt_file}")



classes = ["x"]
mask_to_yolo(
    mask_folder="C:/Users/lyjco/Desktop/final/1327317/mask",  
    label_folder="C:/Users/lyjco/Desktop/final/1327317/labels", 
    classes=classes
)