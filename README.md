# YOLO + MedSAM Project

This repository contains the integration of YOLOv5 and MedSAM for medical image segmentation tasks.  
The codebase is organized under the `yolo` directory, where both **YOLOv5** and **MedSAM** related files are kept.  
Below is the description of the custom scripts and folders added to the project.

---


## 📂 Project Structure

yolo/
│
├── yolov5/                        # YOLOv5 related code
│   ├── extract_box_from_mask.py   # Extract bounding boxes from segmentation masks
│   ├── datasets/                  # Dataset files for YOLO training/testing
│   ├── datasets_val/              # Validation dataset (for later quantitative analysis)
│   ├── runs/                      # YOLO training and inference outputs
│   └── (other model files)        # All other files are from YOLOv5
│
├── medsam/                        # MedSAM related code
│   ├── yolo_boxes_to_medsam.py    # Convert YOLO bounding boxes to MedSAM input format
│   ├── rename_trim_mask.py        # Preprocessing script for mask renaming and trimming
│   ├── rename_trim_annotation.py  # Preprocessing script for annotation renaming and trimming
│   ├── dice_eval.py               # Script to compute Dice scores between GT and predictions
│   │
│   ├── dataset/                   # Dataset files for MedSAM
│   ├── datasets_val/              # Validation dataset results
│   ├── labels/                    # YOLO detection box labels
│   ├── mask_use_for_dice/         # Masks used for Dice evaluation
│   ├── test_set_mask/             # Final test set masks
│   └── test_set_mask_val/         # Validation set masks
│
└── dataprocess.py                 # Data preprocessing script



## ⚠️ Important Notes

- All other files not listed above are **original model files** provided by YOLOv5 or MedSAM and should not be modified.  
- If you want to **reproduce the full pipeline**, please **download the original dataset** first and modify the dataset paths in:
  - `dataprocess.py`
  - `extract_box_from_mask.py`

---

## 📌 Citation

If you use this repository for research or study, please cite the original YOLOv5 and MedSAM papers.
