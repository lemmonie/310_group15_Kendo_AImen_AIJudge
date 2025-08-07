# 310_group23_Kendo_AI-men_AIJudge

This project is a deep learning system that classifies Kendo frames using a fine-tuned ConvNeXt-Tiny model.
It is designed to support automated judging systems by analyzing image sequences and identifying key decision points during Kendo matches.

The model classifies each frame into three categories:  

- **left** → left player strikes  
- **right** → right player strikes  
- **none** → no valid strike action 

---
## Requirements

Before running the project, ensure the following setup is completed:

1. Use **Python 3.11**  
   - Download: [Python 3.11.9](https://www.python.org/downloads/release/python-3119/)  
   - In PyCharm: `File → Settings → Project → Python Interpreter → Add Interpreter → Select Python 3.11`

2. In the terminal, install dependencies: pip install -r requirements.txt

3. Run the scripts in order:
   - offline_augment.py → Generate augmented dataset 
   - train_kendo_convnext_v2.py → Train and save the model 
   - screenMonitoring.py → Start real-time screen capture and prediction

---

## 1. Model Overview

The final model uses ConvNeXt Tiny (pretrained on ImageNet), fine-tuned with Kendo data.

- **Input size**: 224×224  
- **Classes**: `left`, `none`, `right`  
- **Fine-tuning strategy**:  
  - Frozen early layers for stable feature extraction  
  - Unfrozen the last ~100 layers for fine-grained adaptation  
  - No horizontal flip augmentation (to preserve left/right info)  
  - Mild color jitter and resized crop to improve robustness  

### Confusion Matrix and Classification Report
Confusion matrix:
```
['left', 'none', 'right']
[[3 0 0]
 [0 4 0]
 [0 0 5]]
```

Classification Report:
```
              precision    recall  f1-score   support

        left       1.00      1.00      1.00         3
        none       1.00      1.00      1.00         4
       right       1.00      1.00      1.00         5

    accuracy                           1.00        12
   macro avg       1.00      1.00      1.00        12
weighted avg       1.00      1.00      1.00        12
```

---

## 2. Improved Dataset with Pre-Hit-Images

We further improved the dataset by adding **Pre-Hit-Images**(#26-#35 dir: dataset_more_none-->none):  

- **Pre-Hit-Images** → the moment when shinai (bamboo sword) is **very close to hitting** but hasn’t actually landed.  

This makes the model better at distinguishing:  

- **True strikes (left/right)** vs. **almost-hit frames**  
- Reduces false triggers from near-hit actions  

There are now **two dataset variants**:  

- `dataset/` → **does NOT** include pre-hit frames  
- `dataset_more_none/` → **includes pre-hit frames** for better precision  

## 3. Screen Capturing System

This module allows live detection of Kendo strikes during video playback or screen capture.

### Modes Available

- **Voting Mode**
  - Accumulates `left/right` predictions across multiple frames  
  - Ends a strike sequence when a `none` is detected  
  - Uses **majority vote** to decide `LEFT` or `RIGHT`  
  - More stable and ideal for official judging

- **Instant Mode**
  - Stops immediately when a hit is detected
  - Faster but more sensitive to single-frame errors

---

### Performance Comparison
- **Model trained with dataset/**  
  - Sometimes misjudges double strikes (draws)

- **Model trained with dataset_more_none/**  
  - More precise, avoid false positives from near hits

## 4. How to Use

### 4.1 Dataset

- `dataset/`  
  - Contains only **clear strike and idle frames**  
- `dataset_more_none/`  
  - Adds **pre-hit none frames** (shinai close to target but not yet striking)

You can train the model with either dataset depending on your needs.  

---

### 4.2 Video Format

- Supported formats: `.mp4`, `.avi`  
- Videos should have **at least 30 FPS** for better detection  
- If your video FPS is low, consider **frame interpolation (e.g., RIFE)** before inference  

---

### 4.3 Mode Selection

In `video_predict.py`, you can choose **mode** by changing:  

```python
USE_VOTING_MODE = True  # True = voting mode, False = instant mode
```

- **True** → Voting Mode (more stable, pauses after full sequence)  
- **False** → Instant Mode (pauses immediately on detection)  

---

### 4.4 Running Video Detection

1. **Run the script**  

   ```bash
   python video_predict.py
   ```
   
2. **Select a video** when prompted  

3. **During playback**  
   - Predictions appear at the **top bar**  
   - When a strike is detected:  
     - Bottom black bar will show:  
       ```
       >>> HIT DETECTED: LEFT <<<
       Votes: LEFT=7, RIGHT=2   (only in voting mode)
       Press 'C' to continue
       ```
     - Video **pauses automatically**  
   - **Press `C`** → Continue
   - **Press `Q`** → Quit 

---