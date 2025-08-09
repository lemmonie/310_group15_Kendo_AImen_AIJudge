import time
import os
from datetime import datetime
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import mss
from torchvision.models import convnext_tiny
from torchvision.models import resnet18
from collections import Counter

# === Mode Settings ===
USE_VOTING_MODE = True   # True = voting mode, False = instant mode
SCREENSHOT_FPS = 30

# === Model Settings ===
MODEL_PATH = "kendo_resnet18.pth"
IMG_SIZE = 224
DISPLAY_WIDTH = 960

# === Save Hit Frames ===
SAVE_DIR = "hits"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_hit_frame(frame, label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    save_path = os.path.join(SAVE_DIR, f"{timestamp}_{label.upper()}.jpg")
    cv2.imwrite(save_path, frame)
    print(f"✅ Saved hit frame: {save_path}")

# === Transform for model ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load model ===
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
print(f"Classes: {idx_to_class}")

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model ready")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Running on {device}")

sequence_preds = []

def add_bottom_bar(frame, text_lines):
    h, w, _ = frame.shape
    bar_height = 90
    cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    for i, line in enumerate(text_lines):
        y_pos = h - bar_height + 30 + (i * 30)
        cv2.putText(frame, line, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame

# area
with mss.mss() as sct:
    full_screen = sct.grab(sct.monitors[1])
    full_img = np.array(full_screen)[:, :, :3]
    full_img = cv2.cvtColor(full_img, cv2.COLOR_BGRA2BGR)

print("Select an area, press ENTER to confirm, ESC to canccel")
roi = cv2.selectROI("Select Video Region", full_img, False)
cv2.destroyWindow("Select Video Region")

if sum(roi) == 0:
    print("Error, exit")
    exit()

x, y, w, h = roi
CAPTURE_REGION = {"top": y, "left": x, "width": w, "height": h}
print(f"✅ Selected Region: {CAPTURE_REGION}")


with mss.mss() as sct:
    last_time = 0
    print("Start monitoring, press Q to exit")

    while True:
        now = time.time()
        if now - last_time < 1.0 / SCREENSHOT_FPS:
            continue
        last_time = now

        screenshot = sct.grab(CAPTURE_REGION)
        frame = np.array(screenshot)[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        h0, w0 = frame.shape[:2]
        scale = DISPLAY_WIDTH / w0
        resized_frame = cv2.resize(frame, (DISPLAY_WIDTH, int(h0 * scale)))

        img_pil = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = output.argmax(1).item()
            pred_label = idx_to_class[pred_idx]
            pred_conf = probs[0][pred_idx].item()

        cv2.rectangle(resized_frame, (0, 0), (resized_frame.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(resized_frame, f"{pred_label.upper()} ({pred_conf:.2f})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

       
        if USE_VOTING_MODE:
            if pred_label != "none":
                # if it is a first hit, record
                if len(sequence_preds) == 0:
                    first_non_none_frame = resized_frame.copy()
                    first_non_none_label = pred_label

                sequence_preds.append((resized_frame.copy(), pred_label))

            else:
                if len(sequence_preds) > 0:
                    # get label
                    labels_only = [lbl for _, lbl in sequence_preds]

                    # weight for first hit in sque
                    weighted_labels = labels_only.copy()
                    weighted_labels.extend([labels_only[0]] * 2)
                    vote_result = Counter(weighted_labels).most_common(1)[0][0]
                    vote_count = Counter(weighted_labels)

                    print(f"Voting Result: {vote_result.upper()} {vote_count}")

                    # find first frame in swq
                    vote_first_frame = None
                    for frame, lbl in sequence_preds:
                        if lbl == vote_result:
                            vote_first_frame = frame.copy()
                            break

                    # show vote
                    text_lines = [
                        f">>> HIT DETECTED: {vote_result.upper()} <<<",
                        f"Votes: LEFT={vote_count.get('left', 0)}, RIGHT={vote_count.get('right', 0)}",
                    ]
                    resized_frame = add_bottom_bar(resized_frame, text_lines)

                    # save images
                    # 1. first non-none
                    save_hit_frame(first_non_none_frame, f"START_{first_non_none_label}")
                    # 2. the first hit matches the result
                    if vote_first_frame is not None:
                        save_hit_frame(vote_first_frame, f"RESULT_{vote_result}")

                    sequence_preds.clear()

        else:
            if pred_label in ["left", "right"]:
                print(f"Instant Hit: {pred_label.upper()}")
                text_lines = [
                    f">>> HIT DETECTED: {pred_label.upper()} ({pred_conf:.2f}) <<<",
                ]
                resized_frame = add_bottom_bar(resized_frame, text_lines)
                # save_hit_frame(resized_frame, pred_label)

        cv2.imshow("Screen Monitor Kendo Detection", resized_frame)

        if cv2.getWindowProperty("Screen Monitor Kendo Detection", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user.")
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stop monitoring.")
            break

    cv2.destroyAllWindows()
