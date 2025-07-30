import cv2
import os

video_path = "IMG_1853.MOV"
output_dir = "frames_1853"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    filename = os.path.join(output_dir, f"{output_dir}_{frame_idx:06d}.png")
    cv2.imwrite(filename, frame)
    frame_idx += 1

cap.release()
print("Done!")
