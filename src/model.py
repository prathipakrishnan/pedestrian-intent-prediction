import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from ultralytics import YOLO
import os
import csv
from datetime import datetime

# --- 1. Model Structure ---
class EnhancedPedestrianIntentModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=3):
        super(EnhancedPedestrianIntentModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))

# --- 2. Setup & Logging ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = r"C:\Users\pavit\JAAD\directional_intent_model_v5.pth"
VIDEO_PATH = r"C:\Users\pavit\Desktop\mTech\Data set\Autonomous dataset\JAAD Dataset\JAAD_clips\JAAD_clips\video_0125.mp4"
LOG_FILE = "simulation_results.csv"

# Initialize CSV for your Thesis Data
with open(LOG_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Frame", "ID", "Intent_P", "TTC", "Speed_KMPH", "Status"])

intent_model = EnhancedPedestrianIntentModel(input_size=4).to(device)
intent_model.load_state_dict(torch.load(MODEL_PATH))
intent_model.eval()

detector = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture(VIDEO_PATH)

# Simulation Variables
ego_speed = 40.0 
is_braking = False
ped_history = {}
prob_buffer = {}
frame_count = 0

print(f"{'Frame':<8} | {'ID':<5} | {'Status':<18} | {'Intent':<8} | {'TTC':<6} | {'Speed'}")
print("-" * 70)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    
    # Environment Setup
    h_orig, w_orig, _ = frame.shape
    frame = frame[int(h_orig * 0.45):, :] 
    h, w, _ = frame.shape
    road_edge_y = h * 0.85 
    
    # Physics Engine
    if is_braking:
        ego_speed = max(0, ego_speed - 1.5) 
    else:
        ego_speed = min(40.0, ego_speed + 0.1) 
        
    results = detector.track(frame, classes=[0], persist=True, verbose=False)
    is_braking = False 
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
            
            if tid not in ped_history: ped_history[tid] = deque(maxlen=30)
            vx, vy = (cx - ped_history[tid][-1][0], cy - ped_history[tid][-1][1]) if ped_history[tid] else (0, 0)
            ped_history[tid].append([cx, cy, vx, vy])
            
            if len(ped_history[tid]) == 30:
                inp = torch.tensor(np.array(list(ped_history[tid])), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    raw_prob = intent_model(inp).item()
                
                # Smoothing
                if tid not in prob_buffer: prob_buffer[tid] = deque(maxlen=5)
                prob_buffer[tid].append(raw_prob)
                smoothed_prob = sum(prob_buffer[tid]) / len(prob_buffer[tid])
                
                # TTC Calculation
                dist_to_road = road_edge_y - y2
                v_y_px = vy * h
                ttc = dist_to_road / v_y_px if v_y_px > 0.1 else 99
                
                # Safety Logic & Logging
                if smoothed_prob >= 0.80 and ttc < 2.5:
                    color, status = (0, 0, 255), "CRITICAL: BRAKE"
                    is_braking = True
                elif smoothed_prob >= 0.60:
                    color, status = (0, 165, 255), "INTENT: CROSS"
                else:
                    color, status = (0, 255, 0), "SAFE"

                # Output to Terminal
                if is_braking or frame_count % 15 == 0:
                    print(f"{frame_count:<8} | {tid:<5} | {status:<18} | {smoothed_prob:.2f}   | {ttc:4.1f}s | {ego_speed:.1f} km/h")
                    
                    # Log to CSV
                    with open(LOG_FILE, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now(), frame_count, tid, round(smoothed_prob,2), round(ttc,1), round(ego_speed,1), status])

                # Draw UI
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID:{tid} P:{smoothed_prob:.2f}", (int(x1), int(y1)-10), 0, 0.5, color, 2)

    # Dashboard Overlay
    cv2.rectangle(frame, (10, 10), (250, 80), (0,0,0), -1)
    speed_color = (0, 0, 255) if is_braking else (0, 255, 0)
    cv2.putText(frame, f"SPEED: {ego_speed:.1f} km/h", (20, 45), 0, 0.7, speed_color, 2)
    cv2.putText(frame, "BRAKE: ON" if is_braking else "BRAKE: OFF", (20, 70), 0, 0.6, speed_color, 2)

    cv2.imshow("M.Tech Simulation & Logging", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
