# pedestrian-intent-prediction
Spatio-temporal deep learning model combining YOLOv8 and BiLSTM with attention for pedestrian intent prediction on JAAD &amp; PIE datasets.

# Spatio-Temporal Pedestrian Intent Prediction for Autonomous Vehicles

This repository contains the source code and implementation for predicting pedestrian road-crossing intentions using a spatio-temporal deep learning architecture that combines **YOLOv8** for spatial object detection and **BiLSTM with attention mechanisms** for temporal trajectory analysis.

## 🚗 Project Overview
Accurately predicting pedestrian intent is critical for autonomous driving safety. This model processes video sequences from standard datasets like **JAAD** and **PIE**, tracks pedestrian bounding boxes, analyzes historical motion patterns, and computes time-to-collision (TTC) to support real-time braking decisions.

##  Repository Structure
- `src/` - Contains core model architectures, data processing scripts, and real-time simulation pipelines.
- `requirements.txt` - Python package dependencies.
- `LICENSE` - Open-source license for the repository.

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/prathipakrishnan/pedestrian-intent-prediction.git](https://github.com/prathipakrishnan/pedestrian-intent-prediction.git)
   cd pedestrian-intent-prediction
pip install -r requirements.txt

python src/simulation.py

   
