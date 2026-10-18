# Pedestrian Intent Prediction for Autonomous Vehicles

## Project Overview
This project focuses on predicting pedestrian crossing intentions using computer vision and spatio-temporal deep learning. It combines YOLOv8-based object detection with LSTM-based temporal analysis to study pedestrian movement patterns.

## Technologies
- Python
- YOLOv8
- PyTorch
- OpenCV
- NumPy
- Pandas
- Jupyter Notebook

## Repository Structure
```text
pedestrian-intent-prediction/
├── src/
│   ├── JAAD_1 (3).ipynb
│   └── model.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Datasets
The project is intended to work with the JAAD and PIE pedestrian-behavior datasets. These datasets are not included in this repository and must be obtained separately.

## Installation

Clone the repository:

```bash
git clone https://github.com/prathipakrishnan/pedestrian-intent-prediction.git
cd pedestrian-intent-prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `src/JAAD_1 (3).ipynb` and follow the notebook cells. Dataset paths and model weights may need to be configured locally.

## Python Script
The repository also includes `src/model.py`. Check its input requirements before running it independently.

## License
See the `LICENSE` file for licensing information.

## Results

The repository includes a demonstration video of the pedestrian intent prediction system.

**Demo video:** [View the pedestrian prediction results](results/)

The demonstration illustrates the system's output using the uploaded project video.
