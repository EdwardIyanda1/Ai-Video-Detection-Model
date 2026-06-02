# Ai-Video-Detection-Model

This project is a forensic analysis tool designed to identify and classify AI-manipulated or "deepfake" video content by leveraging computer vision and deep learning techniques.

---

## Overview

The goal of this project is to provide a robust pipeline to detect media tampering, specifically focusing on facial manipulation and synthetic content generation. By combining spatial analysis, temporal consistency checks, and metadata forensics, this tool aims to provide a probability-based assessment of video authenticity.

## Key Features

* **Frame Extraction:** Automated processing of video files into sequential frames for granular analysis.
* **Region of Interest (ROI) Detection:** Utilizing `MediaPipe` to isolate and crop facial data, minimizing background noise and focusing on sensitive features.
* **Classification Pipeline:** A deep learning framework (compatible with PyTorch/TensorFlow) designed to ingest processed frames and classify them as "authentic" or "manipulated."
* **Metadata Inspection:** Forensic tools to extract and verify file-level data, checking for inconsistencies in device information or evidence of post-processing software.

## Project Structure

```text
Ai-Video-Detection-Model/
├── data/
│   ├── raw/            # Store original video files here (e.g., 1000547533.mp4)
│   └── processed/      # Extracted frames and ROI crops
├── models/             # Saved weights and model architectures
├── scripts/            # Python utility scripts
│   ├── extract.py      # Script for frame extraction
│   └── analyze.py      # Placeholder for detection/forensic logic
├── main.py             # Entry point for the detection pipeline
└── README.md

```

## Setup Instructions

1. **Clone the repository.**
```bash

```



git clone https://github.com/edwardiyanda1/Ai-Video-Detection-Model.git


```
2. **Install dependencies:**
```bash

```



pip install opencv-python mediapipe torch torchvision

```
3.  **Run extraction:**
    Place your target file (e.g., `1000547533.mp4`) in the `data/raw/` folder and run the extraction script to prepare your dataset.

## Ethical Usage
This project is intended for research, academic, and verification purposes. It should be used to promote transparency and combat the spread of digital misinformation.

---

*Note: This project is currently in development. Future iterations will include temporal consistency modules and a web-based interface for easier verification.*

```