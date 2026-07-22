# Molten Metal Fill Percentage Detection using OpenCV

## Overview

This project detects the molten metal region inside a mould cavity and calculates the fill percentage using computer vision techniques. The system processes either a video file or a live camera stream, extracts the molten metal region using color thresholding, removes noise using morphological operations, and computes the filling percentage in real time.

This project is designed for industrial automation applications where monitoring the filling process is important for quality inspection and process control.

---

## Features

- Detect molten metal using HSV color thresholding
- Works with video files or live camera
- Morphological filtering for noise removal
- Detect the largest molten region
- Calculate molten metal area
- Compute fill percentage
- Display contour, center point, and enclosing circle
- Real-time visualization

---

## Technologies Used

- Python 3.x
- OpenCV
- NumPy

---

## Project Workflow

```
Video Input
      │
      ▼
Read Frame
      │
      ▼
Convert BGR → HSV
      │
      ▼
HSV Thresholding
      │
      ▼
Binary Molten Mask
      │
      ▼
Morphological Operations
(Open + Close)
      │
      ▼
Largest Contour Detection
      │
      ▼
Minimum Enclosing Circle
      │
      ▼
Area Calculation
      │
      ▼
Fill Percentage
      │
      ▼
Display Output
```

---

## Morphological Operations

### MORPH_OPEN

Opening is performed using:

```
Erosion → Dilation
```

Purpose:

- Removes small white noise
- Eliminates unwanted pixels
- Preserves the main molten region

Example:

Before

```
*      *
   #######
   #######
*  #######
```

After Opening

```
   #######
   #######
   #######
```

---

### MORPH_CLOSE

Closing is performed using:

```
Dilation → Erosion
```

Purpose:

- Fills small holes
- Connects nearby molten regions
- Produces a smoother object

---

## Algorithm

1. Read video frame
2. Convert image to HSV
3. Apply HSV threshold
4. Generate molten mask
5. Apply Opening
6. Apply Closing
7. Detect contours
8. Select largest contour
9. Calculate contour area
10. Compute enclosing circle
11. Calculate fill percentage
12. Display results

---

## Fill Percentage Calculation

The fill percentage is calculated using

```
Fill Percentage =

(Detected Molten Area / Mould Area) × 100
```

Where

```
Mould Area = π × Radius²
```

---

## Sample Output

```
Center : (430, 280)

Radius : 186

Fill Percentage : 47.40 %
```

---

## Folder Structure

```
Project/
│
├── data.py
├── bb4.py
├── bb5.py
├── bbb.py
├── up.py
├── video.mp4
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/molten-metal-fill-detection.git
```

Go to project directory

```bash
cd molten-metal-fill-detection
```

Install dependencies

```bash
pip install opencv-python numpy
```

Run

```bash
python up.py
```

---

## Applications

- Steel Industry
- Foundry Automation
- Die Casting
- Aluminium Casting
- Smart Manufacturing
- Industrial Vision Inspection
- PLC and SCADA Integration

---

## Future Improvements

- YOLO-based mould detection
- Automatic cavity tracking
- Multi-cavity monitoring
- AI-based fill prediction
- PLC communication
- SCADA dashboard
- Defect detection
- Self-learning threshold optimization
- Real-time production analytics

---

## Advantages

- Low-cost inspection
- Real-time processing
- High accuracy
- Easy deployment
- Industrial automation ready
- Scalable for multiple moulds

---

## Author

**BOSE A**

BE Artificial Intelligence and Machine Learning

Sree Sakthi Engineering College

Python | OpenCV | Machine Learning | Computer Vision

---

## License

This project is intended for educational and industrial research purposes.