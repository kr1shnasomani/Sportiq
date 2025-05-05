<h1 align="center">Sportiq</h1>
<p align="center" style="margin-top:30px;">
  <img src="https://github.com/user-attachments/assets/34e9b4e7-a768-4143-b819-fc726dbf96d3" height="150cm"/>
</p>
The project is designed for player and ball tracking in tennis. It processes video input to detect court boundaries, track player movements, and map ball trajectories, utilizing computer vision techniques, Mediapipe, and a custom ball detection model for accurate analysis and visualization.

## Repository Structure:
The following is the structure of the repository:
```
Sportiq
├── code/
│   ├── BallDetection.py
│   ├── BallMapping.py
│   ├── BallTrackNet.py
│   ├── BodyTracking.py
│   ├── CourtDetection.py
│   ├── CourtMapping.py
│   └── TraceHeader.py
├── dataset/
│   └── input.mp4
├── model/
│   └── TrackNet-Model.md
├── output/
│   └── output.mp4
├── README.md
└── requirements.txt
```

## Execution Guide:
1. Clone the repository:
   ```
   git clone https://github.com/kr1shnasomani/Sportiq.git
   cd Sportiq
   ```
   
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

3. In the following files, replace the path with yours:
   - `CourtDetection.py` - line 13, enter root directory
   - `TraceHeader.py` - line 4, enter the input video path

4. Run the project:
   
   You will need to run the `CourtDetection.py` script to run thei entire project:
   ```
   /usr/bin/python3 /Users/krishnasomani/Documents/Projects/Sportiq/code/CourtDetection.py
   ```

## Output:

https://github.com/user-attachments/assets/31d239fb-44a8-4906-a2e6-43886348ba5b
