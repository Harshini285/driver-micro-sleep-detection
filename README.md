
#  Driver Micro-Sleep Detection System (Real-Time)

A **real-time computer vision–based driver monitoring system** that detects **micro-sleep, drowsiness, yawning, and driver distraction** using facial landmarks and temporal analysis.  
The system provides **instant audio alerts** to reduce accident risk caused by fatigue.

---

##  Problem Statement

Driver fatigue and micro-sleep are among the leading causes of road accidents.  
Many existing solutions are either hardware-intensive, expensive, or unreliable under real-world conditions such as head tilt, eye-shape variation, or brief landmark loss.

---

##  Solution Overview

This project uses a **standard webcam + MediaPipe Face Landmarker** to continuously monitor the driver and detect unsafe conditions such as:

- Eyes closed for prolonged duration (micro-sleep)
- High blink frequency (fatigue)
- Yawning
- Driver looking away from the road

The system escalates alerts based on severity and triggers **non-blocking audio alarms** in dangerous situations.

---

##  Key Features

-  **Eye Aspect Ratio (EAR)** for eye-closure detection  
-  **Mouth Aspect Ratio (MAR)** for yawning detection  
-  **Blink frequency analysis** (blinks per minute)  
-  **Micro-sleep detection** using time-based logic  
-  **User-specific calibration** (adaptive EAR threshold)  
-  **Multi-stage driver states**:
  - `CALIBRATING`
  - `NORMAL`
  - `DROWSY`
  - `DANGER`
-  **Threaded alarm system** (no video freeze)
-  **Look-away detection** using face-lost persistence
-  **State stabilization logic** to avoid flickering

---

##  Driver States

| State | Condition | Action |
|-----|---------|-------|
| CALIBRATING | Initial 5 seconds | Learns normal eye size |
| NORMAL | Safe driving | Green indicator |
| DROWSY | High blink rate / yawning | Warning alert |
| DANGER | Micro-sleep / Look-away | Emergency alarm |

---

##  Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe Tasks API**
- **Computer Vision**
- **Real-time Signal Processing**
- **Threading**

---

##  System Architecture

1. Webcam captures live video
2. MediaPipe detects 468 facial landmarks
3. EAR and MAR are computed from landmarks
4. Temporal logic analyzes fatigue patterns
5. State machine determines driver condition
6. Audio alerts trigger for unsafe states

---

##  Algorithms Used

### Eye Aspect Ratio (EAR)
Detects eye closure using geometric distances between eye landmarks.

### Mouth Aspect Ratio (MAR)
Detects yawning based on mouth opening.

### Temporal Analysis
- Rolling window blink-rate calculation
- Time-based micro-sleep detection
- Face-lost persistence detection

---

##  Calibration Phase (Personalization)

At startup, the system runs a **5-second calibration**:
- User looks normally at the camera
- Average EAR is calculated
- EAR threshold is set as `0.7 × average_EAR`

This improves robustness across different users and eye shapes.
git clone https://github.com/your-username/driver-micro-sleep-detection.git
cd driver-micro-sleep-detection
