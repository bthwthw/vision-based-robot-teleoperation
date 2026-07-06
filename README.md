# Vision-based Robot Teleoperation
A real-time, markerless robotic teleoperation system using Intel RealSense D435, MediaPipe Hands, and cuRobo for 6-DoF arm and gripper control

## 📖 About The Project

This project presents a robust, real-time, and markerless robotic teleoperation framework. By leveraging computer vision and spatial depth mapping, the system allows human operators to control a 6-DoF robotic arm and a parallel gripper intuitively using bare hands.

The architecture is specifically designed to handle unstructured manipulation tasks and addresses common challenges in vision-based teleoperation, such as self-occlusion in top-down camera views, high-frequency spatial jitter, and kinematic coupling.

## ✨ Key Features

+ Markerless 3D Tracking: Utilizes MediaPipe Hands to extract 2D pixel coordinates $(u, v)$ and dynamically fuses them with aligned depth maps from an Intel RealSense D435 camera to reconstruct accurate 3D physical coordinates $(X, Y, Z)$ via de-projection.

+ Kinematic Decoupling: * Arm Control (TCP): The virtual Tool Center Point (TCP) is anchored at the middle finger base $(P_9)$ with a forward translational offset, preventing positional drift during finger articulation.

+ Gripper Control: The gripper aperture is independently mapped using the 3D Euclidean distance between the thumb tip $(P_4)$ and the index finger tip $(P_8)$.

+ Jitter Reduction: Implements the 1 Euro Filter to dynamically smooth the high-frequency jitter caused by the neural network's topological inference, balancing latency and stability.

+ Collision-free Motion Planning: Integrates NVIDIA cuRobo for GPU-accelerated inverse kinematics and obstacle avoidance.

## 🏗️ System Architecture

The data pipeline consists of 4 main stages:

+ 2D Perception & Synthetic Data Rejection: MediaPipe extracts 2D landmarks from the RGB frame. The synthetic Z-axis (relative depth) and 3D world landmarks from the neural network are strictly discarded to avoid statistical drift.

+ Depth Synchronization: The $(u, v)$ coordinates are used to query the actual physical depth $(Z)$ from the hardware-aligned depth map.

+ Space De-projection: The $(u, v, Z)$ data is fed into the pinhole camera's inverse model to reconstruct real-world $(X, Y, Z)$ metrics.

+ Teleoperation Mapping: The extracted TCP and quaternion orientations are filtered (1 Euro Filter) and sent to the IK solver to actuate the robot.

## 💻 Prerequisites & Installation

Hardware Requirements

+ Camera: Intel RealSense Depth Camera D435

+ Compute Node: PC/Laptop with a CUDA-enabled NVIDIA GPU (Required for cuRobo)

+ Robot: 6-DoF Manipulator (e.g., ABB, UR) equipped with a parallel gripper.

Software Dependencies

+ Ensure you have Python 3.8+ installed.

+ For NVIDIA cuRobo installation, please follow the official cuRobo setup guide as it requires specific CUDA toolkits.

## 🚀 Usage

(Provide a brief guide on how to start your main scripts here)

1. Start the RealSense vision pipeline & Hand Tracking module
-> python src/vision_pipeline.py

2. Launch the Robot Controller & IK Solver
-> python src/teleop_controller.py


## 📂 Project Structure

vision-based-robot-teleop/

├── src/

│   ├── vision_pipeline.py      # RealSense & MediaPipe integration

│   ├── kinematics.py           # Coordinate frames & De-projection math

│   ├── filters.py              # 1 Euro Filter implementation

│   └── teleop_controller.py    # Main node sending commands to the robot

├── docs/                       # Diagrams and detailed technical reports

├── requirements.txt

└── README.md
