# 🚀 Stereo_VIO_ROS2

> 🎯 **Stereo Visual-Inertial Odometry (VIO) using ROS2 on EuRoC MH_01 dataset (C++ version)**

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue?style=for-the-badge&logo=ros)
![Status](https://img.shields.io/badge/Status-Experimental-success?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-EuRoC-orange?style=for-the-badge)


---

## 🧠 Overview

This repository demonstrates **Stereo Visual-Inertial Odometry (VIO)** using **ROS2**, tested on the **EuRoC MH_01 easy dataset**.

✨ It combines:
- 📷 Stereo camera input  
- 📊 IMU data  
- 🧭 Real-time pose estimation  

---

## 🎥 Demo

<p align="center">
  <img src="https://github.com/user-attachments/assets/c45570d4-7b12-4bdd-8da0-d56a115d3094" width="700"/>
</p>

---

## ⚙️ Features

- 🔄 Stereo image processing  
- 📈 IMU fusion  
- 📍 Real-time trajectory estimation  
- 🧪 Tested on EuRoC dataset  
- 🧩 ROS2 modular architecture  

---

---

## 📚 References & Inspiration

This implementation is inspired by well-known Visual-Inertial Odometry frameworks and research:

### 🔹 MSCKF (Multi-State Constraint Kalman Filter)

- Mourikis, A. I., & Roumeliotis, S. I.  
  *"A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation"*  
- 🔗 https://ieeexplore.ieee.org/document/4209642  

💡 A foundational filter-based VIO approach widely used for real-time systems.

---

### 🔸 OpenVINS

- Geneva, P., Eckenhoff, K., Yang, Y., & Huang, G.  
  *"OpenVINS: A Research Platform for Visual-Inertial Estimation"*  
- 🔗 https://github.com/rpng/open_vins  

💡 A modern open-source VIO framework implementing MSCKF-style estimation.

---

## 🎓 Suggested Reading

<details>
<summary>📘 Learn the theory behind this project</summary>

- Visual-Inertial Odometry basics  
- Kalman filtering for state estimation  
- Feature tracking & triangulation  
- Sensor fusion (camera + IMU)  

</details>

---

## 🧠 Notes

- This project follows a **filter-based VIO approach** (inspired by MSCKF)  
- Designed for **learning and experimentation** rather than production use  
- Tested on **EuRoC MH_01 easy dataset**  

---
