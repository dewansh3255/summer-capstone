# Meeting Summary — audio.mp3
**Duration:** ~8 min 40 sec | **Language:** English | **Transcribed via:** OpenAI Whisper

---

## 1. Hardware & System Handover

- The professor/supervisor will hand over **the full system (hardware + setup)** to you on the **coming Saturday or Sunday**.
- Along with the handover, they will give you a **personal walkthrough** of how to use it.
- They noted that you already have a head start because you've gone through the **README files** provided earlier.

---

## 2. Sanskar's Involvement

- **Sanskar** (a team member / senior) is expected to be back on campus by **5 or 6 PM**.
- He will give you an overview of:
  - The **code he has built**
  - The **frontend** of the system
  - How **participants will perform gestures**
  - How **data will be collected** during those sessions

---

## 3. Data Collection Plan

- **Target:** Collect data from **5 participants** in **2 days**.
  - Suggested split: **3 participants on Day 1**, **2 on Day 2**
- The protocol involves:
  - **10 gestures**
  - **3 orientations per gesture**
- Estimated time per participant: **~40–50 minutes**
- Data collected per session will include:
  - **EMG data** (from Delsys sensors)
  - **Image / depth data** (from the Meta Quest) — this combined dataset (EMG + Meta Quest) is what the pipeline will be built on

---

## 4. Synchronization — Clarification of a Doubt

- A doubt was raised: *"Do we have synchronization between EMG and image data, or only timestamps?"*
- **Answer from the supervisor:** The Python code **handles synchronization**.
  - When the script is started, **both IMU and image data collection begin simultaneously** — it functions like a custom **Lab Streaming Layer (LSL)**.
  - If there is any **lag or drift** between the two data streams, it will be handled during **preprocessing**.

---

## 5. Pre-Data-Collection Tasks (Hands-On Phase)

Before the hardware is handed over, you are expected to complete hands-on exploration of **two modalities**:

### 5a. Computer Vision / Image Modality
- Go through **MediaPipe** and **OpenCV**
- A **paper/report** has already been shared — it shows how to **crop/remove the background** from gesture images
- **Task:** Take **2–3 gesture photos yourself** (e.g., thumbs up, victory sign) using your phone **against a white background**, then:
  1. Apply the **MediaPipe + OpenCV pipeline** to remove the background
  2. Isolate just the hand gesture
  3. **Calculate Chamfer Distance** on the gesture shapes
- Use the shared report as guidance for building this pipeline

### 5b. EMG Modality
- Review the **EMG Bench paper** already shared (covers how EMG signals are preprocessed)
- Note: Some papers convert EMG signals into **heatmap images** — but **this project will keep them as raw signals**, not convert to images
- **Task:** Search for **"EMG-based gesture recognition"** pipelines online to understand simple processing workflows
- Pick **one open-source EMG dataset** (suggested options: **NinaPro, DB1, DB2**, among others)
  - You do **not** need to preprocess all subjects — just **pick one subject**
  - Understand the **raw data format** and learn how preprocessing is done
  - Inform the supervisor which dataset you've chosen

---

## 6. Weekly Deliverable

- **Deadline: Next Friday**
- Compile everything into a **document** and share it with the supervisor, including:
  - What you explored and implemented (both modalities)
  - Results from your analysis
  - Any difficulties or blockers you encountered

---

## 7. Collaboration & Help

- **Sanskar** and **Kunal** can help with EMG-related questions
- **Mudit** — you have a good communication channel with him, and he is currently working on **EMG gesture datasets** himself. Reach out to him if you face issues with:
  - EMG preprocessing
  - Choosing the right dataset to start with

---

## 8. Communication Expectation

- **Don't wait until the next meeting** to ask questions — reach out as soon as a doubt comes up, otherwise the entire week is wasted
- You confirmed you will ask immediately if anything comes up

---

## 9. Closing

- The supervisor confirmed this set of tasks is **the agenda until the next meeting**
- Meeting ended with thanks from both sides
