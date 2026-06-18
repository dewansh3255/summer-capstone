# DepthVR

## Unity Version Requirement
This project must be opened using the following Unity version:

```text
Unity 6000.3.2f1
```

Using a different version may cause package or XR compatibility issues.

---

# Project Overview
DepthVR is a VR gesture interaction project designed for standalone VR headsets such as Meta Quest devices.

The application allows users to:
- Navigate the interface using hand tracking
- Perform gestures using the right hand
- Collect gesture data through a configured server URL

---

# Before Running the Project

## 1. Configure the Server URL
Before building the APK, the server URL must be configured so the collected data can be sent to the correct IP address.

### Steps
1. Open the project in Unity.
2. In the **Hierarchy** panel, locate:

```text
XR Data Pipeline
```

3. Select the object.
4. In the **Inspector** panel, find the field for the server URL/IP.
5. Enter the server URL or IP address where the gesture data should be collected.

Example:

```text
http://192.168.1.100:5000
```

---

# Building and Running the APK

## Steps to Build

1. In Unity, go to:

```text
File -> Build Profiles
```

2. Under Platform Settings, locate:

```text
Run Device
```

3. From the dropdown menu, select your connected Quest headset/device.

4. Click:

```text
Build and Run
```

5. Unity will begin building the APK.
6. After the build is complete, the application will automatically launch on the connected headset.

---

# Accessing the Application in the Headset

After installation:

1. Open the application library/menu in the headset.
2. Navigate to:

```text
Unknown Sources
```

3. Launch:

```text
DepthVR
```

---

# Hand Gesture Controls

## Navigation Controls
Use the **left hand pinch gesture** to:
- Navigate menus
- Select UI elements
- Click buttons

## Gesture Performance
Use the **right hand** to:
- Perform the gestures shown in the application
- Interact with the gesture recording/training system

---

# Changing the Gesture Video Order

If you want to reorder the gesture demonstration videos:

## Steps
1. In the **Hierarchy** panel, locate:

```text
Gesture Video Manager
```

2. Select it.
3. In the **Inspector** panel:
- Locate the gesture/video list.
- Reorder the elements as required.

The new order will be reflected in the application.

---

# Notes
- Ensure the headset is connected properly before clicking Build and Run.
- Developer mode must be enabled on the VR headset.
- Hand tracking permissions/features should be enabled on the headset.
- The application is intended to be used with hand tracking enabled.

---

# Quick Summary

## Required Unity Version
```text
6000.3.2f1
```

## Main Controls
- Left Hand Pinch -> Navigation and button interaction
- Right Hand -> Perform gestures

## App Location in Headset
```text
Unknown Sources -> DepthVR
```

## Important Objects in Hierarchy
- XR Data Pipeline -> Configure server URL/IP
- Gesture Video Manager -> Reorder gesture videos

