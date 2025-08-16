import cv2
import numpy as np
from PoseModule import PoseDetector

# Initialize pose detector with confidence thresholds
pose_detector = PoseDetector(
    min_detection_confidence=0.75, min_tracking_confidence=0.65
)

# Initialize counters
count = 0
next_direction = "up"

# Open webcam
webcam = cv2.VideoCapture(0)
webcam.set(propId=cv2.CAP_PROP_FPS, value=60)


while True:
    is_successful, frame = webcam.read()
    if not is_successful:
        break

    # Detect pose
    landmark_list = pose_detector.findPose(frame=frame, draw=False)

    if len(landmark_list) > 0:
        # Calculate arm angle (shoulder–elbow–wrist)
        angle = pose_detector.findAngle(frame=frame, id1=12, id2=14, id3=16)

        # Map angle to percentage and bar position
        percentage = np.interp(x=angle, xp=[50, 150], fp=[100, 0])
        bar_value = np.interp(x=angle, xp=[50, 150], fp=[140, 400])
        color = (255, 0, 255)

        # Update rep count logic
        if percentage == 100:
            color = (0, 255, 0)
            if next_direction == "up":
                count += 0.5
                next_direction = "down"

        if percentage == 0:
            color = (0, 255, 0)
            if next_direction == "down":
                count += 0.5
                next_direction = "up"

        # --- Rep counter box ---
        (text_width, text_height), _ = cv2.getTextSize(
            text=str(int(count)),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=6,
            thickness=8,
        )
        cv2.rectangle(
            img=frame,
            pt1=(0, 350),
            pt2=(text_width + 10, 480),
            color=(220, 220, 220),
            thickness=cv2.FILLED,
        )
        cv2.putText(
            img=frame,
            text=str(int(count)),
            org=(10, 460),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=6,
            color=(255, 0, 255),
            thickness=8,
        )

        # --- Percentage bar ---
        cv2.rectangle(
            img=frame, pt1=(560, 140), pt2=(605, 400), color=color, thickness=3
        )
        cv2.rectangle(
            img=frame,
            pt1=(560, int(bar_value)),
            pt2=(605, 400),
            color=color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            img=frame,
            text=str(int(percentage)) + "%",
            org=(555, 450),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1.1,
            color=color,
            thickness=3,
        )

    # Display frame
    cv2.imshow(winname="frame", mat=frame)

    # Exit on ESC
    key = cv2.waitKey(delay=1)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
