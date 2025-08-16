import math
import cv2
from mediapipe.python.solutions import pose, drawing_utils


class PoseDetector:
    """
    Detects human body landmarks using MediaPipe's Pose solution.
    Provides methods to calculate angles and distances between landmarks.
    """

    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        """
        Initializes the PoseDetector object.

        Args:
            static_image_mode (bool): Process images as static (True) or as a video
            stream (False).
            model_complexity (int): Complexity of the pose landmark model (0, 1, or 2).
            smooth_landmarks (bool): If True, reduces landmark jitter.
            enable_segmentation (bool): If True, enables segmentation mask output.
            smooth_segmentation (bool): Smooth segmentation mask across frames.
            min_detection_confidence (float): Minimum confidence for pose detection.
            min_tracking_confidence (float): Minimum confidence for landmark tracking.
        """

        # Store parameters
        self.mode = static_image_mode
        self.model = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_thresh = min_detection_confidence
        self.tracking_thresh = min_tracking_confidence

        # Initialize MediaPipe Pose
        self.pose_detector = pose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.model,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.detection_thresh,
            min_tracking_confidence=self.tracking_thresh,
        )

    def findPose(self, frame, draw=True):
        """
        Detects pose landmarks in a frame.

        Args:
            frame (ndarray): Input BGR image.
            draw (bool): If True, draws detected landmarks on the frame.

        Returns:
            list: List of dictionaries containing x, y, z coordinates of each landmark.
        """
        self.landmark_list = []

        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe Pose
        self.results = self.pose_detector.process(image=rgb_frame)
        self.detection = self.results.pose_landmarks

        if self.detection:
            if draw:
                # Draw pose landmarks & connections
                drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=self.detection,
                    connections=pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_utils.DrawingSpec(
                        color=(0, 0, 255), thickness=cv2.FILLED, circle_radius=4
                    ),
                    connection_drawing_spec=drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2
                    ),
                )

            # Store all landmark coordinates in pixels
            for position in self.detection.landmark:
                x, y, z = (
                    int(position.x * frame_width),
                    int(position.y * frame_height),
                    int(position.z * frame_width),
                )
                self.landmark_list.append({"x": x, "y": y, "z": z})

        return self.landmark_list

    def findAngle(self, frame, id1, id2, id3, draw=True):
        """
        Calculates the angle between three landmarks.

        Args:
            frame (ndarray): Image for optional drawing.
            id1, id2, id3 (int): Landmark indices.
            draw (bool): If True, draws the angle lines and points.

        Returns:
            float: Angle in degrees.
        """
        # Get coordinates of the three landmarks
        x1, y1 = self.landmark_list[id1]["x"], self.landmark_list[id1]["y"]
        x2, y2 = self.landmark_list[id2]["x"], self.landmark_list[id2]["y"]
        x3, y3 = self.landmark_list[id3]["x"], self.landmark_list[id3]["y"]

        # Calculate angle using arctangent
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )

        if draw:
            # Draw lines between points
            cv2.line(
                img=frame,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(255, 255, 255),
                thickness=3,
            )
            cv2.line(
                img=frame,
                pt1=(x2, y2),
                pt2=(x3, y3),
                color=(255, 255, 255),
                thickness=3,
            )
            # Draw circles on the landmarks
            cv2.circle(
                img=frame,
                center=(x1, y1),
                radius=10,
                color=(0, 0, 255),
                thickness=cv2.FILLED,
            )
            cv2.circle(
                img=frame, center=(x1, y1), radius=15, color=(0, 0, 255), thickness=2
            )
            cv2.circle(
                img=frame,
                center=(x2, y2),
                radius=10,
                color=(0, 0, 255),
                thickness=cv2.FILLED,
            )
            cv2.circle(
                img=frame, center=(x2, y2), radius=15, color=(0, 0, 255), thickness=2
            )
            cv2.circle(
                img=frame,
                center=(x3, y3),
                radius=10,
                color=(0, 0, 255),
                thickness=cv2.FILLED,
            )
            cv2.circle(
                img=frame, center=(x3, y3), radius=15, color=(0, 0, 255), thickness=2
            )

        return angle

    def findDistance(self, frame, id1, id2, draw=True):
        """
        Calculates the Euclidean distance between two landmarks.

        Args:
            frame (ndarray): Image for optional drawing.
            id1, id2 (int): Landmark indices.
            draw (bool): If True, draws the distance line and circles.

        Returns:
            dict: Distance and coordinate details.
        """
        # Get coordinates of the two landmarks
        x1, y1 = self.landmark_list[id1]["x"], self.landmark_list[id1]["y"]
        x2, y2 = self.landmark_list[id2]["x"], self.landmark_list[id2]["y"]

        # Euclidean distance
        distance = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5

        # Midpoint between landmarks
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            # Draw endpoints and midpoint
            cv2.circle(
                img=frame,
                center=(x1, y1),
                radius=15,
                color=(255, 0, 255),
                thickness=cv2.FILLED,
            )
            cv2.circle(
                img=frame,
                center=(x2, y2),
                radius=15,
                color=(255, 0, 255),
                thickness=cv2.FILLED,
            )
            cv2.line(
                img=frame, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=3
            )
            cv2.circle(
                img=frame,
                center=(cx, cy),
                radius=15,
                color=(255, 0, 255),
                thickness=cv2.FILLED,
            )

        return {
            "distance": distance,
            "id1x": x1,
            "id1y": y1,
            "id2x": x2,
            "id2y": y2,
            "center x": cx,
            "center y": cy,
        }


def main():
    """Run a simple webcam test for pose estimation."""
    webcam = cv2.VideoCapture(1)
    pose_detector = PoseDetector()

    while True:
        is_successful, frame = webcam.read()
        if not is_successful:
            break

        # Detect and draw pose
        pose_detector.findPose(frame)
        cv2.imshow(winname="frame", mat=frame)

        # Exit on ESC key
        key = cv2.waitKey(delay=1)
        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
