from gesture_control.hand_pose.hand_pose import get_hand_pose
from gesture_control.finger_extension.finger_extension import get_finger_extension
import mediapipe as mp
import cv2
from gesture_control.util import distance

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def main():
  cap = cv2.VideoCapture(0)
  with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      ret, frame = cap.read()

      # convert the colorspace to RGB for processing
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image = cv2.flip(image, 1)

      # performance gains when not writeable
      image.flags.writeable = False
      results = hands.process(image)
      image.flags.writeable = True

      # recolor back to BGR to display properly with opencv
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      image_height, image_width, _ = image.shape

      if results.multi_hand_landmarks:
        # Draw the hand landmarks on top of the image
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS
          )

          # further processing based on the landmarks such as pose and count
          finger_extensions = get_finger_extension(hand_landmarks.landmark)
          pose = get_hand_pose(finger_extensions, hand_landmarks.landmark)
          cv2.putText(image, f"Count: {sum(finger_extensions)}", (10, 20),
                      cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

          cv2.putText(image, f"Pose: {pose.name}", (10, 50),
                      cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

      # display the result
      cv2.imshow('Hand detection', image)

      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
