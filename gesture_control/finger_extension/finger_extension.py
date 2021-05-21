from numpy.lib import math
from gesture_control.util.util import angle, vec_between_points
import mediapipe as mp
from gesture_control.util import distance

hand_marks = mp.solutions.hands.HandLandmark


def get_finger_extension(hand_pose_landmark):
  wrist = hand_pose_landmark[hand_marks.WRIST]
  pinky_base = hand_pose_landmark[hand_marks.PINKY_MCP]

  points = [hand_pose_landmark[point] for point in hand_marks]

  result = []

  # handle thumb separatly
  tip = points[4]
  middle = points[3]
  result.append(distance(tip, pinky_base) > distance(middle, pinky_base))

  # every other finger
  for i in range(8, 21, 4):
    dip = points[i-1]
    middle = points[i-2]
    base = points[i-3]

    vec_a = vec_between_points(middle, base)
    vec_b = vec_between_points(middle, dip)

    finger_angle = angle(vec_a, vec_b)
    result.append(finger_angle > math.pi / 1.5)

  return result
