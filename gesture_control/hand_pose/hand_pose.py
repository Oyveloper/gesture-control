from enum import Enum

from numpy.lib import math
from gesture_control.util.util import distance, vec_between_points, angle
import mediapipe as mp

hand_marks = mp.solutions.hands.HandLandmark


class HandPose(Enum):
  """
  Includes an enum definition of the recognizable HandPoses
  """
  OPEN = 1
  THUMBS_UP = 2
  SPIDER_MAN = 3
  ROCK = 4
  OK = 5
  PEACE = 6
  SURFS_UP = 7
  OTHER = 99


def get_hand_pose(finger_extension, hand_pose_landmark) -> HandPose:
  """
  returns the hand pose based on the finger_extension array and the hand_pose landmarks
  """
  horisontal_vector = [1, 0, 0]

  thumb_tip = hand_pose_landmark[hand_marks.THUMB_TIP]
  thumb_base = hand_pose_landmark[hand_marks.THUMB_MCP]
  thumb_vec = vec_between_points(thumb_base, thumb_tip)
  index_tip = hand_pose_landmark[hand_marks.INDEX_FINGER_TIP]

  pinch_distance = distance(thumb_tip, index_tip)

  if finger_extension == [1] * 5:
    if pinch_distance < 0.1:
      return HandPose.OK
    return HandPose.OPEN
  elif finger_extension == [1, 0, 0, 0, 0] and 1/4 * math.pi <= angle(thumb_vec, horisontal_vector) <= 3/4 * math.pi:
    return HandPose.THUMBS_UP
  elif finger_extension == [1, 1, 0, 0, 1]:
    return HandPose.SPIDER_MAN
  elif finger_extension == [0, 1, 0, 0, 1]:
    return HandPose.ROCK
  elif finger_extension == [0, 1, 1, 0, 0]:
    return HandPose.PEACE
  elif finger_extension == [1, 0, 0, 0, 1]:
    return HandPose.SURFS_UP
  else:
    return HandPose.OTHER
