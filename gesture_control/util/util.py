import math
import numpy as np


def distance(a, b):
  return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def angle(vec_a, vec_b):
  """
  vec_a: vector
  vec_b: vector

  :return: the angle between the two vectors
  """
  unit_a = vec_a / np.linalg.norm(vec_a)
  unit_b = vec_b / np.linalg.norm(vec_b)
  dot = np.dot(unit_a, unit_b)
  angle = np.arccos(dot)

  return angle


def vec_between_points(a, b):
  return [(b.x - a.x), (b.y - a.y),
          (b.z - a.z)]
