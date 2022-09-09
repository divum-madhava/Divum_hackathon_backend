from  run_head_pose_estimation import  head_pose_tracker
from run_eye_tracker import eye_tracker
import cv2


cap = cv2.VideoCapture(0)

# head_pose_tracker(cap)
eye_tracker(cap)