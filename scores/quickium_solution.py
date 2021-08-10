import sys
import os
sys.path.append(os.path.abspath(os.getcwd() + "/blazepose/"))
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
from calc_score import get_maxes_and_mins, is_max_extension, is_min_extension, pull_score, is_max_extension, is_min_extension, get_arm_angle_array
from BlazeposeDepthaiEdge import BlazeposeDepthai

LINES_BODY = [[28,30,32,28,26,24,12,11,23,25,27,29,31,27], 
			  [23,24],
			  [22,16,18,20,16,14,12], 
			  [21,15,17,19,15,13,11],
			  [8,6,5,4,0,1,2,3,7],
			  [10,9]]

def get_frame_body_points(pose):    
    frame, body = pose.next_frame()   
    body_points = {}
    if body:
        for i, x_y in enumerate(body.landmarks[:33,:2]):
            body_points[i]=x_y
    return frame, body_points

def draw_body_points(frame, body_points):
	bdp_list = []
	for index in body_points:
		bdp_list.append(body_points[index])
	
	lines = [np.array([np.array(bdp_list).reshape(33,2).astype('int')[point,:] for point in line]) for line in LINES_BODY]
	cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)

	for body_point_index in range(33):
		x = body_points[body_point_index][0]
		y = body_points[body_point_index][1]
		if body_point_index > 10:
			color = (0,255,0) if body_point_index%2==0 else (0,0,255)
		elif body_point_index == 0:
			color = (0,255,255)
		elif body_point_index in [4,5,6,8,10]:
			color = (0,255,0)
		else:
			color = (0,0,255)
		cv2.circle(frame, (x, y), 4, color, -1)

	return frame

def count_repetition():
    global repetitions
    repetitions += 1
    print(f"Reps: {repetitions}")

def show_scores(scores):
    print(f"Constancy: {scores[0]}/1000")
    print(f"Amplitude: {scores[1]}/1000")
    print(f"Straight spine: {scores[2]}/1000")
    print(f"Frequency: {scores[3]}/1000")

def animate(i):
    global bd_points_list, wrist_l, wrist_r, find_max, first_one

    frame, bd_points = get_frame_body_points(pose)
    if bd_points:
        frame = draw_body_points(frame, bd_points)
        wrist_r.append(-bd_points[15][1])
        wrist_l.append(-bd_points[16][1])
        if len(wrist_l) > 90:
            wrist_r.pop(0)
            wrist_l.pop(0)
        ax.clear()
        ax.plot(wrist_r, 'r')
        ax.plot(wrist_l, 'g')
        if find_max:
            if is_max_extension(bd_points[15], bd_points[16], bd_points[13], bd_points[14], bd_points[11], bd_points[12], bd_points[0], 
                                np.mean((get_arm_angle_array([bd_points[15]], [bd_points[13]], [bd_points[11]]), 
                                            get_arm_angle_array([bd_points[16]], [bd_points[14]], [bd_points[12]]))), tol=10):
                if first_one:
                    first_one = False
                else:
                    count_repetition()
                find_max = False
        else:
            if is_min_extension(bd_points[15], bd_points[16], bd_points[13], bd_points[14], bd_points[11], bd_points[12], 
                                np.mean((get_arm_angle_array([bd_points[15]], [bd_points[13]], [bd_points[11]]), 
                                            get_arm_angle_array([bd_points[16]], [bd_points[14]], [bd_points[12]]))), tol=10):
                find_max = True
        cv2.imshow("video", frame)
        points = deepcopy(bd_points)
        points['time'] = time.time()
        bd_points_list.append(points)
        key = cv2.waitKey(42)
        if key == 27 or key == ord('q'):
            maxes, mins = get_maxes_and_mins(bd_points_list, tol=10)
            if maxes and mins:
                scores, _ = pull_score(bd_points_list, maxes, mins)
                show_scores(scores)
            else:
                print("Front Pull exercise was not detected.")
                print("Impossible to calculate a score.")
            exit()

pose = BlazeposeDepthai()
fig = plt.figure()
ax = plt.axes()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
find_max = True
first_one = True
bd_points_list = []
wrist_l = []
wrist_r = []
repetitions = 0
ani = animation.FuncAnimation(fig, animate, interval=42)
plt.show()