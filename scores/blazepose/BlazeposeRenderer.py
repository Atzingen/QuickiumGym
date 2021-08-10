import cv2
import numpy as np
#import open3d as o3d
#from o3d_utils import create_segment, create_grid


# LINES_BODY is used when drawing the skeleton onto the source image. 
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
LINES_BODY = [[28,30,32,28,26,24,12,11,23,25,27,29,31,27], 
                [23,24],
                [22,16,18,20,16,14,12], 
                [21,15,17,19,15,13,11],
                [8,6,5,4,0,1,2,3,7],
                [10,9],
                ]

# LINE_MESH_BODY and COLORS_BODY are used when drawing the skeleton in 3D. 
rgb = {"right":(0,1,0), "left":(1,0,0), "middle":(1,1,0)}
LINE_MESH_BODY = [[9,10],[4,6],[1,3],
                    [12,14],[14,16],[16,20],[20,18],[18,16],
                    [12,11],[11,23],[23,24],[24,12],
                    [11,13],[13,15],[15,19],[19,17],[17,15],
                    [24,26],[26,28],[32,30],
                    [23,25],[25,27],[29,31]]

COLORS_BODY = ["middle","right","left",
                    "right","right","right","right","right",
                    "middle","middle","middle","middle",
                    "left","left","left","left","left",
                    "right","right","right","left","left","left"]
COLORS_BODY = [rgb[x] for x in COLORS_BODY]


class BlazeposeRenderer:
    def __init__(self, pose, show_fps = True):
        self.pose = pose
        self.show_fps = show_fps

        # Rendering flags
        self.show_landmarks = True

    def draw_landmarks(self, body):
        if self.show_landmarks:                
            list_connections = LINES_BODY
            lines = [np.array([body.landmarks[point,:2] for point in line]) for line in list_connections]
            # lines = [np.array([body.landmarks_padded[point,:2] for point in line]) for line in list_connections]
            cv2.polylines(self.frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)
            
            # for i,x_y in enumerate(body.landmarks_padded[:,:2]):
            for i,x_y in enumerate(body.landmarks[:self.pose.nb_kps,:2]):
                if i > 10:
                    color = (0,255,0) if i%2==0 else (0,0,255)
                elif i == 0:
                    color = (0,255,255)
                elif i in [4,5,6,8,10]:
                    color = (0,255,0)
                else:
                    color = (0,0,255)
                cv2.circle(self.frame, (x_y[0], x_y[1]), 4, color, -11)
                cv2.putText(self.frame, f'{x_y[0]}, {x_y[1]}', (x_y[0], x_y[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

    def draw(self, frame, body):
        self.frame = frame
        if body:
            self.draw_landmarks(body)
        return self.frame
    
    def exit(self):
        pass

    def waitKey(self, delay=1):
        if self.show_fps:
                self.pose.fps.draw(self.frame, orig=(50,50), size=1, color=(240,180,100))
        cv2.imshow("Blazepose", self.frame)
        key = cv2.waitKey(delay) 
        if key == 32:
            # Pause on space bar
            cv2.waitKey(0)
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        return key
        
            
