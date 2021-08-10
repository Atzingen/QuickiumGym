import numpy as np
from scipy import stats

def get_maxes_and_mins(body_points, tol):
    l_wrist = list(map(lambda parts: parts[15], body_points))
    r_wrist = list(map(lambda parts: parts[16], body_points))
    l_elbow = list(map(lambda parts: parts[13], body_points))
    r_elbow = list(map(lambda parts: parts[14], body_points))
    l_shoulder = list(map(lambda parts: parts[11], body_points))
    r_shoulder = list(map(lambda parts: parts[12], body_points))
    nose = list(map(lambda parts: parts[0], body_points))
    
    maxes = []
    minimums = []
    last_index = 0
    max_frame = 0
    min_frame = 0
    mode = 0

    #Calculate angle between forearm and arm
    left_angle_array = get_arm_angle_array(l_wrist, l_elbow, l_shoulder)
    right_angle_array = get_arm_angle_array(r_wrist, r_elbow, r_shoulder)


    angle_array = np.mean((left_angle_array,right_angle_array), axis=0)

    while max_frame != -1 or min_frame != -1:
        if mode == 0:
            max_frame, last_index = find_max_extension(l_wrist, r_wrist,
                                                       l_elbow, r_elbow,
                                                       l_shoulder, r_shoulder, nose,
                                                       angle_array, tol, last_index)
            if max_frame > -1:
                maxes.append(max_frame)
            mode = 1
        else:
            min_frame, last_index = find_min_extension(l_wrist, r_wrist,
                                                       l_elbow, r_elbow,
                                                       l_shoulder, r_shoulder,
                                                       angle_array, tol, last_index)
            if min_frame > -1:
                minimums.append(min_frame)
            mode = 0

    return maxes, minimums

def is_max_extension(l_wrist, r_wrist, l_elbow, r_elbow, l_shoulder, r_shoulder, nose, angle, tol):
    return bool(angle >= 130 - tol and l_wrist[1] < l_elbow[1] and r_wrist[1] < r_elbow[1] and l_shoulder[1] > l_elbow[1] and 
                r_shoulder[1] > r_elbow[1] and nose[1] < l_shoulder[1] and nose[1] < r_shoulder[1])

def is_min_extension(l_wrist, r_wrist, l_elbow, r_elbow, l_shoulder, r_shoulder, angle, tol):
    return bool(angle <= 30 + tol and l_wrist[1] < l_elbow[1] and r_wrist[1] < r_elbow[1] and l_shoulder[1] < l_elbow[1] and r_shoulder[1] < r_elbow[1])

def find_max_extension(l_wrist, r_wrist, l_elbow, r_elbow, l_shoulder, r_shoulder, nose, angle_array, tol, start_idx=0):
    found_one = False #flag to know if at least one frame has been found, and we can return the frames when the desired position is not found again
    frames = []       #empty list to append the desired frames

    for i in range(start_idx, len(angle_array)):
        angle = angle_array[i]
        if is_max_extension(l_wrist[i], r_wrist[i], l_elbow[i], r_elbow[i], l_shoulder[i], r_shoulder[i], nose[i], angle, tol):
            frames.append(i)
            if not found_one:
                found_one = True
        elif found_one:
            if frames:
                idx = list(angle_array[frames[0]:frames[-1]+1]).index(max(angle_array[frames[0]:frames[-1]+1]))
                return frames[idx], i #returns frame with the widest angle
    
    return -1, start_idx

def find_min_extension(l_wrist, r_wrist, l_elbow, r_elbow, l_shoulder, r_shoulder, angle_array, tol, start_idx=0):
    found_one = False #flag to know if at least one frame has been found, and we can return the frames when the desired position is not found again
    frames = []       #empty list to append the desired frames    

    for i in range(start_idx, len(angle_array)):
        angle = angle_array[i]
        if is_min_extension(l_wrist[i], r_wrist[i], l_elbow[i], r_elbow[i], l_shoulder[i], r_shoulder[i], angle, tol):
            frames.append(i)
            if not found_one:
                found_one = True
        elif found_one:
            if frames:
                idx = list(angle_array[frames[0]:frames[-1]+1]).index(min(angle_array[frames[0]:frames[-1]+1]))
                return frames[idx], i #returns frame with the least wide angle

    return -1, start_idx

def get_arm_angle_array(wrist, elbow, shoulder):
    angle_array = []
    for i in range(len(wrist)):
        forearm = wrist[i] - elbow[i]
        arm = elbow[i] - shoulder[i]      

        #Calculate angle between forearm and arm
        angle = 180 - angle_between_vectors(arm, forearm)
        angle_array.append(angle)

    return np.array(angle_array)

def pull_score(body_points, maxes, mins):
    lines = []

    for i in range(len(maxes)):
        if i < len(mins):
            line = [maxes[i],mins[i]]
            lines.append(line)
        else:
            break

        if i+1 < len(maxes):
            line = [mins[i],maxes[i+1]]
            lines.append(line)
        else:
            break
    
    r_squares = []

    for line in lines:
        l_wrist = list(map(lambda parts: parts[15][1], body_points[line[0]:line[1]]))
        r_wrist = list(map(lambda parts: parts[16][1], body_points[line[0]:line[1]]))

        x = np.arange(len(l_wrist))

        _, _, r_value, _, _ = stats.linregress(x, l_wrist)
        r_squares.append(r_value**2)

        _, _, r_value, _, _ = stats.linregress(x, r_wrist)
        r_squares.append(r_value**2)

    r_square_mean = round(np.mean(r_squares) * 1000)

    l_wrist_max = [parts[15]for i,parts in enumerate(body_points) if i in maxes]
    r_wrist_max = [parts[16]for i,parts in enumerate(body_points) if i in maxes]
    l_elbow_max = [parts[13]for i,parts in enumerate(body_points) if i in maxes]
    r_elbow_max = [parts[14]for i,parts in enumerate(body_points) if i in maxes]
    l_shoulder_max = [parts[11]for i,parts in enumerate(body_points) if i in maxes]
    r_shoulder_max = [parts[12]for i,parts in enumerate(body_points) if i in maxes]

    l_wrist_min = [parts[15]for i,parts in enumerate(body_points) if i in mins]
    r_wrist_min = [parts[16]for i,parts in enumerate(body_points) if i in mins]
    l_elbow_min = [parts[13]for i,parts in enumerate(body_points) if i in mins]
    r_elbow_min = [parts[14]for i,parts in enumerate(body_points) if i in mins]
    l_shoulder_min = [parts[11]for i,parts in enumerate(body_points) if i in mins]
    r_shoulder_min = [parts[12]for i,parts in enumerate(body_points) if i in mins]
   
    l_angle_max = get_arm_angle_array(l_wrist_max, l_elbow_max, l_shoulder_max)
    r_angle_max = get_arm_angle_array(r_wrist_max, r_elbow_max, r_shoulder_max)

    l_angle_min = get_arm_angle_array(l_wrist_min, l_elbow_min, l_shoulder_min)
    r_angle_min = get_arm_angle_array(r_wrist_min, r_elbow_min, r_shoulder_min)

    l_diff_max = np.abs(np.mean(l_angle_max) - 160)
    l_diff_min = np.abs(np.mean(l_angle_min) - 30)

    r_diff_max = np.abs(np.mean(r_angle_max) - 160)
    r_diff_min = np.abs(np.mean(r_angle_min) - 30)

    amp_score = round((1 - l_diff_max/160)*250 + (1 - r_diff_max/160)*250 +
                (1 - l_diff_min/30)*250 + (1 - r_diff_min/30)*250)

    l_hip = np.array([parts[23] for parts in body_points[maxes[0]:mins[-1]]])
    r_hip = np.array([parts[24] for parts in body_points[maxes[0]:mins[-1]]])
    l_shoulder = np.array([parts[11] for parts in body_points[maxes[0]:mins[-1]]])
    r_shoulder = np.array([parts[12] for parts in body_points[maxes[0]:mins[-1]]])

    vector_i = np.array((1,0))
    left_vectors = l_shoulder - l_hip
    right_vectors = r_shoulder - r_hip
    l_angle = np.mean(np.array([angle_between_vectors(v,vector_i) for v in left_vectors]))
    r_angle = np.mean(np.array([angle_between_vectors(v,vector_i) for v in right_vectors]))

    angle_diff = abs((l_angle + r_angle) - 180)
    spine_score = round((1 - angle_diff/180) * 1000)

    max_times = [parts['time'] for i,parts in enumerate(body_points) if i in maxes]
    min_times = [parts['time'] for i,parts in enumerate(body_points) if i in mins]

    max_times_diff = np.diff(max_times)
    min_times_diff = np.diff(min_times)

    max_times_mean = np.mean(max_times_diff)
    min_times_mean = np.mean(min_times_diff)
    max_times_deviation = np.std(max_times_diff)
    min_times_deviation = np.std(min_times_diff)
    
    freq_score = round((1 - max_times_deviation/max_times_mean) * 500 + (1 - min_times_deviation/min_times_mean) * 500)


    return (r_square_mean, amp_score, spine_score, freq_score), body_points[maxes[0]:mins[-1]]

def angle_between_vectors(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    
    angle = np.arccos(dot_product) * 180 / np.pi

    return angle