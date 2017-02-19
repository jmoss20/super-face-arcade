import cv2
import dlib
import numpy
import math
import time

# For key input
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/PyUserInput")
import time

from pymouse import PyMouse
from pykeyboard import PyKeyboard

m = PyMouse()
k = PyKeyboard()

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

FRAME_CHEAT_SHEET = {   
                        '-40':  [-40, 0, 40],
                        '0':    [0, 40, -40],
                        '40':   [40, 0, -40],
                     }  

def calc_angle(img, cascade, predictor, fd, scale_factor):
    landmarks = None
    angle = 0
    delta_angle = 0
    rows, cols, _ = img.shape

    # Check for faces on raw img
    # Not at default angle, try extreme angles
    # GOOD PLACE FOR OPTIMIZATION
    for i in FRAME_CHEAT_SHEET[str(fd)]:
        # Rotate img by i
        if i != 0:
            M = cv2.getRotationMatrix2D((cols/2,rows/2),-1*i,1)
            dst = cv2.warpAffine(img,M,(cols,rows))
            rects = cascade.detectMultiScale(dst, scale_factor, 5)
        else:
            rects = cascade.detectMultiScale(img, scale_factor, 5)
        if len(rects) > 0:
            # Found a face, fuck ya
            if i != 0:
                cv2.imshow('webcam_angle', dst)
            delta_angle = i
            if i != 0:
                landmarks = get_landmarks(dst, rects, predictor)
            else:
                cv2.imshow('webcam_angle', img)
                landmarks = get_landmarks(img, rects, predictor)

    # Make sure we ended up with a face
    if landmarks == None and delta_angle == 0:
        print "No face detected."
        return None, delta_angle, 0, 0
    elif landmarks == None and delta_angle != 0:
        if delta_angle > 0:
            return landmarks, delta_angle, 100
        else:
            return landmarks, delta_angle, -100

    # Unrotate the face
    if i != 0:
        for i in [0, 1, 2, 3, 13, 14, 15, 16, 27, 28, 29, 30, 50, 58, 52, 56]:
            # Translate
            landmarks[i,0] -= (cols/2)
            landmarks[i,1] -= (rows/2)
        
            # Rotate
            landmarks[i,0] = (math.cos(math.radians(-1*delta_angle)) * landmarks[i,0]) + (-1*math.sin(math.radians(-1*delta_angle)) * landmarks[i,1])
            landmarks[i,1] = (math.sin(math.radians(-1*delta_angle)) * landmarks[i,0]) + (math.cos(math.radians(-1*delta_angle)) * landmarks[i,1])

            # Untranslate
            landmarks[i,0] += (cols/2)
            landmarks[i,1] += (rows/2)

    # Calculate angle
    # find lines
    top_line = map(lambda x: landmarks[x,1], [0, 16])
    mid_line = map(lambda x: landmarks[x,1], [1, 15])
    bottom_line = map(lambda x: landmarks[x,1], [2, 14])
    nose_line = map(lambda x: landmarks[x,0], [27, 30])
    # calc angles
    top_angle = top_line[0] - top_line[1]
    mid_angle = mid_line[0] - mid_line[1]
    bottom_angle = bottom_line[0] - bottom_line[1]
    nose_angle = nose_line[1] - nose_line[0]
    cross_avg = sum([top_angle, mid_angle, bottom_angle])/3.0
    # magnitudes
    nose_mult = (landmarks[1,0] - landmarks[15,0]) / (landmarks[27,1] - landmarks[30,1])
    angle = ((nose_angle * nose_mult) + cross_avg) / 2.0

    # Calculate mouth
    # find lines
    left_line = map(lambda x: [landmarks[x,0], landmarks[x,1]], [50, 58])
    right_line = map(lambda x: [landmarks[x,0], landmarks[x,1]], [52, 56])
    # calc distance
    left_distance = numpy.linalg.norm(numpy.array(left_line[0])-numpy.array(left_line[1]))
    right_distance = numpy.linalg.norm(numpy.array(right_line[0])-numpy.array(right_line[1]))
    # avg distances
    avg_mouth = (left_distance + right_distance) / 2.0
    
    return landmarks, delta_angle, angle, avg_mouth

def get_landmarks(img, rects, predictor):
    x, y, w, h = rects[0].astype(long)
    rect = dlib.rectangle(x, y, x+w, y+h)
    return numpy.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])

def paint_frame(img, landmarks, angle, delta_angle):
    if landmarks != None:
        # Copy frame to draw on
        img_an = img.copy()

        # Draw points
        for idx, point in enumerate(landmarks):
            if (idx in [1, 2, 3, 13, 14, 15, 27, 28, 29, 30, 50, 58, 52, 56]): 
                pos = (point[0, 0], point[0, 1])
                cv2.circle(img_an, pos, 3, color=(0, 255, 0))
            elif (idx in range(26) and delta_angle == 0):
                pos = (point[0, 0], point[0, 1])
                cv2.circle(img_an, pos, 3, color=(30, 30, 30))

        # Draw lines
        segment_1 = [1, 27, 15]
        segment_2 = [2, 28, 14]
        segment_3 = [3, 29, 13]
        segment_around = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]
        segment_nose = [27, 28, 29, 30]
        segment_mouth_l = [50, 58]
        segment_mouth_r = [52, 56]

        cv2.polylines(img_an, [numpy.array(map(lambda x: [landmarks[x,0], landmarks[x,1]], segment_1), numpy.int32).reshape((-1,1,2))], isClosed=False, color=(0, 255, 0), thickness=1)
        cv2.polylines(img_an, [numpy.array(map(lambda x: [landmarks[x,0], landmarks[x,1]], segment_2), numpy.int32).reshape((-1,1,2))], isClosed=False, color=(0, 255, 0), thickness=1)
        cv2.polylines(img_an, [numpy.array(map(lambda x: [landmarks[x,0], landmarks[x,1]], segment_3), numpy.int32).reshape((-1,1,2))], isClosed=False, color=(0, 255, 0), thickness=1)
        cv2.polylines(img_an, [numpy.array(map(lambda x: [landmarks[x,0], landmarks[x,1]], segment_mouth_l), numpy.int32).reshape((-1,1,2))], isClosed=False, color=(0, 255, 0), thickness=1)
        cv2.polylines(img_an, [numpy.array(map(lambda x: [landmarks[x,0], landmarks[x,1]], segment_mouth_r), numpy.int32).reshape((-1,1,2))], isClosed=False, color=(0, 255, 0), thickness=1)

        if delta_angle == 0:
            cv2.polylines(img_an, [numpy.array(map(lambda x: [landmarks[x,0], landmarks[x,1]], segment_around), numpy.int32).reshape((-1,1,2))], isClosed=True, color=(30, 30, 30), thickness=1)
            cv2.polylines(img_an, [numpy.array(map(lambda x: [landmarks[x,0], landmarks[x,1]], segment_nose), numpy.int32).reshape((-1,1,2))], isClosed=False, color=(0, 0, 255), thickness=2)

        # Show angle
        cv2.putText(img_an, str(round(angle)), (10,20),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.5,
                    color=(0, 255, 255))

        return img_an
    else:
        return img

def compute_action(angle, state, margin_in, margin_out):
    # Calc strength
    strength = math.fabs(angle) / 35.0 

    if state == None:
        if (angle > margin_out):
            return "LEFT", strength
        elif (angle < (0 - margin_out)):
            return "RIGHT", strength
        return None, 0
    else:
        if (angle > margin_in):
            return "LEFT", strength
        elif (angle < (0 - margin_out)):
            return "RIGHT", strength
        return None, 0

def Press(action):
    if action == "RIGHT":
        k.press_key('J')
    elif action == "LEFT":
        k.press_key('I')
    elif action == "MOUTH":
        k.press_key('A')

def Release(action):
    if action == "RIGHT":
        k.release_key('J')
    elif action == "LEFT":
        k.release_key('I')
    elif action == "MOUTH":
        k.release_key('A')

def take_action(action, strength, state, counter):
    limit = strength * 4
    if state == action:
        if counter > limit:
            Release(action)
            counter = 0
        elif counter == 0:
            Press(action)
            counter += 1
        else:
            counter += 1
    else:
        Release(state)
        Press(action)
        counter = 0
    return action, counter

def handle_mouth(mouth):
    if mouth > 25:
        Press("MOUTH")
    else:
        Release("MOUTH")

def smooth(angle, past_angles):
    past_angles = past_angles[1:]
    past_angles.append(angle)
    return sum(past_angles)/len(past_angles), past_angles

def main():
    cam = cv2.VideoCapture(0)
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    LAST_FRAME_DELTA = 0
    SCALE_FACTOR = 1.3
    past_angles = [0] * 3
    
    # Action
    current_action = None
    strength = 0
    state = None
    counter = 0

    # Main Loop
    while True:
        # Input & Preprocess
        ret_val, img = cam.read() # Webcam img
        img = cv2.flip(img, 1) # Flip img
        img = cv2.resize(img, (0,0), fx=0.35, fy=0.35) # Squish img
        
        # Analyze & Annotate
        landmarks, LAST_FRAME_DELTA, angle, mouth = calc_angle(img, cascade, predictor, LAST_FRAME_DELTA, SCALE_FACTOR)
        angle, past_angles = smooth(angle, past_angles)
        img = paint_frame(img, landmarks, angle, LAST_FRAME_DELTA) # Annotate frame
        cv2.imshow('webcam', img)

        # Figure out what direction to turn
        current_action, strength = compute_action(angle, state, 15, 10)
        state, counter = take_action(current_action, strength, state, counter);
        handle_mouth(mouth)
    
        # Break loop on esc
        if cv2.waitKey(1) == 27:
            break

    # Clean up, kill windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
