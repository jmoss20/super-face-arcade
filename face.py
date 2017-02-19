import cv2
import dlib
import numpy
import math

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

FRAME_CHEAT_SHEET = {   
                        '-55':  [-55, -35, 0, 35, 55],
                        '-35':  [-35, 0, -55, 35, 55],
                        '0':    [0, 35, -35, 55, -55],
                        '35':   [35, 0, 55, -35, -55],
                        '55':   [55, 35, 0, -35, -55]
                     } 
                        

def calc_angle(img, cascade, predictor, fd):
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
            M = cv2.getRotationMatrix2D((cols/2,rows/2),i,1)
            dst = cv2.warpAffine(img,M,(cols,rows))
            rects = cascade.detectMultiScale(dst, 1.3, 5)
        else:
            rects = cascade.detectMultiScale(img, 1.3, 5)
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
    if landmarks == None:
        print "Someone sit the fuck down and play, jesus christ"
        return None, delta_angle

    # Unrotate the face
    if i != 0:
        for i in [0, 1, 2, 3, 13, 14, 15, 16, 27, 28, 29, 30]:
            # Translate
            landmarks[i,0] -= (cols/2)
            landmarks[i,1] -= (rows/2)
        
            # Rotate
            landmarks[i,0] = (math.cos(math.radians(delta_angle)) * landmarks[i,0]) + (-1*math.sin(math.radians(delta_angle)) * landmarks[i,1])
            landmarks[i,1] = (math.sin(math.radians(delta_angle)) * landmarks[i,0]) + (math.cos(math.radians(delta_angle)) * landmarks[i,1])   

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

    print angle, delta_angle
 
    return landmarks, delta_angle

def get_landmarks(img, rects, predictor):
    x, y, w, h = rects[0].astype(long)
    rect = dlib.rectangle(x, y, x+w, y+h)
    return numpy.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])

def annotate_landmarks(img, landmarks):
    if landmarks != None:
        img_an = img.copy()
        for idx, point in enumerate(landmarks):
            if (idx in [1, 2, 3, 13, 14, 15, 27, 28, 29, 30]): 
                pos = (point[0, 0], point[0, 1])
                #cv2.putText(img_an, str(idx), pos,
                #   fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                #   fontScale = 0.4,
                #   color = (0, 0, 255))
                cv2.circle(img_an, pos, 3, color=(0, 0, 255))
        return img_an
    else:
        return img

def main():
    cam = cv2.VideoCapture(0)
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    LAST_FRAME_DELTA = 0

    # Main Loop
    while True:
        ret_val, img = cam.read() # Webcam img
        img = cv2.flip(img, 1) # Flip img
        img = cv2.resize(img, (0,0), fx=0.35, fy=0.35) # Squish img
        landmarks, LAST_FRAME_DELTA = calc_angle(img, cascade, predictor, LAST_FRAME_DELTA) # Get facial landmarks
        img = annotate_landmarks(img, landmarks) # Annotate w facial landmarks  
        cv2.imshow('webcam', img)   
 
        # Break loop
        if cv2.waitKey(1) == 27:
            break # end on esc

    # Clean up, kill windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
