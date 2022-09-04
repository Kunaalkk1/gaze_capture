# Dependencies
import cv2
import dlib
from math import hypot
import numpy as np

# Capture video stream
cap = cv2.VideoCapture(0)

# The dlib face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialization
blink_count = 0
prev_blink_left, prev_blink_right = 10, 10

# Function to get midpoint of the eye
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

# Function to get blink ratio of a particular eye
def blink_detector(eye_points, landmarks):
    left_point = (landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y)
    right_point = (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y)
    center_top = midpoint(landmarks.part(eye_points[1]), landmarks.part(eye_points[2]))
    center_bottom = midpoint(landmarks.part(eye_points[5]), landmarks.part(eye_points[4]))
    
    # Draw horizontal and vertical lines accross the particular eye

    cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
       
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    
    ratio = hor_line_length / ver_line_length

    return ratio

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        # Eye landmarks of both eyes
        landmarks = predictor(gray, face)

        left_eye_array = [36, 37, 38, 39, 40, 41]
        left_eye_ratio = blink_detector(left_eye_array, landmarks)
        right_eye_array = [42, 43, 44, 45, 46, 47]
        right_eye_ratio = blink_detector(right_eye_array, landmarks)
        
        # Blinking detection from ratio
        if left_eye_ratio > 5.7 and right_eye_ratio > 5.7 and not(prev_blink_left > 5.7 and prev_blink_right > 5.7):

            # To mention on the video frame that person blinked, uncomment the following line
            prev_blink_left = left_eye_ratio
            prev_blink_right = right_eye_ratio
            cv2.putText(frame, "Blinking", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
            # Get blink count
            blink_count += 1
        else:
            prev_blink_left = left_eye_ratio
            prev_blink_right = right_eye_ratio

        # From left_eye_ratio > 5.7 and right_eye_ratio > 5.7, you can count the blinking of individual eyes.

        # Gaze detection of the eyes

        # Generating a black frame to draw the eyes separately
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask1 = np.zeros(frame.shape[:2], np.uint8)
        mask2 = np.zeros(frame.shape[:2], np.uint8)


        # Get the left eye and right eye on the mask
        
        # Left eye ##############################################################
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)],
                                    np.int32)
        

        # Separate the left eye from the face
        min_left_x = np.min(left_eye_region[:, 0])
        max_left_x = np.max(left_eye_region[:, 0])
        min_left_y = np.min(left_eye_region[:, 1])
        max_left_y = np.max(left_eye_region[:, 1])

        left_eye = frame[min_left_y: max_left_y, min_left_x: max_left_x]
        left_eye = cv2.resize(left_eye, None, fx=3, fy=3)
        gray_left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        gray_left_eye = cv2.resize(gray_left_eye, None, fx=3, fy=3)
        _, threshold_left_eye = cv2.threshold(gray_left_eye, 70, 255, cv2.THRESH_BINARY)
        
        # Fill the mask with left eye region
        cv2.polylines(mask, [left_eye_region], True, 255, 1)
        cv2.fillPoly(mask, [left_eye_region], 255)
        cv2.polylines(mask2, [left_eye_region], True, 255, 1)
        cv2.fillPoly(mask2, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask=mask2)
        left_eye_copy = left_eye
        left_eye_copy = cv2.bitwise_and(gray, gray, mask=mask)

        gray_left_eye = left_eye[min_left_y: max_left_y, min_left_x: max_left_x]
        _, threshold_left_eye = cv2.threshold(gray_left_eye, 70, 255, cv2.THRESH_BINARY)
        
        
        # Right eye ##############################################################
        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                     (landmarks.part(43).x, landmarks.part(43).y),
                                     (landmarks.part(44).x, landmarks.part(44).y),
                                     (landmarks.part(45).x, landmarks.part(45).y),
                                     (landmarks.part(46).x, landmarks.part(46).y),
                                     (landmarks.part(47).x, landmarks.part(47).y)],
                                     np.int32)
        

        # Separate the right eye from the face
        min_right_x = np.min(right_eye_region[:, 0])
        max_right_x = np.max(right_eye_region[:, 0])
        min_right_y = np.min(right_eye_region[:, 1])
        max_right_y = np.max(right_eye_region[:, 1])

        right_eye = frame[min_right_y: max_right_y, min_right_x: max_right_x]
        right_eye = cv2.resize(right_eye, None, fx=3, fy=3)
        gray_right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        gray_right_eye = cv2.resize(gray_right_eye, None, fx=3, fy=3)
        _, threshold_right_eye = cv2.threshold(gray_right_eye, 70, 255, cv2.THRESH_BINARY)
        
        # Fill the mask with right eye region
        cv2.polylines(mask, [right_eye_region], True, 255, 1)
        cv2.fillPoly(mask, [right_eye_region], 255)
        cv2.polylines(mask1, [right_eye_region], True, 255, 1)
        cv2.fillPoly(mask1, [right_eye_region], 255)
        right_eye = cv2.bitwise_and(gray, gray, mask=mask1)
        right_eye_copy = right_eye
        right_eye_copy = cv2.bitwise_and(gray, gray, mask=mask)
        gray_right_eye = right_eye[min_right_y: max_right_y, min_right_x: max_right_x]
        _, threshold_right_eye = cv2.threshold(gray_right_eye, 70, 255, cv2.THRESH_BINARY)

        # We have got the left and right eye separately outlined #########################

        #  Face detection box - uncomment the next three lines to see

        x0, y0 = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        # Draw the left eye region
        cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 1)

        # Draw the right eye region
        cv2.polylines(frame, [right_eye_region], True, (0, 0, 255), 1)


    # Show the frame and the mask
    cv2.imshow("Face", frame)
    # cv2.imshow("Mask", mask)

    # Show left eye separately

    # cv2.imshow("Left eye", left_eye)
    cv2.imshow("Left eye outline", gray_left_eye)
    cv2.imshow("Threshold left eye", threshold_left_eye)

    # Show right eye separately

    # cv2.imshow("Right eye", right_eye)
    cv2.imshow("Right eye outline", gray_right_eye)
    cv2.imshow("Threshold right eye", threshold_right_eye)

    
    # Since mask is same for left_eye_copy and right_eye_copy, 
    # we can use it to show both eyes together by calling the variable that
    # we have used to show the right_eye_copy as it was created later

    # cv2.imshow("Both eyes", right_eye_copy)

    # Exits the loop when 'esc' is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# After breaking from loop, exit the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Get the blink count
# print("Blink count: ", blink_count)
