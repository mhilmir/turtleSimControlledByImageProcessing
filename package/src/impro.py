#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import math
import rospy
from geometry_msgs.msg import Twist
import trackingModule as tm

def controll():
    # get from webcam
    vid = cv.VideoCapture(0)
    # set resolution
    vid.set(3, 1280)
    vid.set(4, 720)

    # initialization 
    mode = str()  # variable for mode determination
    moveDir = Twist()  # move direction
    buttonMode2Trigg = [0, 0, 0, 0, 0, 0]  # trigger for mode 2

    rate = rospy.Rate(10)  # looping will occur in 10 hz speed
    while True:
        isTrue, frame = vid.read()
        frame = cv.flip(frame, 1)  # flipping image so what I see is like a mirror

        # make a copy to use in color detection
        coppiedFrame = frame.copy()

        # detecting face and hand
        frame = detectFace.findFace(frame)
        lmlistFace = detectFace.findPosition(frame, False)  # store face landmarks
        frame = detectHand.findHands(frame)
        lmlist1, lmlist2 = detectHand.findPosition(frame, draw=False)  # store hand landmarks
        # lmlist1 for left hand, lmlist2 for right hand

        # determine mode
        if lmlist1:
            # if finger touch mode1 box
            if (10 < lmlist1[8][1] < 150) & (10 < lmlist1[8][2] < 160):
                mode = "mode 1"
            # if finger touch mode2 box
            elif (10 < lmlist1[8][1] < 150) & (170 < lmlist1[8][2] < 320):
                mode = "mode 2"
            # if finger touch mode3 box
            elif (10 < lmlist1[8][1] < 150) & (330 < lmlist1[8][2] < 480):
                mode = "mode 3"
            # if finger touch mode4 box
            elif (10 < lmlist1[8][1] < 150) & (490 < lmlist1[8][2] < 640):
                mode = "mode 4"
            
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # controll the turtle
        if lmlist2:
            moveDir = Twist()  # reset move direction
            buttonMode2Trigg = [0, 0, 0, 0, 0, 0]  # to reset trigger value in mode 2
            # controll with mode 1
            if mode == "mode 1":
                fingers2 = detectHand.fingersUp(lmlist2, "right")
                length, _ = detectHand.findDistance(8, 12, frame, "right", False)
                if fingers2 == [1, 1, 0, 0, 0]:
                    moveDir.linear.x = 1
                elif fingers2 == [0, 0, 0, 0, 1]:
                    moveDir.linear.x = -1
                elif fingers2 == [0, 1, 1, 0, 0]:
                    if length > 60:
                        moveDir.angular.z = 1
                    elif length < 60:
                        moveDir.angular.z = -1
                elif fingers2 == [1, 1, 1, 0, 0]:
                    if length > 60:
                        moveDir.linear.x = 1
                        moveDir.angular.z = 1
                    elif length < 60:
                        moveDir.linear.x = 1
                        moveDir.angular.z = -1
                elif fingers2 == [0, 1, 1, 0, 1]:
                    if length > 60:
                        moveDir.linear.x = -1
                        moveDir.angular.z = 1
                    elif length < 60:
                        moveDir.linear.x = -1
                        moveDir.angular.z = -1
                # adjust speed
                if lmlistFace:
                    moveDir = eyeAdjustSpeed(lmlistFace, moveDir, frame)
                pub.publish(moveDir)

            # controll with mode 2
            elif mode == "mode 2":
                if (480 < lmlist2[8][1] < 630) & (10 < lmlist2[8][2] < 100):
                    if (480 < lmlist2[12][1] < 630) & (10 < lmlist2[12][2] < 100):
                        moveDir.linear.x = 1
                        moveDir.angular.z = -1
                        buttonMode2Trigg[0] = 1
                    else:
                        moveDir.linear.x = 1
                        moveDir.angular.z = 1
                        buttonMode2Trigg[1] = 1
                elif (640 < lmlist2[8][1] < 790) & (10 < lmlist2[8][2] < 100):
                    moveDir.linear.x = 1
                    buttonMode2Trigg[2] = 1
                elif (800 < lmlist2[8][1] < 950) & (10 < lmlist2[8][2] < 100):
                    moveDir.linear.x = -1
                    buttonMode2Trigg[3] = 1
                elif (960 < lmlist2[8][1] < 1110) & (10 < lmlist2[8][2] < 100):
                    if (960 < lmlist2[12][1] < 1110) & (10 < lmlist2[12][2] < 100):
                        moveDir.angular.z = -1
                        buttonMode2Trigg[4] = 1
                    else:
                        moveDir.angular.z = 1
                        buttonMode2Trigg[5] = 1
                # adjust speed
                if lmlistFace:
                    moveDir = eyeAdjustSpeed(lmlistFace, moveDir, frame)
                pub.publish(moveDir)
        
        # controll with mode
        if mode == "mode 3":
            # convert to HSV
            hsvFrame = cv.cvtColor(coppiedFrame, cv.COLOR_BGR2HSV)

            # determine range for red color
            red_lower = np.array([170, 100, 100], np.uint8)
            red_upper = np.array([180, 255, 255], np.uint8)
            red_mask = cv.inRange(hsvFrame, red_lower, red_upper)
            # determine range for yellow color
            yellow_lower = np.array([25, 50, 50], np.uint8)
            yellow_upper = np.array([35, 255, 255], np.uint8)
            yellow_mask = cv.inRange(hsvFrame, yellow_lower, yellow_upper)
            # determine range for cyan color
            cyan_lower = np.array([90, 50, 50], np.uint8)
            cyan_upper = np.array([110, 255, 255], np.uint8)
            cyan_mask = cv.inRange(hsvFrame, cyan_lower, cyan_upper)
            # determine range for purple color
            purple_lower = np.array([130, 50, 50], np.uint8)
            purple_upper = np.array([145, 255, 255], np.uint8)
            purple_mask = cv.inRange(hsvFrame, purple_lower, purple_upper)
            
            # making image that only contain ones binary type
            allOnes = np.ones((5,5), "uint8")

            moveDir = Twist()  # reset move direction
            # find red contours
            red_mask = cv.dilate(red_mask, allOnes)
            contours, _ = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv.contourArea(contour)
                if (area > 20000):
                    moveDir.linear.x = 1
            # find yellow contours
            yellow_mask = cv.dilate(yellow_mask, allOnes)
            contours, _ = cv.findContours(yellow_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv.contourArea(contour)
                if (area > 20000):
                    moveDir.linear.x = -1
            # find cyan contours
            cyan_mask = cv.dilate(cyan_mask, allOnes)
            contours, _ = cv.findContours(cyan_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv.contourArea(contour)
                if (area > 20000):
                    moveDir.angular.z = 1
            # find purple contours
            purple_mask = cv.dilate(purple_mask, allOnes)
            contours, _ = cv.findContours(purple_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv.contourArea(contour)
                if (area > 20000):
                    moveDir.angular.z = -1
            # adjust speed
            if lmlistFace:
                moveDir = eyeAdjustSpeed(lmlistFace, moveDir, frame)
            pub.publish(moveDir)

        # controll with mode 4
        if (mode == "mode 4") & (len(lmlistFace) != 0):
            moveDir = Twist()  # reset move direction
            y1 = lmlistFace[168][2]-lmlistFace[159][2]
            y2 = lmlistFace[168][2]-lmlistFace[385][2]
            x = lmlistFace[10][1]-lmlistFace[151][1]
            #print("y1=", y1, "   y2=", y2, "   x=", x)
            if (y1 <= -12) & (y2 <= -12):
                moveDir.linear.x = 1
            elif (y1 >= 3) & (y2 >= 3):
                moveDir.linear.x = -1
            if (x <= -7):
                moveDir.angular.z = 1
                if y1 <= -30:
                    moveDir.linear.x = 1
                elif y1 >= -15:
                    moveDir.linear.x = -1
            elif (x >= 7):
                moveDir.angular.z = -1
                if y2 <= -30:
                    moveDir.linear.x = 1
                elif y2 >= -15:
                    moveDir.linear.x = -1
            # adjust speed
            if lmlist2:
                moveDir = fingersAdjustSpeed(lmlist2, moveDir, frame)
            pub.publish(moveDir)



        # putting mode information
        if mode:
            cv.putText(frame, f"we are in {mode}", (10, 690), cv.FONT_ITALIC, 2, (255, 0, 0), 4)   
        # mode 1
        cv.rectangle(frame, (10,10), (150,160), (0, 255, 0), -1)
        cv.putText(frame, "mode 1", (20, 70), cv.FONT_ITALIC, 1, (255, 0, 0))
        # mode 2
        cv.rectangle(frame, (10,170), (150,320), (0, 255, 0), -1)
        cv.putText(frame, "mode 2", (20, 240), cv.FONT_ITALIC, 1, (255, 0, 0))
        # mode 3
        cv.rectangle(frame, (10,330), (150,480), (0, 255, 0), -1)
        cv.putText(frame, "mode 3", (20, 400), cv.FONT_ITALIC, 1, (255, 0, 0))
        # mode 4
        cv.rectangle(frame, (10,490), (150,640), (0, 255, 0), -1)
        cv.putText(frame, "mode 4", (20, 560), cv.FONT_ITALIC, 1, (255, 0, 0))

        # Make a button based on triggerred mode
        # for mode 1
        if mode == "mode 1":
            cv.rectangle(frame, (10,10), (150,160), (255, 0, 0), -1)
            cv.putText(frame, "mode 1", (20, 70), cv.FONT_ITALIC, 1, (0, 0, 0))
        # for mode 2
        elif mode == "mode 2":
            cv.rectangle(frame, (10,170), (150,320), (255, 0, 0), -1)
            cv.putText(frame, "mode 2", (20, 240), cv.FONT_ITALIC, 1, (0, 0, 0))
            cv.rectangle(frame, (480, 10), (630, 100), (0, 0, 255), -1)
            cv.circle(frame, (550, 55), 40, (0, 0, 0), 4)
            cv.rectangle(frame, (640, 10), (790, 100), (0, 0, 255), -1)
            cv.putText(frame, "FW", (680, 75), cv.FONT_ITALIC, 2, (0, 0, 0), 5)
            cv.rectangle(frame, (800, 10), (950, 100), (0, 0, 255), -1)
            cv.putText(frame, "back", (840, 60), cv.FONT_ITALIC, 1, (0, 0, 0), 5)
            cv.rectangle(frame, (960, 10), (1110, 100), (0, 0, 255), -1)
            cv.circle(frame, (1035, 60), 20, (0, 0, 0), 6)
            if buttonMode2Trigg[0] == 1:
                cv.rectangle(frame, (480, 10), (630, 100), (99, 3, 114), -1)
                cv.circle(frame, (550, 55), 40, (0, 0, 0), 4)
            elif buttonMode2Trigg[1] == 1:
                cv.rectangle(frame, (480, 10), (630, 100), (253, 139, 255), -1)
                cv.circle(frame, (550, 55), 40, (0, 0, 0), 4)
            elif buttonMode2Trigg[2] == 1:
                cv.rectangle(frame, (640, 10), (790, 100), (253, 139, 255), -1)
                cv.putText(frame, "FW", (680, 75), cv.FONT_ITALIC, 2, (0, 0, 0), 5)
            elif buttonMode2Trigg[3] == 1:
                cv.rectangle(frame, (800, 10), (950, 100), (253, 139, 255), -1)
                cv.putText(frame, "back", (840, 60), cv.FONT_ITALIC, 1, (0, 0, 0), 5)
            elif buttonMode2Trigg[4] == 1:
                cv.rectangle(frame, (960, 10), (1110, 100), (99, 3, 114), -1)
                cv.circle(frame, (1035, 60), 20, (0, 0, 0), 6)
            elif buttonMode2Trigg[5] == 1:
                cv.rectangle(frame, (960, 10), (1110, 100), (253, 139, 255), -1)
                cv.circle(frame, (1035, 60), 20, (0, 0, 0), 6)
        # for mode 3
        elif mode == "mode 3":
            cv.rectangle(frame, (10,330), (150,480), (255, 0, 0), -1)
            cv.putText(frame, "mode 3", (20, 400), cv.FONT_ITALIC, 1, (0, 0, 0))
        # for mode 4
        elif mode == "mode 4":
            cv.rectangle(frame, (10,490), (150,640), (255, 0, 0), -1)
            cv.putText(frame, "mode 4", (20, 560), cv.FONT_ITALIC, 1, (0, 0, 0))

        # show the result
        cv.imshow("impro", frame)

        # press q to close camera
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        rate.sleep()  # make sure looping run in 10 hz

def eyeAdjustSpeed(lmlistFace, moveDir, img):
    length, _ = detectFace.findDistance(144, 159, img, False)
    if moveDir.linear.x == 1:
        if length < 8:
            moveDir.linear.x += -0.5
        elif length > 16:
            moveDir.linear.x += 1.5
    elif moveDir.linear.x == -1:
        if length < 8:
            moveDir.linear.x += 0.5
        elif length > 16:
            moveDir.linear.x += -1.5
    
    return moveDir

def fingersAdjustSpeed(lmlist, moveDir, img):
    length, _ = detectHand.findDistance(8, 12, img, "right", False)
    if moveDir.linear.x == 1:
        if length < 35:
            moveDir.linear.x += -0.25
        elif length < 50:
            moveDir.linear.x += -0.5
        elif length > 90:
            moveDir.linear.x += 1
        elif length > 105:
            moveDir.linear.x += 2
    elif moveDir.linear.x == -1:
        if length < 35:
            moveDir.linear.x += 0.25
        elif length < 50:
            moveDir.linear.x += 0.5
        elif length > 90:
            moveDir.linear.x += -1
        elif length > 105:
            moveDir.linear.x += -2
    return moveDir
            
if __name__ == "__main__":
    # node & publisher initialization
    rospy.init_node("impro", anonymous=False)
    pub = rospy.Publisher("/turtle1/cmd_vel", Twist, queue_size=10)

    # object initialization
    detectHand = tm.handDetector(detectionCon=0.8, maxHands=2)
    detectFace = tm.faceDetector(detectionCon=0.5, trackCon=0.5)

    try:
        controll()
    except rospy.ROSInterruptException:
        pass