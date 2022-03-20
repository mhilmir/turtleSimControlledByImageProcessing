import cv2 as cv
import numpy as np
import time
import math
import mediapipe as mp

class faceDetector():
    def __init__(self, detectionCon=0.5, trackCon=0.5):
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    def findFace(self, img):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        self.results = self.face_mesh.process(imgRGB)
        imgRGB.flags.writeable = True
        return img

    def findPosition(self, img, draw=True):
        self.lmlistFace = []
        if self.results.multi_face_landmarks:
            myFace = self.results.multi_face_landmarks[0]
            for id, lm in enumerate(myFace.landmark):
                h, w = img.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlistFace.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 1, (255, 0, 0), -1)
                    cv.putText(img, f"{id}", (cx, cy), cv.FONT_ITALIC, 0.35, (0, 0, 0), 1)
        return self.lmlistFace
    
    def findDistance(self, p1, p2, img, draw=True, r=3, t=1):
        x1, y1 = self.lmlistFace[p1][1:]
        x2, y2 = self.lmlistFace[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex, self.detectionCon, self.trackCon)
        self.tipIds = [4, 8, 12, 16, 20]
 
    def findHands(self, img, draw=False):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
        return img
 
    def findPosition(self, img, draw=True):
        self.lmlist1 = []  # left hand
        self.lmlist2 = []  # right hand
        handsType = []
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_handedness:
                handsType.append(hand.classification[0].label)
            n = 0
            for handType in handsType:
                if handType == "Left":
                    myHand1 = self.results.multi_hand_landmarks[n]
                    for id, lm in enumerate(myHand1.landmark):
                        h, w, _ = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        self.lmlist1.append([id, cx, cy])
                        if draw:
                            cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
                elif handType == "Right":
                    myHand2 = self.results.multi_hand_landmarks[n]
                    for id, lm in enumerate(myHand2.landmark):
                        h, w, _ = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        self.lmlist2.append([id, cx, cy])
                        if draw:
                            cv.circle(img, (cx, cy), 5, (253, 139, 255), cv.FILLED)
                n = n + 1
        return self.lmlist1, self.lmlist2
 
    def fingersUp(self, lmlist, handType):
        fingers = []
        # Thumb
        if handType == "right":
            if lmlist[self.tipIds[0]][1] < lmlist[self.tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        elif handType == "left":
            if lmlist[self.tipIds[0]][1] > lmlist[self.tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        # Fingers
        for id in range(1, 5):
            if lmlist[self.tipIds[id]][2] < lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
 
    def findDistance(self, p1, p2, img, handtype, draw=True, r=15, t=3):
        if handtype == "right":
            x1, y1 = self.lmlist2[p1][1:]
            x2, y2 = self.lmlist2[p2][1:]
        elif handtype == "left":
            x1, y1 = self.lmlist1[p1][1:]
            x2, y2 = self.lmlist1[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img