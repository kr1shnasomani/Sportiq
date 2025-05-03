from numpy import pi, ones, zeros, uint8, where, cos, sin, sqrt
import os
from mediapipe import solutions
from cv2 import VideoCapture, cvtColor, Canny, line, imshow, waitKey, destroyAllWindows, COLOR_BGR2GRAY, HoughLinesP
from cv2 import threshold, THRESH_BINARY, dilate, floodFill, circle, HoughLines, erode, rectangle, VideoWriter, VideoWriter_fourcc

from TraceHeader import videoFile, findIntersection, calculatePixels
from CourtMapping import courtMap, showLines, showPoint, heightP, widthP, givePoint
from BodyTracking import bodyMap
from BallDetection import BallDetector
from BallMapping import euclideanDistance, withinCircle

PROJECT_ROOT = "/Users/krishnasomani/Documents/Projects/Sportiq"
INPUT_VIDEO_PATH = f"{PROJECT_ROOT}/dataset/input.mp4"
MODEL_PATH = f"{PROJECT_ROOT}/model/TrackNet.pth"
FINAL_OUTPUT_PATH = f"{PROJECT_ROOT}/output/output.mp4"
OUTPUT_VIDEO_PATH = f"{PROJECT_ROOT}/output/output_video.mp4"

video = VideoCapture(INPUT_VIDEO_PATH)
width = int(video.get(3))
height = int(video.get(4))

fourcc = VideoWriter_fourcc(*'mp4v')
clip = VideoWriter(OUTPUT_VIDEO_PATH,fourcc,25.0,(widthP,heightP))
processedFrame = None

class crop1:
    x: float = 50/100
    xoffset: float = 0/100
    xcenter: int = 1 
    
    y: float = 33/100
    yoffset: float = 0/100
    ycenter: int = 0
    
class crop2:
    x: float = 83/100
    xoffset: float = 0/100
    xcenter: int = 1 
    
    y: float = 60/100
    yoffset: float = 40/100
    ycenter: int = 0

crop1 = calculatePixels(crop1, width, height)
crop2 = calculatePixels(crop2, width, height)

n = 3
counter = 0
mp_pose = solutions.pose

class body1:
    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25)
    x: int
    xAvg: float = 0
    y: int
    yAvg: float = 0
    
class body2:
    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25) 
    x: int
    xAvg: float = 0
    y: int
    yAvg: float = 0

extraLen = width/3
class axis:
    top = [[-extraLen,0],[width+extraLen,0]]
    right = [[width+extraLen,0],[width+extraLen,height]]
    bottom = [[-extraLen,height],[width+extraLen,height]]
    left = [[-extraLen,0],[-extraLen,height]]

NtopLeftP = None
NtopRightP = None
NbottomLeftP = None
NbottomRightP = None

ball_detector = BallDetector(MODEL_PATH, out_channels=2)
ballProximity = []
ball = None
lastSeen = None
handPoints = None
flag = [0,0,0,0]
coords = []
minDist1 = height*width
minDist2 = height*width
velocities = []

while video.isOpened():
    ret, frame = video.read()
    if frame is None:
        break
    
    gry = cvtColor(frame, COLOR_BGR2GRAY)
    bw = threshold(gry, 156, 255, THRESH_BINARY)[1]
    canny = Canny(bw, 100, 200)
    
    hPLines = HoughLinesP(canny, 1, pi/180, threshold=150, minLineLength=100, maxLineGap=10)
    intersectNum = zeros((len(hPLines),2))
    i = 0
    for hPLine1 in hPLines:
        Line1x1, Line1y1, Line1x2, Line1y2 = hPLine1[0]
        Line1 = [[Line1x1,Line1y1],[Line1x2,Line1y2]]
        for hPLine2 in hPLines:
            Line2x1, Line2y1, Line2x2, Line2y2 = hPLine2[0]
            Line2 = [[Line2x1,Line2y1],[Line2x2,Line2y2]]
            if Line1 is Line2:
                continue
            if Line1x1>Line1x2:
                temp = Line1x1
                Line1x1 = Line1x2
                Line1x2 = temp
                
            if Line1y1>Line1y2:
                temp = Line1y1
                Line1y1 = Line1y2
                Line1y2 = temp
                
            intersect = findIntersection(Line1, Line2, Line1x1-200, Line1y1-200, Line1x2+200, Line1y2+200)
            if intersect is not None:
                intersectNum[i][0] += 1
        intersectNum[i][1] = i
        i += 1

    i = p = 0
    dilation = dilate(bw, ones((5, 5), uint8), iterations=1)
    nonRectArea = dilation.copy()
    intersectNum = intersectNum[(-intersectNum)[:, 0].argsort()]
    for hPLine in hPLines:
        x1,y1,x2,y2 = hPLine[0]
        for p in range(8):
            if (i==intersectNum[p][1]) and (intersectNum[i][0]>0):
                floodFill(nonRectArea, zeros((height+2, width+2), uint8), (x1, y1), 1) 
                floodFill(nonRectArea, zeros((height+2, width+2), uint8), (x2, y2), 1) 
        i+=1
    dilation[where(nonRectArea == 255)] = 0
    dilation[where(nonRectArea == 1)] = 255
    eroded = erode(dilation, ones((5, 5), uint8)) 
    cannyMain = Canny(eroded, 90, 100)
    
    xOLeft = width + extraLen
    xORight = 0 - extraLen
    xFLeft = width + extraLen
    xFRight = 0 - extraLen
    
    yOTop = height
    yOBottom = 0
    yFTop = height
    yFBottom = 0
    
    hLines = HoughLines(cannyMain, 2, pi/180, 300)
    for hLine in hLines:
        for rho,theta in hLine:
            a = cos(theta)
            b = sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + width*(-b))
            y1 = int(y0 + width*(a))
            x2 = int(x0 - width*(-b))
            y2 = int(y0 - width*(a))
            
            intersectxF = findIntersection(axis.bottom, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            intersectyO = findIntersection(axis.left, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            intersectxO = findIntersection(axis.top, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            intersectyF = findIntersection(axis.right, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            
            if (intersectxO is None) and (intersectxF is None) and (intersectyO is None) and (intersectyF is None):
                continue
            
            if intersectxO is not None:
                if intersectxO[0] < xOLeft:
                    xOLeft = intersectxO[0]
                    xOLeftLine = [[x1,y1],[x2,y2]]
                if intersectxO[0] > xORight:
                    xORight = intersectxO[0]
                    xORightLine = [[x1,y1],[x2,y2]]
            if intersectyO is not None:
                if intersectyO[1] < yOTop:
                    yOTop = intersectyO[1]
                    yOTopLine = [[x1,y1],[x2,y2]]
                if intersectyO[1] > yOBottom:
                    yOBottom = intersectyO[1]
                    yOBottomLine = [[x1,y1],[x2,y2]]
                    
            if intersectxF is not None:
                if intersectxF[0] < xFLeft:
                    xFLeft = intersectxF[0]
                    xFLeftLine = [[x1,y1],[x2,y2]]
                if intersectxF[0] > xFRight:
                    xFRight = intersectxF[0]
                    xFRightLine = [[x1,y1],[x2,y2]]
            if intersectyF is not None:
                if intersectyF[1] < yFTop:
                    yFTop = intersectyF[1]
                    yFTopLine = [[x1,y1],[x2,y2]]
                if intersectyF[1] > yFBottom:
                    yFBottom = intersectyF[1]
                    yFBottomLine = [[x1,y1],[x2,y2]]

    yOTopLine[0][1] = yOTopLine[0][1]+4
    yOTopLine[1][1] = yOTopLine[1][1]+4
    
    yFTopLine[0][1] = yFTopLine[0][1]+4
    yFTopLine[1][1] = yFTopLine[1][1]+4
    
    topLeftP = findIntersection(xOLeftLine, yOTopLine, -extraLen, 0, width+extraLen, height)
    topRightP = findIntersection(xORightLine, yFTopLine, -extraLen, 0, width+extraLen, height)
    bottomLeftP = findIntersection(xFLeftLine, yOBottomLine, -extraLen, 0, width+extraLen, height)
    bottomRightP = findIntersection(xFRightLine, yFBottomLine, -extraLen, 0, width+extraLen, height)
    
    if (not(topLeftP == NtopLeftP)) and (not(topRightP == NtopRightP)) and (not(bottomLeftP == NbottomLeftP)) and (not(bottomRightP == NbottomRightP)):
        
        NtopLeftP = topLeftP
        NtopRightP = topRightP
        NbottomLeftP = bottomLeftP
        NbottomRightP = bottomRightP

    handPointsPrev = handPoints
    feetPoints, handPoints, nosePoints = bodyMap(frame, body1.pose, body2.pose, crop1, crop2)

    if (not any(item is None for sublist in feetPoints for item in sublist)) or (not any(item is None for sublist in handPoints for item in sublist)) or (not any(item is None for sublist in nosePoints for item in sublist)):
        if feetPoints[0][1] > feetPoints[1][1]:
            lowerFoot1 = feetPoints[0][1]
            higherFoot1 = feetPoints[1][1]
        else:
            lowerFoot1 = feetPoints[1][1]
            higherFoot1 = feetPoints[0][1]
            
        if feetPoints[2][1] > feetPoints[3][1]:
            lowerFoot2 = feetPoints[2][1]
            higherFoot2 = feetPoints[3][1]
        else:
            lowerFoot2 = feetPoints[3][1]
            higherFoot2 = feetPoints[2][1]
        
        body1.x = (feetPoints[0][0]+feetPoints[1][0])/2
        body1.y = lowerFoot1*0.8+higherFoot1*0.2

        body2.x = (feetPoints[2][0]+feetPoints[3][0])/2
        body2.y = lowerFoot2*0.8+higherFoot2*0.2
        
        counter += 1
        coeff = 1. / min(counter, n)
        body1.xAvg = coeff * body1.x + (1. - coeff) * body1.xAvg
        body1.yAvg = coeff * body1.y + (1. - coeff) * body1.yAvg
        body2.xAvg = coeff * body2.x + (1. - coeff) * body2.xAvg
        body2.yAvg = coeff * body2.y + (1. - coeff) * body2.yAvg
        
        circleRadiusBody1 = int(0.65 * euclideanDistance(nosePoints[0], [body1.x, body1.y]))
        circleRadiusBody2 = int(0.6 * euclideanDistance(nosePoints[1], [body2.x, body2.y]))
        
        # Distorting frame and outputting results
        processedFrame, M = courtMap(frame, NtopLeftP, NtopRightP, NbottomLeftP, NbottomRightP)
        # Create black background
        rectangle(processedFrame, (0,0),(967,1585),(188,145,103),2000)
        processedFrame = showLines(processedFrame)

        processedFrame = showPoint(processedFrame, M, [body1.xAvg,body1.yAvg])
        processedFrame = showPoint(processedFrame, M, [body2.xAvg,body2.yAvg])
        
        ballPrev = ball
        ball_detector.detect_ball(frame)
        if ball_detector.xy_coordinates[-1][0] is not None:
            ball = ball_detector.xy_coordinates[-1]
            lastSeen = counter
        
        circle(frame, (handPoints[1]), circleRadiusBody1, (255,0,0), 2) 
        circle(frame, (handPoints[3]), circleRadiusBody2, (255,0,0), 2)
        
        if ball is not None:
            circle(frame, ball, 4, (0,255,0), 3)
            circle(frame, ballPrev, 3, (0,255,0), 2)
            
            if ball is not ballPrev:
                if withinCircle(handPoints[1], circleRadiusBody1, ball):
                    if minDist1>euclideanDistance(handPoints[1], ball):
                        minDist1 = euclideanDistance(handPoints[1], ball)
                        coords.append((ball, givePoint(M, ball), givePoint(M, (body1.x,body1.y)), counter))
                else:
                    minDist1 = circleRadiusBody1
                if withinCircle(handPoints[3], circleRadiusBody2, ball):
                    if minDist2>euclideanDistance(handPoints[3], ball):
                        minDist2 = euclideanDistance(handPoints[3], ball)
                        coords.append((ball, givePoint(M, ball), givePoint(M, (body2.x,body2.y)), counter))
                else:
                    minDist2 = circleRadiusBody2


                if ball_detector.xy_coordinates[-2][0] is not None:
                    xVelocity = ball[0] - ballPrev[0]
                    yVelocity = (ball[1] - ballPrev[1])*(1+(height-ball[1])*0.4/height)
                    if withinCircle(handPoints[3], circleRadiusBody2, ball) or withinCircle(handPoints[1], circleRadiusBody1, ball):
                        within = True
                    else:
                        within = False
                    velocities.append(([xVelocity,yVelocity], counter, givePoint(M, ball), within))
                        

        if len(coords)>=2:
            if euclideanDistance(coords[-1][0], coords[-2][0]) < 200:
                del coords[-2]

        for i in range(len(coords)):
            circle(frame, coords[i][0], 4, (0,0,255), 4)

    if processedFrame is not None:    
        clip.write(processedFrame)
    imshow("Frame", frame)
    if waitKey(1) == ord("q"):
        break

if ball is not None:
    coords.append((ball, givePoint(M, ball), givePoint(M, ball), lastSeen))

clip.release()
video.release()
destroyAllWindows()

accelerations = []
for i in range(2,len(velocities)):
    if velocities[i][1]-2 == velocities[i-2][1] and velocities[i-1][3] is False:
        xAcceleration = (velocities[i][0][0]-velocities[i-2][0][0])/2
        yAcceleration = (velocities[i][0][1]-velocities[i-2][0][1])/2
        accelerations.append((int(yAcceleration), velocities[i][1]))
        if abs(yAcceleration) > (height/77):
            for k in range(len(coords)):
                if coords[k][3] > velocities[i-1][1]:
                    coords.insert(k, (velocities[i-1][2], velocities[i-1][2], velocities[i-1][2], velocities[i-1][1]))
                    break

ballArray = []
while len(coords)>1:
    time = coords[0][3]
    location = [coords[0][1][0],coords[0][2][1]]
    del coords[0]
    timeDiff = coords[0][3]-time
    for i in range(time, coords[0][3]):
        x = int(location[0]+((i-time)/timeDiff)*(coords[0][1][0]-location[0]))
        y = int(location[1]+((i-time)/timeDiff)*(coords[0][2][1]-location[1]))
        ballArray.append(((x,y), i))       

ballArray.append(((coords[0][1][0], coords[0][2][1]), coords[0][3]))

video = VideoCapture(OUTPUT_VIDEO_PATH)
clip = VideoWriter(FINAL_OUTPUT_PATH, fourcc, 25.0, (widthP, heightP))
counter = 0
writeFlag = False
while video.isOpened():
    ret, frame = video.read()
    if frame is None:
        break
    
    counter += 1

    for i in range(len(ballArray)):
        if counter == ballArray[i][1]:
            circle(frame, (ballArray[i][0]), 4,(0,255,255),3)
            break

    if counter == ballArray[0][1]:
        writeFlag = True

    if ballArray[-1][1] == counter:
        writeFlag = False
        
    if (writeFlag):
        index = counter - ballArray[0][1]
        circle(frame, (ballArray[index][0]), 2,(0,255,255),3)
    
    clip.write(frame)

video.release()
clip.release()
destroyAllWindows()

if os.path.exists(OUTPUT_VIDEO_PATH):
    os.remove(OUTPUT_VIDEO_PATH)