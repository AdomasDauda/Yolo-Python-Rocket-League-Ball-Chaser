import numpy as np
import cv2

#while True:
#    name = random.randint(200000000,3000000000000000)
#    screenshot = pyautogui.screenshot()
#    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
#    cv2.imwrite(f'{name}.png',screenshot)
#    cv2.waitKey(2000)
from PIL import Image, ImageGrab
import time
from pynput.keyboard import Controller as Ckeyb, Listener
import pyautogui
import math
import pygame
from pynput.mouse import Controller as Cmouse, Button
import win32gui
import win32ui
from ctypes import windll


print('imported')

net = cv2.dnn.readNet("yolov4-tiny-custom_final.weights", "yolov4_tiny_custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
print('loaded model')

mouse = Cmouse()
keyboarda = Ckeyb()

#pygame.init()

width_display = 1680
height_display = 1080

width_resize = 1000
height_resize = 700

#gameDisplay = pygame.display.set_mode((width_resize,height_resize))

#clock = pygame.time.Clock()

#def image(x,y, img):
#    gameDisplay.blit(img, (x,y))

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

hwnd = win32gui.FindWindow(None, 'Rocket League (64-bit, DX11, Cooked)')
left, top, right, bot = win32gui.GetWindowRect(hwnd)
w = right - left
h = bot - top

hwndDC = win32gui.GetWindowDC(hwnd)
mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
saveDC = mfcDC.CreateCompatibleDC()


def scrnshot():

    # screen capture
    #img = ImageGrab.grab(bbox=(0,0,1680,1080))
    #img_np = np.array(img)
    #frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    #cv2.imwrite('frame.jpg', frame)
    global saveBitMap
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    # Change the line below depending on whether you want the whole window
    # or just the client area. 
    #result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    img = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)


    if result == 1:
        return img


def yolo(img):

    #img = cv2.imread('frame.jpg')

    img = cv2.resize(np.array(img), None, fx=1, fy=1)
    height, width, channels = img.shape

    classes = ["ball"]

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    start = time.time()
    outs = net.forward(output_layers)


    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            global instructions, ball, ball_circle_area, last_ball_location, differencex
            x, y, w, h = boxes[i]
            #cv2.rectangle(img, (x, y), (x + w, y + h), [0,200,200], 2)
            ball_coordinates = (x+round(w/2),y+round(h/2))
            ball_circle_radius = round((math.sqrt(h*w))/2)
            ball_circle_area = ball_circle_radius*ball_circle_radius*math.pi
            cv2.circle(img,ball_coordinates, ball_circle_radius, [0,0,255], 2)
            center_coordinates = (round(width_display/2), round(height_display/2))
            cv2.circle(img,center_coordinates, 3, [0,255,0], -1)
            cv2.line(img, center_coordinates, ball_coordinates, [0,255,0], 1, lineType=8)
            differencey = center_coordinates[0]-ball_coordinates[0]
            differencex = center_coordinates[1]-ball_coordinates[1]
            #difference = (abs(center_coordinates[0])-abs(ball_coordinates[0]))**2+(abs(center_coordinates[1])-abs(ball_coordinates[1]))**2
            #cv2.putText(img, str(difference), center_coordinates, font, 3, [0,255,0], 2)
            if differencey > 0:
                instructions = ['left', differencey]
                ball = True
            if differencey < 0:
                instructions = ['right', differencey]
                ball = True
    if len(boxes) == 0:
        ball = False
        try:
            last_ball_location = instructions[0]
        except:
            pass
        instructions=[]

    end = time.time()
    # show timing information on YOLO
    print("[INFO] Yolo took {:.6f} seconds".format(end - start))

    dsize = (width_resize, height_resize)

    # resize image
    img = cv2.resize(img, dsize)
    #cv2.imwrite('frame.jpg', img)

    return img


def on_press(key):
    global k
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keyswa

listener = Listener(on_press=on_press)
listener.start()


dic = {
    'forward': 'w',
    'left': 'a',
    'right': 'd'
}


ball = False
global k
k = 'c'
while k != 'q':
    yolo(scrnshot())
    #cv2.imshow('img', )
    #image(0,0, py_image)wa
    #print('ball: ',ball)
    if ball == True:

        keyboarda.press(dic.get('forward'))
        if instructions[1] not in range(-40,40):
            keyboarda.press(dic.get(instructions[0]))
            #cv2.waitKey(round(abs(instructions[1])/2))
            #keyboarda.release(dic.get(instructions[0]))
            if ball_circle_area > 10000 and instructions[1] in range(-100,100) and differencex in range(-75,75):
                mouse.release(Button.left)
                print(f'dodging, because circle are is {ball_circle_area}')
                mouse.click(Button.right, 2)
            #if ball_circle_area < 5000 and instructions[1] in range(-75,75):
            #    mouse.press(Button.left)
        if instructions[1] in range(-100,100):
            keyboarda.release('a')
            keyboarda.release('d')
    if ball == False:

        try:
            if last_ball_location == 'left':
                print('ball last seen on the left')
                keyboarda.press('a')
                keyboarda.release('d')
            if last_ball_location == 'right':
                print('ball last seen on the right')
                keyboarda.press('d')
                keyboarda.release('a')
        except:
            pass
    #if ball == False:
    #    keyboard.press('d')
    #pygame.display.update()
    #clock.tick(60)
    #for event in pygame.event.get():
    #    if event.type == pygame.QUIT:
    #        break

win32gui.DeleteObject(saveBitMap.GetHandle())
saveDC.DeleteDC()
mfcDC.DeleteDC()
win32gui.ReleaseDC(hwnd, hwndDC)
keyboarda.release('w')
cv2.destroyAllWindows()
#pygame.quit()