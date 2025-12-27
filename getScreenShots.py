import pyautogui
import numpy as np
import keyboard
import matplotlib
import matplotlib.pyplot as plt
import cv2 
import time
import math
#matplotlib.use("GTK3Agg")

CLEAR_WHEN_SAVE = True

size = 2**8

t_lower = 25#40  # Lower Threshold 
t_upper = 150#175 #200  # Upper threshold 

timeBetween = 1

print("ctrl to start")
while keyboard.is_pressed("ctrl") == False: pass

print("starting")
frameArray = []
takenPhoto = False
#startTime = time.time()
prevTime = 0
while keyboard.is_pressed("u") == False:
    currentTime = math.floor(time.time())
    if currentTime - prevTime >= 1:
        #startTime = time.time()
        im1 = pyautogui.screenshot()
        im1 = im1.resize((size,size))
        im1 = np.array(im1)
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY, 0)
        frameArray.append(im1.astype(np.uint8).copy())
        #time.sleep(timeBetween - (time.time()- startTime))
        print(currentTime - prevTime)
        prevTime = currentTime

#print(frameArray[0].shape)
#input()

    


print("Saving data")
arr = np.array(frameArray, dtype=np.uint8)
f = open("imgData_newLua.txt", ("w" if CLEAR_WHEN_SAVE else "a") + "b")
f.write(arr.tobytes())
f.close()
print("Stopping")

'''
    if math.floor(time.time() - startTime) % (timeBetween) == 0:
        if takenPhoto == False:
            #print("taken screenshot", time.time())
            im1 = pyautogui.screenshot()
            im1 = im1.resize((size,size))
            im1 = np.array(im1)
            im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY, 0)

            ##edge = cv2.Canny(im1, t_lower, t_upper)
            #plt.imshow(im1)
            #plt.show()
            #cv2.imshow('edgeBig', cv2.resize(edge, (600, 600),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)) 
            ##edge = cv2.resize(edge, (100,100),fx=0, fy=0, interpolation = cv2.INTER_AREA)
            #cv2.imshow('original', cv2.resize(im1, (600, 600),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)) 
            #cv2.imshow('edge', cv2.resize(edge, (600, 600),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)) 
            #cv2.waitKey(0) 
            #cv2.destroyAllWindows() 
            frameArray.append(im1.astype(np.uint8).copy())
            takenPhoto = True
    else:
        takenPhoto = False

'''

