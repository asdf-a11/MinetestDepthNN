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

PATH = "/home/william/.minetest/mods/MinetestDepthNN/screenshots/"

size = 2**8

t_lower = 25#40  # Lower Threshold 
t_upper = 150#175 #200  # Upper threshold 

timeBetween = 1

print("ctrl to start")
while keyboard.is_pressed("ctrl") == False: pass

print("starting")
frameArray = []
takenPhoto = False
frameCounter = 0
prevTime = 0
while keyboard.is_pressed("u") == False:
    currentTime = math.floor(time.time())
    if currentTime - prevTime >= 1:
        second_monitor_region = (2560, 0, 3840, 2160)  # (x, y, width, height)
        im1 = pyautogui.screenshot(region=second_monitor_region)
        im1 = im1.resize((size,size))
        im1 = np.array(im1)
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY, 0)

        cv2.imwrite(f"{PATH}{frameCounter}.png", im1)

        print(currentTime - prevTime, frameCounter)
        prevTime = currentTime
        frameCounter += 1

#print(frameArray[0].shape)
#input()

    


print("Saving data")
#arr = np.array(frameArray, dtype=np.uint8)
#f = open("imgData_newLua.txt", ("w" if CLEAR_WHEN_SAVE else "a") + "b")
#f.write(arr.tobytes())
#f.close()
print("Stopping")

