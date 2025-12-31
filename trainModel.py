import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#from torch import *
import torch.optim as optim
import random
import keyboard
import torch.nn.functional as F
import math
#import pygame
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

if torch.cuda.is_available():
  print("CUDA is available! Running on GPU:", torch.cuda.get_device_name(0))
  device = "cuda"
else:
  print("CUDA is not available. Running on CPU.")
  device = "cpu"

print(f"Using {device} device")

imgSize = 2**8
depthImageSize = 2**6
MAX_DEPTH =  100
BATCH_SIZE = 16
TRAINING_DATA_COUNTER = 6
POWER = 1
SAVE_NAME = "model.pt"
activationFunction =  nn.LeakyReLU(0.1)


def LoadScreenShots():
  lst = []
  folder = "/home/william/.minetest/mods/MinetestDepthNN/screenshots/"
  for i in range(TRAINING_DATA_COUNTER):
    fileName = f"{folder}{i}.png"
    image = cv2.imread(fileName)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY, 0)
    lst.append(image)
  lst = np.reshape(lst, (TRAINING_DATA_COUNTER, 1, imgSize, imgSize))
  return np.array(lst, dtype=np.float32) / 255.0
def LoadDepthBuffer():
  string = "/home/william/.minetest/worlds/TrainDepth/depth/out_new.txt"
  f = open(string, "r")
  content = f.read()
  f.close()
  print("read file")
  frameList = []
  frameListString = content.split("?")
  for frameString in frameListString[:-1]:
    #if len(frameString) < 2: break
    frameList.append([])
    elementListString = frameString.split("|")
    for elementString in elementListString[:-1]:
      frameList[-1].append(float(elementString))
  numberOfFrames = len(frameList)
  frameList = np.array(frameList, dtype=np.float32)
  print(frameList.shape)
  frameList = frameList.flatten()
  frameList = np.reshape(frameList, (numberOfFrames, 1, depthImageSize, depthImageSize)) / MAX_DEPTH
  resizedFrameList = frameList
  print(np.max(frameList))
  print(np.min(frameList))
  c = 62.0
  resizedFrameList = np.log10(c*resizedFrameList + 0.1) / math.log10(c+0.1)
  return resizedFrameList 

def color_jitter(img,
                 brightness=0.2,
                 contrast=0.2,
                 gamma=0.2):
  img = img.copy()

  # --- Brightness ---
  b = 1.0 + np.random.uniform(-brightness, brightness)
  img *= b

  # --- Contrast ---
  c = 1.0 + np.random.uniform(-contrast, contrast)
  mean = img.mean(axis=(0, 1), keepdims=True)
  img = (img - mean) * c + mean

  # --- Gamma ---
  g = 1.0 + np.random.uniform(-gamma, gamma)
  img = np.power(np.clip(img, 0, 1), g)

  return np.clip(img, 0, 1)
def DataAugmentation(inputData, outputData):
  in_orig = inputData.copy()
  out_orig = outputData.copy()

  # vertical and horizontal flips
  in_v = np.flip(in_orig, axis=2)
  in_h = np.flip(in_orig, axis=3)
  out_v = np.flip(out_orig, axis=2)
  out_h = np.flip(out_orig, axis=3)

  # concatenate originals with flips
  inputs = np.concatenate([in_orig, in_v, in_h], axis=0)
  outputs = np.concatenate([out_orig, out_v, out_h], axis=0)

  return inputs, outputs

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual        # add input to output
        out = self.relu(out)
        return out
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # -------- Encoder --------
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 256 → 128
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 128 → 64
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64 → 32
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32 → 16
            nn.ReLU()
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 16 → 8
            nn.ReLU()
        )

        # -------- Bottleneck (global context) --------
        self.bottleneck = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # -------- Decoder --------
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 8 → 16
        self.dec4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)   # 16 → 32
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)    # 32 → 64
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Conv2d(16, 1, 3, padding=1)  # log-depth output
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Bottleneck
        b = self.bottleneck(e5)

        # Decoder with skips
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        return self.out(d2)
if False:
  model = NeuralNetwork().to(device)
  input("Loading new model enter to continue might overwrite old model")
  pass
else:
  model = NeuralNetwork()  # Re-create the model structure
  model.load_state_dict(torch.load("model.pt", map_location=device))
  model.eval()
  model.to(device)
print(model)

screenShots = LoadScreenShots()
print("Loaded screenshots")
correctDepthBuffer = LoadDepthBuffer().copy()
print("Loded depth buffer")

print(screenShots.shape, correctDepthBuffer.shape)

#screenShots, correctDepthBuffer = DataAugmentation(screenShots, correctDepthBuffer)
#flippedInput, flippedOutput = DataAugmentation(screenShots, correctDepthBuffer)
#screenShots += list(flippedInput)
#correctDepthBuffer += list(flippedOutput)

minSize = min(screenShots.shape[0], correctDepthBuffer.shape[0])
screenShots = screenShots[:minSize]
correctDepthBuffer = correctDepthBuffer[:minSize]

print(screenShots.shape, correctDepthBuffer.shape)

testDataNumber = 1
testScreenShots = screenShots[-testDataNumber:]
testCorrectDepthBuffer = correctDepthBuffer[-testDataNumber:]
if testDataNumber != 0:
  screenShots = screenShots[:-testDataNumber]
  correctDepthBuffer = correctDepthBuffer[:-testDataNumber]

print(screenShots.shape)
print("-------")



print(screenShots.shape, correctDepthBuffer.shape)

minSize = min(screenShots.shape[0], correctDepthBuffer.shape[0])
#minSize = 10

screenShots = torch.tensor(screenShots[:minSize]).to(device)
correctDepthBuffer = torch.tensor(correctDepthBuffer[:minSize]).to(device)
testScreenShots = torch.tensor(testScreenShots[:testDataNumber]).to(device)
testCorrectDepthBuffer = torch.tensor(testCorrectDepthBuffer[:testDataNumber]).to(device)


def gradient_loss(pred, target):
    # pred, target: (B, 1, H, W)
    dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    dx_gt = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    dy_gt = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

    return torch.mean(torch.abs(dx_pred - dx_gt)) + \
           torch.mean(torch.abs(dy_pred - dy_gt))
def scale_invariant_loss(pred, target):
  d = pred - target
  return torch.mean(d**2) - torch.mean(d)**2
def charbonnier(x, eps=1e-3):
  return torch.sqrt(x*x + eps*eps)

print(screenShots.shape, correctDepthBuffer.shape)
#recalculate min size after removing test data
minSize = min(screenShots.shape[0], correctDepthBuffer.shape[0])

# create your optimizer
#optimizer = optim.SGD(model.parameters(), lr = 1e-3)
optimizer = optim.AdamW(model.parameters(), lr=1.0e-3)
lmbda = lambda epoch: 0.95
#scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

criterion = nn.L1Loss()#nn.MSELoss()#nn.L1Loss()#nn.SmoothL1Loss()#nn.L1Loss()

plt.ion()
plt.show()
print("k to stop training")

class MinecraftDataset(Dataset):
    def __init__(self, imgs, depths, augment=True):
        self.imgs = imgs
        self.depths = depths
        self.augment = augment
        self.jitter = T.ColorJitter(brightness=0.3, contrast=0.3)

    def __len__(self):
      return len(self.imgs)


    def __getitem__(self, idx):
        x = self.imgs[idx]
        y = self.depths[idx]

        if self.augment:
            x = self.jitter(x)

        return x, y

ds = MinecraftDataset(screenShots, correctDepthBuffer)
dl = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE)

lossList = []
for epoch in range(600,600):
  # in your training loop:
  stopTraining = False
  # zero the gradient buffers
     
  #for dataCounter in range(minSize):
  meanLoss = 0
  #batchNumber = math.ceil(minSize / BATCH_SIZE)
  #perm = torch.randperm(minSize)
  #screenShots = screenShots[perm]
  #correctDepthBuffer = correctDepthBuffer[perm]
  for batchCounter,batch in enumerate(dl):
    inputBatch, outputBatch = batch
    optimizer.zero_grad()
    out = model(inputBatch)
    l1_loss = criterion(out, outputBatch)
    #grad_loss = gradient_loss(out, outputBatch)
    #edge = sobel(inputBatch).detach()
    #loss = (torch.abs(out - outputBatch) * (1 - edge)).mean()
    #loss = (
    #    l1_loss
    #    + 0.5 * grad_loss
    #    + 0.1 * scale_invariant_loss(out, outputBatch)
    #)
    loss = (
        charbonnier(out - outputBatch).mean()
        + 0.5 * gradient_loss(torch.exp(out), torch.exp(outputBatch))
    )
    loss.backward()
    optimizer.step()
    meanLoss += loss.item()
    if batchCounter % 15 == 0:
      print(round(batchCounter / (len(ds.imgs) / BATCH_SIZE) * 100,4),"%, epoch=",epoch,end="")
    if keyboard.is_pressed("j"):
      plt.close()
      plt.plot([ (lossList[l] / lossList[l-1]) for l in range(1,len(lossList))])
      plt.plot(list(range(len(lossList))), [1 for i in range(len(lossList))])
      plt.show(block=True)
    if keyboard.is_pressed("k"):
      print("Stopping training")
      stopTraining = True
      break
  print("")
  #scheduler.step()
  #plt.imshow(np.reshape(out.cpu().detach().numpy(),(depthImageSize, depthImageSize)), cmap="hot")
  #plt.draw()
  #plt.pause(0.005)
  lossList.append(meanLoss)
  if epoch % 5 == 0:
    plt.close()
    plt.imshow(np.reshape(out[0].cpu().detach().numpy(),(depthImageSize, depthImageSize))**POWER, cmap="hot", vmin=0, vmax=1)
    plt.draw()
    plt.pause(0.005)
    print(epoch, f"loss: {meanLoss} mean over last 10 losses: {sum(lossList[-10:]) / 10}")
  #optimizer.step()    # Does the update  
  if (epoch % 1000 == 0 and epoch != 0) or stopTraining or epoch == minSize - 1:
    print("Saving model")
    torch.save(model.state_dict(), SAVE_NAME)
  if stopTraining:
    break
print("Saving model")
#torch.save(model.state_dict(), "model.pt")
print("Saved")

plt.close()
#plt.plot([ (lossList[l] / lossList[l-1]) for l in range(1,len(lossList))])
#plt.plot(list(range(len(lossList))), [1 for i in range(len(lossList))])
plt.plot(lossList)
plt.show(block=True)
#plt.pause(2.005)

with torch.no_grad():
  
  setSize = 1
  testCorrectDepthBuffer = correctDepthBuffer
  testScreenShots = screenShots
  out = model(testScreenShots)
  for testNumber in range(0,len(testCorrectDepthBuffer)):  
    #testNumber = random.randint(0,testDataNumber)
    for i in range(0,1):    #min(setSize, testDataNumber-testNumber)
      plt.subplot(setSize, 3, i*3+1)  
      plt.imshow(np.resize(testScreenShots[testNumber+i].cpu().detach().numpy(),(imgSize,imgSize)), cmap="hot")
      plt.title("Input") 
      plt.subplot(setSize, 3, i*3+2)
      plt.title("Output") 
      #out = model(torch.tensor(testScreenShots[testNumber+i]))
      plt.imshow(np.reshape(out[testNumber+i].cpu().detach().numpy(),(depthImageSize,depthImageSize))**POWER, cmap="hot", vmin=0, vmax=1)#depthImageSize, depthImageSize
      plt.subplot(setSize, 3, i*3+3)
      plt.title("Correct depth") 
      plt.imshow(testCorrectDepthBuffer[testNumber+i][0].cpu().detach().numpy()**POWER, cmap="hot", vmin=0, vmax=1)
    plt.show(block=True)
    print(testNumber)

print("Done")
