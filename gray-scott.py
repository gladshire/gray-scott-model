import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from PIL import Image
from scipy.signal import convolve2d
import subprocess
import sys
import os
import time



class gsSystem:
    def __init__(self, imagePath, diffRateU, diffRateV, feedRate, killRate, rxnRate, timeStep):
        img = Image.open(imagePath).convert('L')
        imgArray = (np.asarray(img) / 255.0)

        self.prefix = imagePath[0: imagePath.find('.')]
        self.rxnSpace = np.zeros((imgArray.shape[0], imgArray.shape[1], 2))
        self.rxnSpace[:, :, 0] = imgArray
        self.rxnSpace[:, :, 1] = 1.0 - imgArray
        self.diffRateU = diffRateU
        self.diffRateV = diffRateV
        self.feedRate = feedRate
        self.killRate = killRate
        self.rxnRate  = rxnRate
        self.timeStep = timeStep
        

    def renderMovie(self):
        cmd = ['ffmpeg', '-framerate', '30', '-i', os.path.join(self.prefix, 'frame_%06d.png'),
               '-b:v', '90M', '-vcodec', 'mpeg4', os.path.join("./", self.prefix + ".mp4")]
        subprocess.run(cmd)

    def laplacian(self, cArray):
        kernel = np.array([[0.05, 0.2, 0.05],
                           [0.2, -1.0, 0.2],
                           [0.05, 0.2, 0.05]])
        #kernel = np.array([[0, 1, 0],
        #                   [1, -4, 1],
        #                   [0, 1, 0]])
        return convolve2d(cArray, kernel, mode='same', boundary='wrap')

    def periodicBC(self, cArray):
        cArray[0, :] = cArray[-2, :]
        cArray[-1, :] = cArray[1, :]
        cArray[:, 0] = cArray[:, -2]
        cArray[:, -1] = cArray[:, 1]
        
    def diffuseStep(self):
        cU = self.rxnSpace[:, :, 0]
        cV = self.rxnSpace[:, :, 1]

        lapU = self.laplacian(cU)
        lapV = self.laplacian(cV)

        rxn = self.rxnRate * np.multiply(cU, np.multiply(cV, cV))

        diffU = self.diffRateU * lapU
        diffV = self.diffRateV * lapV

        dcU = (diffU - rxn + self.feedRate * (1 - cU)) * timeStep
        dcV = (diffV + rxn - ((self.feedRate + self.killRate) * cV)) * timeStep

        newCU = np.clip(cU + dcU, 0.0, 1.0)
        newCV = np.clip(cV + dcV, 0.0, 1.0)
        #newCU = cU + dcU
        #newCV = cV + dcV

        self.periodicBC(newCU)
        self.periodicBC(newCV)

        self.rxnSpace = np.stack([newCU, newCV], axis=-1)

    def runSim(self, numSteps):
        print("Initiating simulation")
        subprocess.run(['mkdir', self.prefix])
        for i in range(numSteps):
            fig, ax = plt.subplots(1, 1)
            plt.gca().invert_yaxis()
            currState = self.rxnSpace[:, :, 1]
            #ax.contourf(np.clip(self.rxnSpace[:, :, 1], 0.0, 1.0) * 255)
            #ax.contourf(self.rxnSpace[:, :, 1] * 255)
            ax.contourf((255 * (currState - currState.min()) / (currState.max() - currState.min())),
                       levels = 50)
            fig.savefig(os.path.join(self.prefix, f"frame_{i:06d}.png"), dpi=300)
            self.diffuseStep()
            plt.close(fig)
            print("{:.2f} %".format(i / numSteps * 100), end = '\r')
        print("Simulation complete")





if __name__ == "__main__":
    diffRateU = 1
    diffRateV = 0.5
    feedRate = 0.0545
    killRate = 0.062
    rxnRate = 1
    timeStep = 1
    photoPath = "horse.jpg"
    '''
    photoPath = str(sys.argv[1])
    diffRateU = float(sys.argv[2])
    diffRateV = float(sys.argv[3])
    feedRate  = float(sys.argv[4])
    killRate  = float(sys.argv[5])
    rxnRate   = float(sys.argv[6])
    timeStep  = float(sys.argv[7])
    '''
    sys = gsSystem(photoPath, diffRateU, diffRateV, feedRate, killRate, rxnRate, timeStep)
    sys.runSim(100000)

    sys.renderMovie()
