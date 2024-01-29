import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from PIL import Image
from scipy.signal import convolve2d
import subprocess
import os
import sys
import time
import gc



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
        cmd = ['ffmpeg', '-framerate', '1000', '-i', os.path.join(self.prefix, 'frame_%06d.png'),
               '-b:v', '90M', '-vcodec', 'mpeg4', os.path.join("./", self.prefix + ".mp4")]
        subprocess.run(cmd)

    def laplacian(self, cArray):
        kernel = np.array([[0.05, 0.2, 0.05],
                           [0.2, -1.0, 0.2],
                           [0.05, 0.2, 0.05]])
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

        self.periodicBC(newCU)
        self.periodicBC(newCV)

        self.rxnSpace = np.stack([newCU, newCV], axis=-1)

    def runSim(self, numSteps, dumpRate):
        print("Initiating simulation")
        subprocess.run(['mkdir', self.prefix])
        figArray = []
        for i in range(numSteps):
            fig, ax = plt.subplots(1, 1)
            fig.set_figheight(10)
            fig.set_figwidth(10)
            plt.gca().invert_yaxis()
            plt.axis('off')
            currState = self.rxnSpace[:, :, 1]
            ax.contourf((255 * (currState - currState.min()) / (currState.max() - currState.min())),
                        levels = 50, cmap = 'gray')
            figArray.append(fig)
            if len(figArray) == dumpRate:
                for j, fig in enumerate(figArray):
                    fig.savefig(os.path.join(self.prefix, f"frame_{i - (dumpRate - j):06d}.png"),
                                dpi=50, bbox_inches='tight')
                    plt.close(fig)
                figArray.clear()
                figArray = []
                gc.collect()
            self.diffuseStep()
            plt.close(fig)
            print("{:.2f} %".format(i / numSteps * 100), end = '\r')
        print("Simulation complete")





if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:\n  python3 gray-scott.py feedRate killRate photoPath") 
    else:
        diffRateU = 1
        diffRateV = 0.5
        #feedRate = 0.0545
        #killRate = 0.062
        feedRate = float(sys.argv[1])
        killRate = float(sys.argv[2])
        photoPath = sys.argv[3]
        rxnRate = 1
        timeStep = 1
    
        system = gsSystem(photoPath, diffRateU, diffRateV, feedRate, killRate, rxnRate, timeStep)

        print("Running Gray-Scott model with:")
        print("  feedRate: {}".format(feedRate))
        print("  killRate: {}".format(killRate))
        print("  initial:  {}".format(photoPath))

        system.runSim(100000, 500)

        system.renderMovie()
