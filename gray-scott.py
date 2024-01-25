import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from PIL import Image
from scipy.signal import convolve2d
import subprocess
import os


class gsSystem:
    def __init__(self, imagePath, diffRateU, diffRateV, feedRate, killRate, timeStep):
        img = Image.open(imagePath).convert('L')
        imgArray = (np.asarray(img) / 255.0)

        self.rxnSpace = np.zeros((imgArray.shape[0], imgArray.shape[1], 2))
        self.rxnSpace[:, :, 0] = imgArray
        self.rxnSpace[:, :, 1] = 1.0 - imgArray
        self.diffRateU = diffRateU
        self.diffRateV = diffRateV
        self.feedRate = feedRate
        self.killRate = killRate
        self.timeStep = timeStep

        self.stateArray = [copy.deepcopy(self.rxnSpace)]

    def renderMovie(self):
        cmd = ['ffmpeg', '-framerate', '500', '-i', os.path.join('./gsTemp', 'frame_%06d.png'),
               '-b:v', '90M', '-vcodec', 'mpeg4', os.path.join('./', 'movie.mp4')]
        subprocess.run(cmd)

    def laplacian(self, cArray):
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])
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

        rxn = np.multiply(cU, np.multiply(cV, cV))

        diffU = self.diffRateU * lapU
        diffV = self.diffRateV * lapV

        dcU = (diffU - rxn + self.feedRate * (1 - cU)) * timeStep
        dcV = (diffV + rxn - ((self.feedRate + self.killRate) * cV)) * timeStep

        newCU = cU + dcU
        newCV = cV + dcV

        self.periodicBC(newCU)
        self.periodicBC(newCV)

        self.rxnSpace = np.stack([newCU, newCV], axis=-1)

    def runSim(self, numSteps):
        print("Initiating simulation")
        subprocess.run(['mkdir', 'gsTemp'])
        for i in range(numSteps):
            self.diffuseStep()
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.contourf((self.rxnSpace[:, :, 1] - self.rxnSpace[:, :, 1].min()) * 255 /
                        (self.rxnSpace[:, :, 1].max() - self.rxnSpace[:, :, 1].min()), levels=50,
                        cmap = 'gray')
            fig.savefig(os.path.join("./gsTemp", f"frame_{i:06d}.png"), dpi=300)
            plt.close(fig)
            print("{:.2f} %".format(i / numSteps * 100), end = '\r')
        print("Simulation complete")

if __name__ == "__main__":

    diffRateU = 0.2
    diffRateV = 0.1
    feedRate = 0.042
    killRate = 0.062
    timeStep = 1

    sys = gsSystem("polka.png", diffRateU, diffRateV, feedRate, killRate, timeStep)
    sys.runSim(10000)

    sys.renderMovie()
