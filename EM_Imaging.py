# /!\ WARNING /!\ : tileSize 5 does not exist /!\

import win32com.client
from win32com.client import VARIANT
import pythoncom

import os, time, sys, shutil, pickle, Tkinter, tkFileDialog, subprocess
import logging, colorlog # colorlog is not yet standard
# import logging # if colorlog not available

from operator import itemgetter
from datetime import datetime

import numpy as np
from numpy import sin, pi, cos, arctan, tan, sqrt

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import skimage
from skimage import feature
from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.filters.rank import median
from skimage.morphology import disk

from Tkinter import *
import tkMessageBox

import winsound

# /!\ Warning : the merlin is flipped on the x axis. All stage variables in this script are in real coordinates. Only when I read from and write to the Merlin I flip the x axis.

#####################
### I/O Functions ###
def mkdir_p(path):
    try:
        os.mkdir(path)
        logger.debug('Folder created: ' + path)
    except Exception, e:
        if e[0] == 20047 or e[0] == 183:
            # IJ.log('Nothing done: folder already existing: ' + path)
            pass
        else:
            logger.error('Exception during folder creation :', exc_info=True)
            raise
    return path

def getDirectory(text, startingFolder = None):
    if startingFolder:
        direc = os.path.join(tkFileDialog.askdirectory(title = text, initialdir = startingFolder), '')
    else:
        direc = os.path.join(tkFileDialog.askdirectory(title = text), '')
    logger.debug('Directory chosen by user: ' + direc)
    return direc

def getPath(text, startingFolder = None):
    if startingFolder:
        path = os.path.join(tkFileDialog.askopenfilename(title = text, initialdir = startingFolder), '')
    else:
        path = os.path.join(tkFileDialog.askopenfilename(title = text), '')
    logger.debug('Path chosen by user: ' + path)
    return path

def getText(text):
    userText = raw_input(text)
    return userText
    
    
def readPoints(path):
    x,y = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
        for point in lines:
            x.append(int(point.split('\t')[0] ))
            y.append(int(point.split('\t')[1] ))
    return np.array([x,y])
    
def readSectionCoordinates(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        sections = []
        for line in lines:
#            print line
            points = line.replace('\n', '').split('\t')
            print points
            if points[-1] == '':
                points.pop()
            section = [ [int(float(point.split(',')[0])), int(float(point.split(',')[1]))] for point in points ]
            sections.append(section)
    return sections

def initLogger(path):
    fileFormatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(fileFormatter)
    fileHandler.setLevel(logging.DEBUG) # should I also save an .INFO log ? no: if someone wants to check a log, he probably wants to see the .debug one ...

    colorFormatter = colorlog.ColoredFormatter('%(log_color)s%(asctime)s %(levelname)s %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    streamHandler = colorlog.StreamHandler()
    streamHandler.setFormatter(colorFormatter)

    logger = logging.getLogger(__name__)
    # clean the logger in case the script is run again in the same console
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger
    
def logMerlinParameters():
    params = {}
    with open(logPath, 'a') as f:
        for parameter in allMerlinParameters:
            if parameter == 'AP_STAGE_AT_X': # /!\ Flipping x axis during read
                params[parameter] = - a.Get(parameter)[1]
            else:
                params[parameter] = a.Get(parameter)[1]
            f.write(parameter + ' = ' + str(params[parameter]) + '\n')
    return params

def durationToPrint(d):
    return str(round(d/60., 1)) + ' min = ' + str(round(d/3600., 1)) + ' hours = ' + str(round(d/(3600.*24), 1)) + ' days'
    
#####################
### GUI Functions ###
class App:
    global wafer
    def __init__(self, master):
        self.frame = Frame(master)
        self.frame.pack()
        
        self.button1 = Button(self.frame, text='Acquire wafer', command = self.acquireWaferButtonAction)
        self.button1.pack(side=LEFT)

        self.button11 = Button(self.frame, text='Acquire *sub*wafer', command = self.acquireSubWaferButtonAction)
        self.button11.pack(side=LEFT)
        
        self.button2 = Button(self.frame, text='Add mosaic here', command = addMosaicHere)
        self.button2.pack(side=LEFT)

        self.button3 = Button(self.frame, text='Add landmark', command = addLandmark)
        self.button3.pack(side=LEFT)

        self.button4 = Button(self.frame, text='Load wafer', command = loadWafer)
        self.button4.pack(side=LEFT)

        self.button7 = Button(self.frame, text='Save wafer', command = saveWafer)
        self.button7.pack(side=LEFT)

        self.button5 = Button(self.frame, text='Load sections and landmarks from pipeline', command = loadSectionsAndLandmarksFromPipeline)
        self.button5.pack(side=LEFT)

        self.button6 = Button(self.frame, text='Turn high tension off', command = turnHighTensionOff)
        self.button6.pack(side=LEFT)

        self.buttonQuit = Button(self.frame, text='Quit', command = root.destroy)
        self.buttonQuit.pack(side=LEFT)

    def acquireWaferButtonAction(self):
        # self.frame.quit() # no I should start an independent thread that scans the wafer and close this GUI
        turnOff = tkMessageBox.askquestion("Question", "Turn off high tension after acquisition ?")
        acquireWafer()
        if turnOff == 'yes':
            turnHighTensionOff()

    def acquireSubWaferButtonAction(self):
        sectionIndicesToAcquire = map(int, getText('What sections should be scanned (e.g., "1,3,5,7,9") ?').split(','))
        turnOff = tkMessageBox.askquestion("Question", "Turn off high tension after acquisition ?")
        acquireWafer(userDefinedSectionsToAcquire = sectionIndicesToAcquire)
        if turnOff == 'yes':
            turnHighTensionOff()

        
def addLandmark():
    stageXY = getStageXY()
    wafer.targetLandmarks.append([stageXY[0], stageXY[1], getWD()])
    nValidatedLandmarks = len(wafer.targetLandmarks)
    if nValidatedLandmarks == len(sourceLandmarks.T): # all target landmarks have been identified
        logger.info('Good. All landmarks have been calibrated.')
        
        targetSections = []
        targetAngles = []
        
        for sourceSectionTissueCoordinates in sourceSections:
            if hasattr(wafer, 'sourceROIDescription'): # transform the sourceRoi to the ROI in the sourceSection using the transform sourceSectionRoi -> sourceSectionCoordinates
                sourceSectionTissueCoordinates = affineT(np.array(wafer.sourceROIDescription[0]).T, np.array(sourceSectionTissueCoordinates).T, np.array(wafer.sourceROIDescription[1]).T).T

            targetTissueCoordinates = affineT(sourceLandmarks, np.array(wafer.targetLandmarks).T[:2], np.array(sourceSectionTissueCoordinates).T)
            
            # np.array([x_target_points, y_target_points])
            # x1,y1,x2,y2
        
            # the angle is simply the angle of the second line in the template. Should be enhanced ...
            targetAngle = getAngle([targetTissueCoordinates[0][2], targetTissueCoordinates[1][2], targetTissueCoordinates[0][3], targetTissueCoordinates[1][3]])
            
            targetAngle = ((- targetAngle)*180/float(pi) + 90)%360 # the second line is at the bottom and horizontal
            # targetAngle = ((- targetAngle)*180/float(pi) + 0)%360 # the second line is at the bottom and horizontal
            
            targetAngles.append(targetAngle)
            
            targetTissue = getCenter(targetTissueCoordinates)
            targetTissueCenterZ = focusThePoints(np.array(wafer.targetLandmarks).T, np.array([targetTissue]).T)[-1][0]
            targetSections.append([targetTissue[0], targetTissue[1], targetTissueCenterZ])
            
        for idSection, targetSection in enumerate(targetSections):
            section = Section(idSection, [targetSection[0], targetSection[1]], targetAngles[idSection], mp, sp, targetSection[2], wafer.folderWaferSave)
            wafer.sections.append(section)        
    elif nValidatedLandmarks > 1:
        logger.info('There are still ' + str(len(sourceLandmarks) - nValidatedLandmarks ) + ' landmarks to calibrate. The stage has been moved to the next landmark to be calibrated')
        moveStage(*affineT(sourceLandmarks, np.array(wafer.targetLandmarks).T[:2], sourceLandmarks).T[nValidatedLandmarks])
    else:
        logger.info('Please go manually to the second landmark.')
        
def addMosaicHere():
    wafer.addCurrentPosition()
    
def acquireWafer(userDefinedSectionsToAcquire = None):
    wafer.acquire(userDefinedSectionsToAcquire = userDefinedSectionsToAcquire)
    
def loadWafer():
    global wafer
    waferPath = getPath('Select the wafer pickle file', startingFolder = folderSave)
    f = open(os.path.normpath(waferPath), 'r')
    wafer = pickle.load(f)
    f.close()

def saveWafer():
    wafer.save()
    
def turnHighTensionOff():
    a.Execute('CMD_EHT_OFF')
    logger.info('High tension turned off')
    
def loadSectionsAndLandmarksFromPipeline():
    global sourceSections, sourceTissueMagDescription, sourceLandmarks
    pipelineFolder = getDirectory('Select the folder containing the sections and landmarks from the pipeline', startingFolder = folderSave)
    
    sourceSectionsPath = os.path.join(pipelineFolder, 'source_sections_tissue.txt.')
    sourceSections = readSectionCoordinates(sourceSectionsPath)
    
    sourceTissueMagDescriptionPath = os.path.join(pipelineFolder, 'source_tissue_mag_description.txt.')
    sourceTissueMagDescription = readSectionCoordinates(sourceTissueMagDescriptionPath)
    
    sourceLandmarksPath = os.path.join(pipelineFolder, 'source_landmarks.txt.')
    sourceLandmarks = readPoints(sourceLandmarksPath)
    
    sourceROIDescriptionPath = os.path.join(pipelineFolder, 'source_ROI_description.txt')
    if os.path.isfile(sourceROIDescriptionPath):
        wafer.sourceROIDescription = readSectionCoordinates(sourceROIDescriptionPath)
    else:
        logger.info('There is no source_ROI_description. The center of the tissue will be used as the center of the ROI.')

        
###########################
### Geometric functions    ###
def applyAffineT(points,coefs):
    x,y = np.array(points)
    x_out = coefs[1]*x - coefs[0]*y + coefs[2]
    y_out = coefs[1]*y + coefs[0]*x + coefs[3]
    return np.array([x_out,y_out])

def rotate(points, angle):
    angleRadian = angle * pi / 180.
    coefs = [sin(angleRadian), cos(angleRadian), 0, 0]
    return applyAffineT(points,coefs)

def translate(points, v):
    coefs = [0, 1, v[0], v[1]]
    return applyAffineT(points,coefs)
    
def affineT(sourceLandmarks, targetLandmarks, sourcePoints):
    # separating the x and y into separate variables
    x_sourceLandmarks, y_sourceLandmarks = np.array(sourceLandmarks).T[:len(targetLandmarks.T)].T # sourceLandmarks trimmed to the number of existing targetlandmarks
    x_targetLandmarks, y_targetLandmarks = targetLandmarks
    x_sourcePoints, y_sourcePoints = sourcePoints

    # Solving the affine transform
    A_data = []
    for i in range(len(x_sourceLandmarks)):
        A_data.append( [-y_sourceLandmarks[i], x_sourceLandmarks[i], 1, 0])
        A_data.append( [x_sourceLandmarks[i], y_sourceLandmarks[i], 0, 1])
    b_data = []
    for i in range(len(x_targetLandmarks)):
        b_data.append(x_targetLandmarks[i])
        b_data.append(y_targetLandmarks[i])
    A = np.matrix( A_data )
    b = np.matrix( b_data ).T
    c = np.linalg.lstsq(A, b)[0].T #solving happens here
    c = np.array(c)[0]
#   print('Absolute errors in target coordinates : (xError, yError)')
#   for i in range(len(x_sourceLandmarks)):
      #print ("%f, %f" % (
    #   np.abs(c[1]*x_sourceLandmarks[i] - c[0]*y_sourceLandmarks[i] + c[2] - x_targetLandmarks[i]),
    #   np.abs(c[1]*y_sourceLandmarks[i] + c[0]*x_sourceLandmarks[i] + c[3] - y_targetLandmarks[i])))

    #computing the accuracy
    x_target_computed_landmarks, y_target_computed_landmarks = applyAffineT(sourceLandmarks, c)
    accuracy = 0
    for i in range(len(x_targetLandmarks)):
            accuracy = accuracy +   np.sqrt( np.square( x_targetLandmarks[i] - x_target_computed_landmarks[i] ) + np.square( y_targetLandmarks[i] - y_target_computed_landmarks[i] ) )
    accuracy = accuracy/float(len(x_sourceLandmarks) + 1)
#   print 'The mean accuracy in target coordinates is', accuracy

    #computing the target points
    x_target_points, y_target_points = applyAffineT(sourcePoints,c)
    return np.array([x_target_points, y_target_points])

def getCenter(corners):
    center = np.array(map(np.mean, corners))
    return center

def getAngle(line):
    line = np.array(line)
    diff = line[0:2] - line[2:4]
    theta = np.arctan2(diff[1], diff[0])
    return theta

def getZ(x,y,abc): #Fitted plane function
    return float(abc[0]*x + abc[1]*y + abc[2])

def focusThePoints(focusedPoints, pointsToFocus):
    x_pointsToFocus, y_pointsToFocus = pointsToFocus[0], pointsToFocus[1] # works even if pointsToFocus has no z coordinates
    x_focusedPoints, y_focusedPoints, z_focusedPoints = focusedPoints
    
    # remove outliers
    idInliers = getInlierIndices(z_focusedPoints)
    logger.debug('There are ' + str(idInliers.size) + ' inliers in ' + str(map(lambda x:round(x, 2), z_focusedPoints*1e6)) + ' um' )
    if idInliers.size == 3:
        logger.warning('Warning - One autofocus point has been removed for interpolative plane calculation')
        x_focusedPoints, y_focusedPoints, z_focusedPoints = focusedPoints.T[idInliers].T
    elif idInliers.size < 3: 
        logger.warning('WARNING - There are only ' + str(idInliers.size) + ' inliers for the interpolative plane calculation. A strategy should be developed to address such an event.')
    
    A = np.column_stack([x_focusedPoints, y_focusedPoints, np.ones_like(x_focusedPoints)])
    abc,residuals,rank,s = np.linalg.lstsq(A, z_focusedPoints)
    z_pointsToFocus = map(lambda a: getZ (a[0],a[1],abc), np.array([x_pointsToFocus.transpose(), y_pointsToFocus.transpose()]).transpose())
    
    # calculating the accuracy
    z_check = np.array(map(lambda a: getZ (a[0],a[1],abc), np.array([x_focusedPoints.transpose(), y_focusedPoints.transpose()]).transpose()))
    diff = z_check - z_focusedPoints
    meanDiff = np.mean(np.sqrt(diff * diff))
    logger.debug('The plane difference is  ' + str(diff*1e6) + ' um')
    logger.info('The mean distance of focus points to the plane is ' + str(round(meanDiff*1e6, 3))  + ' um')
    
    return np.array([x_pointsToFocus, y_pointsToFocus, z_pointsToFocus])

def transformCoordinates(coordinates, center, angle):
    return (translate(rotate(coordinates.T, angle), center)).T

def pointsToXY(l): # probably useless, use simply a.T for numpy arrays
    return np.array([[p[0] for p in l],[p[1] for p in l]])

def XYtoPoints(XY): # probably useless, use simply a.T for numpy arrays
    l = []
    for x, y in zip(XY[0], XY[1]):
        l.append([x,y])
    return np.array(l)

def getInlierIndices(data, m = 8.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    print 'd', d
    print 's', s
    return np.where(s<m)[0]
    
    
############################
### Microscope functions ###
def moveStage(x, y):
    x = -x
    time.sleep(1)
    a.MoveStage(x, y, stageZ, 0, stageRotation, 0) #xxx check backlash
    time.sleep(sleepMoveStage)
    while a.Get('DP_STAGE_IS') == 'Busy':
        logger.info('Moving stage ...')
        time.sleep(sleepMoveStage)
    logger.debug('Stage has been moved to ' + str(round(x*1e6, 2)) + ', ' + str(round(y*1e6, 2)) + ' um')
    return
    
def    unfreeze():
    if a.Get('DP_FROZEN')[1] == 'Frozen':
        a.Execute('CMD_UNFREEZE_ALL')
        time.sleep(sleepUnfreeze)
        
def freezeNow():
    if a.Get('DP_FREEZE_ON')[1] != 'Command':
        a.Set('DP_FREEZE_ON', VARIANT(pythoncom.VT_R4, tableFreeze['Command'] ))
        # time.sleep(sleepFreeze)
    a.Execute('CMD_FREEZE_ALL')
    # time.sleep(sleepFreeze)

def    freezeAtEndOfFrame():
    if a.Get('DP_FREEZE_ON')[1] != 'End Frame':
        a.Set('DP_FREEZE_ON', VARIANT(pythoncom.VT_R4, tableFreeze['End Frame'] ))
        time.sleep(sleepFreezeEndOfFrame)
    a.Execute('CMD_FREEZE_ALL')
    time.sleep(sleepFreezeEndOfFrame)

def autofocus(sp, autofocusMode):
    if autofocusMode == 'rough':
        setMag(sp.roughFocusMag)
        setScanRate(sp.roughFocusScanRate)
        setTileSize(sp.roughFocusTileSizeIndex)
    elif autofocusMode == 'fine':
        setMag(sp.focusMag)
        setScanRate(sp.focusScanRate)
        setTileSize(sp.focusTileSizeIndex)
        
    unfreeze()
    a.Execute('CMD_AUTO_FOCUS_FINE')
    time.sleep(sleepFocus)
    logger.info('Autofocusing ...')
    while a.Get('DP_AUTO_FUNCTION')[1] == 'Focus':
        time.sleep(sleepFocus)

    WD = a.Get('AP_WD')[1]
    logger.info(autofocusMode + ' autofocus: WD = ' + str(round(WD * 1e6, 3)) + ' um')
    return WD

def autostig(sp, autofocusMode):
    if autofocusMode == 'rough':
        setMag(sp.roughFocusMag)
        setScanRate(sp.roughFocusScanRate)
        setTileSize(sp.roughFocusTileSizeIndex)
    elif autofocusMode == 'fine':
        setMag(sp.focusMag)
        setScanRate(sp.focusScanRate)
        setTileSize(sp.focusTileSizeIndex)
    
    unfreeze()
    
    a.Execute('CMD_AUTO_STIG')
    time.sleep(sleepFocus)
    logger.info('Autostigmating ...')
    while a.Get('DP_AUTO_FUNCTION')[1] == 'Stigmation':
        time.sleep(sleepFocus)

    stig = getStig()
    logger.info(autofocusMode + ' autostig: stigX = ' + str(round(stig[0], 3)) + ' % ; stigY = ' + str(round(stig[1], 3)) + ' %')
    return stig
    
def setMag(mag):
    a.Set('AP_MAG', VARIANT(pythoncom.VT_R4, mag))
    time.sleep(sleepSetMag)
    logger.debug('Magnification set to ' + str(round(mag)) + 'x')

def setScanRate(scanRate):
    scanRate = int(scanRate)
    a.Execute('CMD_SCANRATE' + str(scanRate))
    time.sleep(sleepSetScanRate)
    logger.debug('Scan rate has been set to ' + str(scanRate))
    
def setTileSize(ts):
    if ts == 5:
        logger.error('Attempt to set the tileSize to 5 but tileSize 5 does not exist. Setting tileSize 6 instead.')
        ts = 6
    a.Set('DP_IMAGE_STORE', VARIANT(pythoncom.VT_R4, int(ts) ))
    time.sleep(sleepSetTileSize)
    logger.debug('Tile size has been set to ' + str(ts) + ' : ' + str(a.Get('DP_IMAGE_STORE')))

def setScanRotation(angle):
    a.Set('AP_SCANROTATION', VARIANT(pythoncom.VT_R4, angle))
    time.sleep(sleepSetRotation)
    logger.debug('Scan rotation has been set to ' + str(round(angle, 2)) + ' degrees')

def setStig(stig):
    a.Set('AP_STIG_X', VARIANT(pythoncom.VT_R4, stig[0])) # check int or float ?
    a.Set('AP_STIG_Y', VARIANT(pythoncom.VT_R4, stig[1])) # check int or float ?
    time.sleep(sleepSetStig)
    logger.debug('Stig has been set to ' + str(map(lambda x:round(x,3), stig)) + ' %')

def setWD(wd):
    a.Set('AP_WD', VARIANT(pythoncom.VT_R4, wd))
    time.sleep(sleepSetWD)
    logger.debug('WD has been set to ' + str(round(wd * 1e6, 3)) + ' um')
    
def getPixelSize():
    res = a.Get('DP_IMAGE_STORE')[1]
    width = int(res[0: res.index('*')])
    return magCalib/(a.Get('AP_MAG')[1] * width)

def getScanRotation():
    return float(a.Get('AP_SCANROTATION')[1])
    
def getStageXY():
    x,y = a.GetStagePosition()[1:3]
    x = -x # flipping to real world coordinates
    return np.array([x, y])

def getWD():
    return     float(a.Get('AP_WD')[1])
    
def getStig():
    return np.array([float(a.Get('AP_STIG_X')[1]), float(a.Get('AP_STIG_Y')[1])])
    
def acquireInSitu(tileSize, sectionIndex, tileIndex, folder):
    freezeNow()
    time.sleep(sleepAcquireStart) # time.sleep(2) is not enough
    setTileSize(mp.tileSizeIndex)
    time.sleep(3)
    freezeAtEndOfFrame()
    logger.info('Scanning ...')
    while a.Get('DP_FROZEN')[1] == 'Live':
        time.sleep(sleepScanning)
        
    logger.info('Tile ' + str(sectionIndex) + '-' + str(tileIndex) + ' acquired. Now grabbing ...')
    tilePath = os.path.join(folder, 'Tile_' + str(tileIndex[0]) + '-' +  str(tileIndex[1]) + '.tif')
    a.Grab(0, 0, 1024, 768, 0, tilePath)
    freezeNow() # I think that Grab triggers an unfreeze ?

    logger.debug('Tile grabbed and saved in ' + tilePath)

def contrastLocation(imPath):
    # open the low resolution image
    im = skimage.io.imread(imPath)[0]
    imSize = np.array([len(im[0]), int(len(im[0]))*3/4.]) # true x,y
    
    # imEdges = skimage.img_as_ubyte(prewitt(im))
    imEdges = median(skimage.img_as_ubyte(feature.canny(im, sigma = contrastSigma)), disk(1))
    skimage.io.imsave(os.path.join(folderSave, os.path.splitext(os.path.basename(imPath))[0] + '_Edges.tif'), imEdges) # for testing

    # define the number of patches for the two resolution levels
    lowGrid = np.floor(imSize/findContrastPatchLow).astype(int)
    
    # compute the sum of the edge intensities in the subblocks of the two resolution levels
    twoLevelsContrast = []
    for x in range(lowGrid[0]):
        for y in range(lowGrid[1]):
            center = ((np.array([x,y]) + 1/2.) * findContrastPatchLow).astype(int)
            
            lowResContrast = np.sum(imEdges[np.ix_(range(findContrastPatchLow[1]*y, findContrastPatchLow[1]*(y+1)), range(findContrastPatchLow[0]*x, findContrastPatchLow[0]*(x+1), 1)) ])  /  float(findContrastPatchLow[0] * findContrastPatchLow[1])
            
            highResContrast = np.sum(imEdges[np.ix_(range(int(center[1] - findContrastPatchHigh[1]/2.), int(center[1] + findContrastPatchHigh[1]/2.)), range(int(center[0] - findContrastPatchHigh[0]/2.), int(center[0] + findContrastPatchHigh[0]/2.)) ) ])

            twoLevelsContrast.append([[x,y], center, lowResContrast, highResContrast])

    # find the maximum contrast regions at low resolution
    bestLowResPatches = sorted(twoLevelsContrast, key = itemgetter(2), reverse = True)[:3] # take the 3 best low res blocks
    bestHighResPatch = sorted(bestLowResPatches, key = itemgetter(3), reverse = True)[0] # take one of the 3 best low blocks that has the best center
    
    # saving the best patch selected
    center = bestHighResPatch[1]
    bestPatchImage = im[np.ix_(range(int(center[1] - findContrastPatchLow[1]/2.), int(center[1] + findContrastPatchLow[1]/2.)), range(int(center[0] - findContrastPatchLow[0]/2.), int(center[0] + findContrastPatchLow[0]/2.)) ) ]
    skimage.io.imsave(os.path.join(folderSave, os.path.splitext(os.path.basename(imPath))[0] + '_bestPatch.tif'), bestPatchImage) # for testing
    logger.debug('I should have saved the contrast location image in ' +  os.path.join(folderSave, os.path.splitext(os.path.basename(imPath))[0] + '_bestPatch.tif'))
    
    displacementContrast = bestHighResPatch[1] - imSize.astype(float)/2
    
    # displacementContrast = np.array([displacementContrast[0], -displacementContrast[1]]) # 20/11/2017: noticed a y-flip ! Really ? Is it not a problem with the angle instead ?
 
    return displacementContrast
    
def moveToContrast(sectionId, pointId):
    setScanRate(sp.contrastScanRate)
    setMag(sp.contrastMag)
    setTileSize(0) # 1024*768
    freezeAtEndOfFrame()
    logger.debug('Scanning briefly to find a good contrast location')
    while a.Get('DP_FROZEN')[1] == 'Live':
        time.sleep(sleepScanning)
        
    tilePath = os.path.join(folderSave, 'AssessContrast_' + str(sectionId) + '-' +  str(pointId) + '.tif')
    a.Grab(0, 0, 1024, 768, 0, tilePath)
    freezeNow() # I think that Grab triggers an unfreeze ?
    
    displacementContrast = contrastLocation(tilePath)
    contrastXY = transformCoordinates((displacementContrast * getPixelSize()), getStageXY(), -getScanRotation()) # changed the rotation to MINUS rotation
    moveStage(contrastXY[0], contrastXY[1])

def isNewStigOk(newStig, oldStig):
    return ((np.abs(newStig[0]-oldStig[0])/float(oldStig[0]) < thresholdStig) and (np.abs(newStig[1]-oldStig[1])/float(oldStig[1]) < thresholdStig))
    
    
####################################
### Scanning and Tile parameters ###
class ScanningParameters(object):
    def __init__(self, *args):
        self.scanRate = args[0]
        self.dwellTime = dwellTimes[args[0]]
        self.brightness = args[1]
        self.contrast = args[2]
        self.startingStig = np.array(args[3])

        self.roughFocusScanRate = 5
        self.roughFocusMag = 20000
        self.roughFocusTileSizeIndex = 0
        
        self.focusScanRate = args[4]
        self.focusMag = 70000
        self.focusTileSizeIndex = 0
        
        self.contrastScanRate = 5
#        self.contrastMag = 30000
        self.contrastMag = 15000 # because of big blood vessel in B6

        logger.debug('Scanning parameters initialized')
        
class MosaicParameters(object):
    def __init__(self, *args):
        self.tileSizeIndex = args[0]
        self.tileSize_px = availableTileSizes_px[args[0]]
        self.tileGrid = np.array(args[1])
        self.overlap_pct = args[2]
        self.pixelSize = args[3]
        self.mag = magCalib/(self.pixelSize * self.tileSize_px[0])
        self.autofocusOffsetFactor = args[4] # represents 1/4 of the diagonal when equal to 1/2.
        self.mosaicSize = 0
        self.ids = [] # indices of the successive tiles [[0,0],[1,0],...,[n,n]]
        self.layoutFigurePath = os.path.join(folderSave, 'mosaicLayout.png')
        self.templateTileCoordinates = self.getTemplateTileCoordinates()
        logger.debug('Mosaic parameters initialized')
        
    def getTemplateTileCoordinates(self):
        fig = plt.figure() # producing a figure of the mosaic and autofocus locations
        ax = fig.add_subplot(111)    
    
        tileSize = self.pixelSize * self.tileSize_px
        
        # compute and plot mosaic size
        mosaic_px = np.round(self.tileSize_px * self.tileGrid - (self.tileGrid - 1) * (self.overlap_pct/100. * self.tileSize_px))
        self.mosaicSize = self.pixelSize * mosaic_px
        logger.debug('The size of the mosaic is ' + str(self.mosaicSize[0] * 1e6) + ' um x ' + str(self.mosaicSize[1] * 1e6) + ' um')
        p = patches.Rectangle((-self.mosaicSize[0]/2., -self.mosaicSize[1]/2.), self.mosaicSize[0], self.mosaicSize[1], fill=False, clip_on=False, color = 'blue', linewidth = 3)
        ax.add_patch(p)

        # compute tile locations starting from the first on the top left (which is actually top right in the Merlin ...)
        topLeftCenter_px = (- mosaic_px + self.tileSize_px)/2.
        topLeftCenter = self.pixelSize * topLeftCenter_px

        tilesCoordinates = []
        for idY in range(self.tileGrid[1]):
            for idX in range(self.tileGrid[0]):
                id = np.array([idX, idY])
                self.ids.append(id)
                tileCoordinates = (topLeftCenter + id * (1-self.overlap_pct/100.) * tileSize)
                tilesCoordinates.append(tileCoordinates)
                plt.plot(tileCoordinates[0], tileCoordinates[1], 'ro')
                p = patches.Rectangle((tileCoordinates[0] - tileSize[0]/2. , tileCoordinates[1] - tileSize[1]/2.), tileSize[0], tileSize[1], fill=False, clip_on=False, color = 'red')
                ax.add_patch(p)

        tilesCoordinates = np.array(tilesCoordinates)
        tilesCoordinates_px = np.round(tilesCoordinates/self.pixelSize)

        # compute autofocus locations
        if (self.tileGrid == np.array([1,1])).all():
            autofocusCoordinates = np.array([[0,0]])
        else:
            autofocusCoordinates = self.mosaicSize/2. * (1 - self.autofocusOffsetFactor) * np.array([ [-1 , -1], [1, -1], [-1, 1], [1, 1]]) # 4 points focus
    
        # plot autofocus locations
        for point in autofocusCoordinates:
            plt.plot(point[0], point[1], 'bo')
        
        plt.savefig(self.layoutFigurePath)

        return tilesCoordinates, autofocusCoordinates

#################################
### Wafer and Section classes ###
class Wafer(object):
    def __init__(self, *args):
        self.name = args[0]
        self.mp = args[1] # MosaicParameters
        self.sp = args[2] # scanningParameters
        self.sections = []
        self.folderWaferSave = mkdir_p(os.path.join(folderSave, self.name))
        self.waferPath = os.path.join(self.folderWaferSave, 'Wafer_' + self.name)
        shutil.copy(mp.layoutFigurePath, self.folderWaferSave)
        logger.info('Wafer ' + self.name + ' initiated.')
        self.startingTime = -1
        self.finishingTime = -1
        self.targetLandmarks = []
        self.timeEstimate = 0
        self.params = logMerlinParameters()
        
    def addCurrentPosition(self):
        params = logMerlinParameters() # maybe a bit too much I should just take the parameters I need
        currentStageX = params['AP_STAGE_AT_X']
        currentStageY = params['AP_STAGE_AT_Y']
        currentStageT = params['AP_STAGE_AT_T']
        currentStageR = params['AP_STAGE_AT_R']
        currentStageM = params['AP_STAGE_AT_M']
        currentScanRotation = params['AP_SCANROTATION']
        currentWD = params['AP_WD']
        currentStigX = params['AP_STIG_X']
        currentStigY = params['AP_STIG_Y']
        currentBrightness = params['AP_BRIGHTNESS']
        currentContrast= params['AP_CONTRAST']
        section = Section(len(self.sections), [currentStageX, currentStageY], currentScanRotation, self.mp, self.sp, currentWD, self.folderWaferSave)
        self.sections.append(section)
        logger.info('Section initialized with current position')
        self.timeEstimate = len(self.sections) * (mosaicAutofocusRoughDuration + self.mp.tileGrid[0] * self.mp.tileGrid[1] * (self.mp.tileSize_px[0] * self.mp.tileSize_px[1] * self.sp.dwellTime + tileRoughOverhead))
        logger.info('There are ' + str(len(self.sections)) + ' sections and it will take approximately ' + durationToPrint(self.timeEstimate) )

    def save(self):
        f = open(self.waferPath, 'w')
        pickle.dump(self, f)
        f.close()
        logger.debug('Wafer pickled in ' + self.waferPath)
        
        with open(self.waferPath + '_EM_Metadata.txt', 'w') as f:
            f.write('name = ' + str(self.name) + '\n')
            f.write('nSections = ' + str(len(self.sections)) + '\n')
            f.write('scanRate = ' + str(self.sp.scanRate) + '\n')
            f.write('dwellTime = ' + str(self.sp.dwellTime) + '\n')
            f.write('brightness = ' + str(self.sp.brightness) + '\n')
            f.write('contrast = ' + str(self.sp.contrast) + '\n')
            f.write('tileWidth = ' + str(self.mp.tileSize_px[0]) + '\n')
            f.write('tileHeight = ' + str(self.mp.tileSize_px[1]) + '\n')
            f.write('numTilesX = ' + str(self.mp.tileGrid[0]) + '\n')
            f.write('numTilesY = ' + str(self.mp.tileGrid[1]) + '\n')
            # f.write('tileOverlapX = ' + str(self.mp.overlap_pct[0]/100. * self.mp.tileSize_px[0]) + '\n')
            # f.write('tileOverlapY = ' + str(self.mp.overlap_pct[1]/100. * self.mp.tileSize_px[1]) + '\n')
            f.write('tileOverlapX = ' + str(self.mp.overlap_pct[0]/100.) + '\n')
            f.write('tileOverlapY = ' + str(self.mp.overlap_pct[1]/100.) + '\n')
            f.write('pixelSize = ' + str(self.mp.pixelSize) + '\n')
            f.write('xPatchEffectiveSize = ' + str(self.mp.tileSize_px[0] * (1 - float(self.mp.overlap_pct[0]/100.) )) + '\n')
            f.write('yPatchEffectiveSize = ' + str(self.mp.tileSize_px[1] *(1 - float(self.mp.overlap_pct[1]/100.))) + '\n')
            f.write('magnification = ' + str(self.mp.mag) + '\n')
            f.write('autofocusOffsetFactor = ' + str(self.mp.autofocusOffsetFactor) + '\n')
            f.write('mosaicSize_x = ' + str(self.mp.mosaicSize[0]) + '\n')
            f.write('mosaicSize_y = ' + str(self.mp.mosaicSize[1]) + '\n')

    def acquire(self, userDefinedSectionsToAcquire = None):
        self.save()
        logger.info('Starting acquisition of wafer ' + str(self.name))
        self.startingTime = time.time()
        logger.info(str(len(filter(lambda x: x.acquireFinished, self.sections))) + ' sections have been already scanned before this start')
        
        nSectionsAcquired = sum([section.acquireFinished for section in self.sections] ) # for after interruptions
        sectionIndicesToAcquire = range(nSectionsAcquired, len(self.sections), 1)
        
        if userDefinedSectionsToAcquire == None:
            sectionsToAcquire = filter(lambda x: (not x.acquireFinished), wafer.sections)
        else:
            sectionsToAcquire = filter(lambda x: x.index in userDefinedSectionsToAcquire, wafer.sections)
            # sectionsToAcquire = [wafer.sections[i] for i in userDefinedSectionsToAcquire] # this should work too (except in some reordering cases ?)
        
        #xxx do I need a mechanism to average the past WD and stig values ?
        for id, sectionToAcquire in enumerate(sectionsToAcquire):
            logger.info('Starting acquisition of section index ' + str(sectionToAcquire.index) + ' (number ' +  str(id) + ' of the current session) in wafer ' + str(self.name))
            sectionToAcquire.acquire()
            #logging some durations
            averageSectionDuration = (time.time()- self.startingTime)/float(id + 1)
            timeRemaining = len(filter(lambda x: (not x.acquireFinished), wafer.sections)) * averageSectionDuration
            logger.info(str(id + 1) + ' sections have been scanned during this session, with an average of ' + str(round(averageSectionDuration/60., 1)) + ' min/section.' )
            logger.info('Time remaining estimated: ' + durationToPrint(timeRemaining) + ' for ' + str(len(filter(lambda x: (not x.acquireFinished), wafer.sections))) + ' sections remaining')
            self.save()
        self.finishingTime = time.time()
        elapsedTime = (self.finishingTime - self.startingTime)
        logger.info('The current session for the wafer took ' + durationToPrint(elapsedTime))
        winsound.Beep(440,1000)
        winsound.Beep(880,500)
        winsound.Beep(440,1000)
        winsound.Beep(880,500)
        winsound.Beep(440,1000)
        winsound.Beep(880,500)
        winsound.Beep(440,1000)
        
        if elapsedTime>3600: # turn off high tension if scan took more than 1 hour
            logger.critical('Turning the beam off because the scan took more than 1 hour')
            a.Execute('CMD_EHT_OFF')

class Section(object):
    def __init__(self, *args):
        self.index = args[0]
        self.center = args[1]
        self.angle = args[2]
        self.mp = args[3] # MosaicParameters
        self.sp = args[4] # scanningParameters
        self.startingWD = args[5] # given by the interpolative plane
        self.folderWaferSave = args[6]
        self.imagingCoordinates = {}
        self.imagingCoordinates['tiles'] = transformCoordinates(self.mp.templateTileCoordinates[0], self.center, -self.angle)
        self.imagingCoordinates['autofocus'] = transformCoordinates(self.mp.templateTileCoordinates[1], self.center, -self.angle)
        
        self.params = None
        self.focusedPoints = []
        self.stigs = []
        self.acquireStarted = False
        self.acquireFinished = False
        self.currentWD = self.startingWD
        self.startingStig = None
        self.startingTile = 0 # for after interruptions
        self.folderSectionSave = os.path.join(self.folderWaferSave, 'section_' + str(self.index).zfill(4))
        self.startingTime = -1
        self.finishingTime = -1
        
    def acquire(self):
        self.startingTime = time.time()
        self.params = logMerlinParameters()
        if (not self.acquireStarted): # to handle the case when the section is reset manually by setting acquireStarted=False
            self.startingTile = 0
        self.acquireStarted = True
        mkdir_p(self.folderSectionSave)
        self.moveToSection()
        self.computeWDPlaneAndGetStig()
        self.scanTiles()
        self.finishingTime = time.time()
        self.acquireFinished = True
        logger.debug('Section ' + str(self.index) + ' acquired. It has taken ' + str((self.finishingTime - self.startingTime)/60.) + ' min.' )
        
    def getRoughFocusStig(self): # should I assume that it will never fail ?
        self.currentWD = autofocus(self.sp, 'rough')
        self.currentStig = autostig(self.sp, 'rough')

    def getRoughFocus(self): # should I assume that it will never fail ?
        self.currentWD = autofocus(self.sp, 'rough')    

    def computeWDPlaneAndGetStig(self):
        if len(self.imagingCoordinates['autofocus']) == 1:

            moveStage(self.imagingCoordinates['autofocus'][0][0], self.imagingCoordinates['autofocus'][0][1]) # probably useless
            
            time.sleep(sleepBeforeContrast)
            moveToContrast(0, 0)
            
            autofocus(self.sp, 'fine')
            autostig(self.sp, 'fine')
            
            
            self.imagingCoordinates['tiles'] = np.array([[self.imagingCoordinates['tiles'][0][0], self.imagingCoordinates['tiles'][0][1], autofocus(self.sp, 'fine')]])
            logger.debug('The imaging coordinates will be ' + str(self.imagingCoordinates['tiles']))
        
        elif ((self.mp.tileGrid == np.array([2,2])).all()) or ((self.mp.tileGrid == np.array([3,3])).all()):

            # already at the center of the section
        
            logger.debug('Special focusing for [2,2] grid')
            time.sleep(sleepBeforeContrast)
            moveToContrast(0, 0)
            
            autofocus(self.sp, 'fine')
            autostig(self.sp, 'fine')
            WD = autofocus(self.sp, 'fine')
            
            
            allImagingCoordinates = []
            for id, imagingCoordinates in enumerate(self.imagingCoordinates['tiles']): # set the same working distance for all tiles
                # self.imagingCoordinates['tiles'][id] = np.array([imagingCoordinates[0], imagingCoordinates[1], WD])
                allImagingCoordinates.append([imagingCoordinates[0], imagingCoordinates[1], WD])
            self.imagingCoordinates['tiles'] = np.array(allImagingCoordinates)
            logger.debug('The imaging coordinates will be ' + str(self.imagingCoordinates['tiles']))

        
        else:
            self.focusedPoints = [] # this is needed for after interruptions: the focused points should be cleared
            self.startingStig = getStig()
            for idPoint, autofocusPosition in enumerate(self.imagingCoordinates['autofocus']):
                logger.debug('Autofocusing/stig of point number ' + str(idPoint) + ' in Tile number ' + str(self.index))
                moveStage(autofocusPosition[0], autofocusPosition[1])
                
                setWD(self.startingWD) # the case happened that the focus failed in the first corner, and the wrong focus propagated. Going back each time to startingWD is a first approximation. Ideally it would go back to the average of the previous section(s)

                if (idPoint == 0):
                    self.getRoughFocus()

                time.sleep(sleepBeforeContrast)
                moveToContrast(self.index, idPoint)
                

                if (idPoint == 0): # foc stig foc for first corner of the autofocuses
                    WD = autofocus(self.sp, 'fine')
                    stig = autostig(self.sp, 'fine')
                    if (not isNewStigOk(stig, self.startingStig)):
                        setStig(self.startingStig) # set to the stig of the previous section
                        self.stigs.append(self.startingStig)
                        logger.warning('Warning in section ' + str(self.index) + ': Rejection of autostig in first corner. Setting stigmation to stig of previous section')
                    else:
                        self.stigs.append(stig)

                if (idPoint == 3): # foc stig foc for fourth corner of the autofocuses
                    WD = autofocus(self.sp, 'fine')
                    stig = autostig(self.sp, 'fine')
                    if (not isNewStigOk(stig, self.stigs[0])):
                        self.stigs.append(self.stigs[0])
                        setStig(self.stigs[0]) # set to the stig of the previous section
                        logger.warning('Warning in section ' + str(self.index) + ': Rejection of autostig in fourth corner. Setting stigmation to stig of first corner')
                    else:
                        self.stigs.append(stig)
                
                WD = autofocus(self.sp, 'fine')
                self.focusedPoints.append([autofocusPosition[0], autofocusPosition[1], WD])
        
            self.imagingCoordinates['tiles'] = focusThePoints(np.array(self.focusedPoints).T, self.imagingCoordinates['tiles'].T).T
            
            logger.debug('The imaging coordinates will be ' + str(self.imagingCoordinates['tiles']))
            logger.info('Interpolative plane calculated for Section number ' + str(self.index))

    def scanTiles(self):
        setScanRate(self.sp.scanRate)
        setMag(self.mp.mag)
        
        tilesToScan = range(self.startingTile, len(self.imagingCoordinates['tiles']), 1) # for restart after interruption
        
        for idTile in tilesToScan: 
            tileCoordinates = self.imagingCoordinates['tiles'][idTile]
            logger.info('Scanning Tile ' + str(idTile) + ' of section ' + str(self.index))
            moveStage(tileCoordinates[0], tileCoordinates[1])
            if len(tileCoordinates) == 3: #tileCoordinates might lack the WD coordinate when I am testing short scans
                setWD(tileCoordinates[2])
            acquireInSitu(self.mp.tileSizeIndex, self.index, self.mp.ids[idTile], self.folderSectionSave)
            self.startingTile = idTile + 1
            # xxx should I autostig from time to time ?
        
    def moveToSection(self):
        a.Set('DP_X_BACKLASH', VARIANT(pythoncom.VT_R4, 3))
        a.Set('DP_Y_BACKLASH', VARIANT(pythoncom.VT_R4, 3))
        moveStage(self.center[0], self.center[1])
        a.Set('DP_X_BACKLASH', VARIANT(pythoncom.VT_R4, 0))
        a.Set('DP_Y_BACKLASH', VARIANT(pythoncom.VT_R4, 0))
        setScanRotation(self.angle)
        logger.debug('Moved to center of section ' + str(self.index))
        

#################
### Constants ###
if __name__ == '__main__':

    # Initializations
    tableFreeze = {}
    tableFreeze['End Frame'] = 0
    tableFreeze['End Line'] = 1
    tableFreeze['Command'] = 2

    pixelsCalib = 32768 * 24576
    dwellTimes = np.array([53.2/60., 1.6, 2.9, 5.6, 11, 21.7, 43.2, 1.4*60, 2.9*60, 6, 11, 1.9*24*60, 3.8*60*24, 7.6*24*60, 15*60*24]) * 60. / float(pixelsCalib)
    # /!\ Recheck the availableTilesize, whether it makes sense with the fact that tilSize number 5 does not exist
    availableTileSizes_px = np.array([[1024, 768], [512, 384], [2048, 1536], [3072, 2304], [4096, 3072], [6144, 4608], [6144, 4608], [8192, 6144], [12288, 9216], [16384, 12288], [24576, 18432], [32768, 24576]])

    magCalib = 0.787197714089416

    sleepFocus = 0.3
    sleepUnfreeze = 0.5
    sleepScanning = 0.5
    sleepMoveStage = 3
    sleepSetMag = 0.3
    sleepSetScanRate = 0.2
    sleepSetTileSize = 0.5
    sleepSetRotation = 0.5
    sleepSetStig = 0.3
    sleepSetWD = 0.2
    sleepFreeze = 0.2
    sleepFreezeEndOfFrame = 0.2
    sleepAcquireStart = 4 # 2 is not working (maybe because of the backlash ?)
    sleepBeforeContrast = 1

    mosaicAutofocusRoughDuration = 120
    tileRoughOverhead = 5 + 5 # stage move + writing to disk

    gunParameters = ['AP_GUNALIGN_X', 'AP_GUNALIGN_Y', 'AP_EXTCURRENT', 'AP_MANUALEXT', 'AP_MANUALKV', 'AP_ACTUALKV', 'AP_ACTUALCURRENT', 'AP_FILAMENT_AGE', 'DP_FIL_BLOWN', 'DP_RUNUPSTATE', 'DP_HIGH_CURRENT']
    beamParameters = ['AP_BEAMSHIFT_X', 'AP_BEAMSHIFT_Y', 'AP_BEAM_OFFSET_X', 'AP_BEAM_OFFSET_Y', 'AP_BRIGHTNESS', 'AP_CONTRAST', 'AP_MAG', 'AP_WD', 'AP_SPOT', 'AP_PIXEL_SIZE', 'AP_SCM', 'AP_SPOTSIZE', 'AP_IPROBE', 'AP_STIG_X', 'AP_STIG_Y', 'AP_AUTO_BRIGHT', 'AP_AUTO_CONTRAST', 'AP_ZOOM_FACTOR', 'AP_TILT_ANGLE', 'DP_BEAM_BLANKED', 'DP_BEAM_BLANKING', 'DP_AUTO_FUNCTION', 'DP_SCM_RANGE', 'DP_SCM', 'DP_AUTO_VIDEO', ]
    scanParameters = ['AP_SPOT_POSN_X', 'AP_SPOT_POSN_Y', 'AP_LINE_POSN_X', 'AP_LINE_POSN_Y', 'AP_LINE_LENGTH', 'AP_SCANROTATION', 'AP_PIXEL_SIZE', 'AP_LINE_TIME', 'AP_FRAME_TIME', 'AP_FRAME_AVERAGE_COUNT', 'AP_FRAME_INT_COUNT', 'AP_LINE_INT_COUNT', 'AP_RED_RASTER_POSN_X', 'AP_RED_RASTER_POSN_Y', 'AP_RED_RASTER_W', 'AP_RED_RASTER_H', 'AP_LINE_AVERAGE_COUNT', 'AP_NR_COEFF', 'AP_WIDTH', 'AP_HEIGHT', 'DP_SCAN_ROT', 'DP_FREEZE_ON', 'DP_LINE_SCAN', 'DP_EXT_SCAN_CONTROL', 'DP_MAX_RATE', 'DP_SCANRATE', 'DP_NOISE_REDUCTION', 'DP_IMAGE_STORE', 'DP_FROZEN', 'DP_LEFT_FROZEN', 'DP_RIGHT_FROZEN', 'DP_DISPLAY_CHANNELS', 'DP_AUTO_FUNCTION']
    apertureParameters = ['AP_APERTURESIZE', 'AP_APERTURE_ALIGN_X', 'AP_APERTURE_ALIGN_Y', 'AP_APERTUREPOSN_X' , 'AP_APERTUREPOSN_Y', 'DP_APERTURE', 'DP_APERTURE_STATE', 'DP_APERTURE_TYPE']
    detectorParameters = ['AP_PHOTO_NUMBER', 'AP_COLLECTOR_BIAS', 'DP_OUT_DEV', 'DP_4QBSD_Q1', 'DP_4QBSD_Q2', 'DP_4QBSD_Q3', 'DP_4QBSD_Q4', 'DP_4QBSD_VISIBLE', 'DP_4QBSD', 'DP_ZONE', 'DP_DETECTOR_CHANNEL', 'DP_DETECTOR_TYPE', 'DP_HRRU_SPEED', 'DP_HRRU_PHOTO_STATUS', 'DP_HRRU_SOURCE']
    stageParameters = ['AP_STAGE_AT_X', 'AP_STAGE_AT_Y', 'AP_STAGE_AT_Z', 'AP_STAGE_AT_T', 'AP_STAGE_AT_R', 'AP_STAGE_AT_M', 'AP_STAGE_GOTO_X', 'AP_STAGE_GOTO_Y', 'AP_STAGE_GOTO_Z', 'AP_STAGE_GOTO_T', 'AP_STAGE_GOTO_R', 'AP_STAGE_GOTO_M', 'AP_STAGE_HIGH_X', 'AP_STAGE_HIGH_Y', 'AP_STAGE_HIGH_Z', 'AP_STAGE_HIGH_T', 'AP_STAGE_HIGH_R', 'AP_STAGE_HIGH_M', 'AP_STAGE_LOW_X', 'AP_STAGE_LOW_Y', 'AP_STAGE_LOW_Z', 'AP_STAGE_LOW_T', 'AP_STAGE_LOW_R', 'AP_STAGE_LOW_M', 'AP_PIEZO_AT_X', 'AP_PIEZO_GOTO_X', 'AP_PIEZO_GOTO_Y', 'DP_STAGE_TYPE', 'DP_STAGE_BACKLASH', 'DP_STAGE_INIT', 'DP_STAGE_IS', 'DP_STAGE_TOUCH', 'DP_X_BACKLASH', 'DP_Y_BACKLASH', 'DP_Z_BACKLASH', 'DP_T_BACKLASH', 'DP_R_BACKLASH', 'DP_M_BACKLASH', 'DP_X_LIMIT_HIT', 'DP_Y_LIMIT_HIT', 'DP_Z_LIMIT_HIT', 'DP_T_LIMIT_HIT', 'DP_R_LIMIT_HIT', 'DP_X_AXIS_IS', 'DP_Y_AXIS_IS', 'DP_Z_AXIS_IS', 'DP_T_AXIS_IS', 'DP_R_AXIS_IS', 'DP_M_AXIS_IS', 'DP_X_AXIS', 'DP_Y_AXIS', 'DP_Z_AXIS', 'DP_T_AXIS', 'DP_R_AXIS', 'DP_M_AXIS', 'DP_X_ENABLED', 'DP_Y_ENABLED', 'DP_Z_ENABLED', 'DP_T_ENABLED', 'DP_R_ENABLED', 'DP_M_ENABLED', 'DP_STAGE_TILTED', 'DP_JOYSTICK_DISABLE', 'DP_STAGE_SCAN', 'DP_STAGE_SCANNING']
    vacuumParameters = ['AP_HP_TARGET', 'AP_SYSTEM_VAC', 'AP_COLUMN_VAC', 'AP_CHAMBER_PRESSURE', 'DP_COLUMN_CHAMBER_VALVE', 'DP_COLUMN_PUMPING', 'DP_COLUMN_PUMP', 'DP_HP_STATUS', 'DP_VACSTATUS', 'DP_VAC_MODE', 'DP_EP_OK', 'DP_AIRLOCK', 'DP_AIRLOCK_CONTROL', 'DP_AIRLOCK_READY', 'DP_EHT_VAC_READY', 'DP_BAKEOUT', 'DP_BAKEOUT_STATUS']
    allMerlinParameters = gunParameters + beamParameters + scanParameters + apertureParameters + detectorParameters + stageParameters + vacuumParameters

    thresholdStig = 0.1

    findContrastPatchLow = np.array([200, 200]) #/!\ works only for square patches
    findContrastPatchHigh = np.array([120,120])
    
    ######################
    ### I/O Parameters ###
    # folder = getDirectory('Please give me the folder containing all landmark and scanning files')
    folderSave = os.path.join(r'D:\Atlas_Images\0926\Thesis\ATEST', '')
    waferName = 'AWAFERTEST'

    logPath = os.path.join(folderSave, 'log_' + waferName + '.txt')
    logger = initLogger(logPath)

    ##########################################################################
    ### Initialization of communication and parameters with the microscope ###
    a = win32com.client.Dispatch('CZ.EMApiCtrl.1')
    a.InitialiseRemoting()
    a.Set('DP_SCAN_ROT', VARIANT(pythoncom.VT_R4, 1)) # activate scan rotation
    a.Set('DP_FREEZE_ON', VARIANT(pythoncom.VT_R4, 0))
    a.Set('DP_Z_BACKLASH', VARIANT(pythoncom.VT_R4, 0))
    a.Set('DP_T_BACKLASH', VARIANT(pythoncom.VT_R4, 0))

    stageM = float(a.Get('AP_STAGE_AT_M')[1])
    stageRotation = float(a.Get('AP_STAGE_AT_R')[1])
    stageTilt = float(a.Get('AP_STAGE_AT_T')[1])
    stageZ = float(a.Get('AP_STAGE_AT_Z')[1])
    if stageM != 0:
        print 'Warning, the "master-z of the stage" is not equal to 0 and the scripts assume that it is equal to 0. Exiting.'
        sys.exit()
    if stageTilt != 0:
        print 'Warning, the tilt angle is not equal to 0: ' + str(stageTilt) + '. Exiting.'
        sys.exit()

    params = logMerlinParameters()

    # Initialization of GUI
    root = Tk()

    focusScanRate = 3
    ###########################
    ### Scanning Parameters ###
    brightness = 25 # should B/C be recorded after the very first tile and kept until the end ?
    contrast = 40
    startingStig = getStig()

    # tileSizeIndex = 7 = 8192*6144
    
    # INLENS
    scanRate = 4 # for calibration inlens
    tileSizeIndex = 8 # for inlens
    tileGrid = [3,3]
    contrastSigma = 2.5

    overlap = 8
    overlap_pct = np.array([overlap, overlap * 4./3.])
    pixelSize = 8 * 1e-9
    autofocusOffsetFactor = 0.5

    mp = MosaicParameters(tileSizeIndex, tileGrid, overlap_pct, pixelSize, autofocusOffsetFactor)
    sp = ScanningParameters(scanRate, brightness, contrast, startingStig, focusScanRate)

    ##########################
    ### Initializing wafer ###

    wafer = Wafer(waferName, mp, sp)

    app = App(root)
    root.mainloop()
