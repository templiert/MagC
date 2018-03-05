'''
'Mag' usually refers to the magnetic resin that contains magnetic and fluorescent particles of different wavelengths
'''
#########################################################
### Start of config example for NikonPeter microscope ###
#########################################################
mic = 'NikonPeter'
micromanagerFolder = r'C:\Micro-Manager-1.4.23N'
mmConfigFile = r'E:\UserData\Templier\MM\MMConfig_BCNikon2_Default_Sungsik.cfg'
folderSave = os.path.join(r'E:\UserData\Templier\WorkingFolder', '')

NikonColors = ['Blue', 'Cyan', 'Green', 'Red', 'Teal', 'Violet', 'White'] # the name of the colors of the Lumencor source - /!\ Warning: white has to be last

# size of field of view in micrometers for different magnifications
magnificationImageSizes = {
20: [20,  1328.6, 1020.6],
63: [63, 220, 220],
}

# properties to read the current objective
objectiveProperties = ['TINosePiece', 'Label']

# properties to set during initialization
initialProperties = []
initialProperties.append(['Core', 'Focus','TIZDrive']) # otherwise the z drive is not recognized
initialProperties.append(['TIFilterBlock1', 'Label', '2-Quad']) # or 3-FRAP
initialProperties.append(['TILightPath', 'Label', '2-Left100']) # or 3-Right100 probably for the other camera
initialProperties.append(['Core', 'TimeoutMs', '20000']) # to prevent timeout during long stage movements
for NikonColor in NikonColors:
	initialProperties.append(['SpectraLED', NikonColor + '_Level', '100'])

# Change stage speed for accuracy: faster than 6 seemed to be inacurrate
initialProperties.append(['TIXYDrive', 'SpeedX', '6'])
initialProperties.append(['TIXYDrive', 'SpeedY', '6'])
initialProperties.append(['TIXYDrive', 'ToleranceX', '0'])
initialProperties.append(['TIXYDrive', 'ToleranceY', '0'])

acquisitionIntervalBF = 5 # in ms, acquisitionInterval during live brightfield imaging
acquisitionIntervalMag = 5 # in ms, acquisitionInterval during live fluo imaging

##############################
### All channel parameters ###
channelNames = {
'brightfield': ['White', '8-emty', ['SpectraLED', 'White_Level', '10']],
'dapi': ['Violet', '9-DAPI'],
488: ['Cyan', '0-FITC'],
546: ['Green', '5-mCherry'],
647: ['Red', '2-Cy5']}

objectives = [20, 63]
channelSpecs = ['exposure', 'offset']
channelContexts = ['imaging', 'focusing', 'live']
channelTargets = ['beads', 'tissue', 'general'] # 'general' used during live

# initialize the channels dictionnary: contains exposure times and z-offset of all channels
channels = {}
for channelName in channelNames:
	channels[channelName] = {}
	for objective in objectives:
		channels[channelName][objective] = {}
		for channelSpec in channelSpecs:
			channels[channelName][objective][channelSpec] = {}
			for channelContext in channelContexts:
				channels[channelName][objective][channelSpec][channelContext] = {}
for channelContext in channelContexts:
	channels[channelContext] = {}

objectiveBeads = 20
objectiveTissue = 63

### General exposure parameters independent of the imaging target (beads or tissue) for live imaging ###
channels['brightfield'][objectiveTissue]['exposure']['live']['general'] = 2
channels['dapi'][objectiveTissue]['exposure']['live']['general'] = 1
channels[488][objectiveTissue]['exposure']['live']['general'] = 1
channels[546][objectiveTissue]['exposure']['live']['general'] = 5
channels[647][objectiveTissue]['exposure']['live']['general'] = 5

channels['brightfield'][objectiveBeads]['exposure']['live']['general'] = 0.2
channels['dapi'][objectiveBeads]['exposure']['live']['general'] = 5
channels[488][objectiveBeads]['exposure']['live']['general'] = 5
channels[546][objectiveBeads]['exposure']['live']['general'] = 5
channels[647][objectiveBeads]['exposure']['live']['general'] = 5

### TISSUE-LIVE ### with objectiveTissue
channels['brightfield'][objectiveTissue]['exposure']['live']['tissue'] = 10
channels['dapi'][objectiveTissue]['exposure']['live']['tissue'] = 10
channels[488][objectiveTissue]['exposure']['live']['tissue'] = 10
channels[546][objectiveTissue]['exposure']['live']['tissue'] = 10
channels[647][objectiveTissue]['exposure']['live']['tissue'] = 10

### TISSUE-LIVE ### with objectiveBeads
channels['brightfield'][objectiveBeads]['exposure']['live']['tissue'] = 1
channels['dapi'][objectiveBeads]['exposure']['live']['tissue'] = 10
channels[488][objectiveBeads]['exposure']['live']['tissue'] = 10
channels[546][objectiveBeads]['exposure']['live']['tissue'] = 10
channels[647][objectiveBeads]['exposure']['live']['tissue'] = 10

### TISSUE-IMAGING ###
channels['brightfield'][objectiveTissue]['exposure']['imaging']['tissue'] = 1
channels['dapi'][objectiveTissue]['exposure']['imaging']['tissue'] = 500
channels[488][objectiveTissue]['exposure']['imaging']['tissue'] = 500
channels[546][objectiveTissue]['exposure']['imaging']['tissue'] = 500
channels[647][objectiveTissue]['exposure']['imaging']['tissue'] = 500

### BEADS-LIVE ###
channels['brightfield'][objectiveBeads]['exposure']['live']['beads'] = 0.2
channels['dapi'][objectiveBeads]['exposure']['live']['beads'] = 20
channels[488][objectiveBeads]['exposure']['live']['beads'] = 20
channels[546][objectiveBeads]['exposure']['live']['beads'] = 20
channels[647][objectiveBeads]['exposure']['live']['beads'] = 20

### BEADS-IMAGING ###
channels['brightfield'][objectiveBeads]['exposure']['imaging']['beads'] = 0.1
channels['dapi'][objectiveBeads]['exposure']['imaging']['beads'] = 100
channels[488][objectiveBeads]['exposure']['imaging']['beads'] = 100
channels[546][objectiveBeads]['exposure']['imaging']['beads'] = 100
channels[647][objectiveBeads]['exposure']['imaging']['beads'] = 100

### OFFSET-OBJECTIVEBEADS ###
channels['brightfield'][objectiveBeads]['offset']['imaging']['beads'] = 0
channels['dapi'][objectiveBeads]['offset']['imaging']['beads'] = 0
channels[488][objectiveBeads]['offset']['imaging']['beads'] = 0.925 
channels[546][objectiveBeads]['offset']['imaging']['beads'] = 0 # reference
channels[647][objectiveBeads]['offset']['imaging']['beads'] = 0 # reference

### OFFSET-TISSUE ###
channels[546][objectiveTissue]['offset']['imaging']['tissue'] = 0 # reference
channels['brightfield'][objectiveTissue]['offset']['imaging']['tissue'] = 0.2 # well calibrated ...
channels[647][objectiveTissue]['offset']['imaging']['tissue'] = 0.2
channels[488][objectiveTissue]['offset']['imaging']['tissue'] = 0
##############################

#######################################################
### End of config example for NikonPeter microscope ###
#######################################################

############################
# Sample specific parameters 
############################
waferName = 'C1_Wafer_500_Tissue'

# Mosaic Parameters for tissue
tileGrid = [2,2]
overlap = 20 # in percentage
# Mosaic Parameters for mag
tileGridMag = [1,1]
overlapMag = 20 # in percentage

# What channels should be used for bead imaging
channels['imaging']['beads'] = [488, 546, 'dapi', 'brightfield']

# What channels should be used for tissue imaging
channels['imaging']['tissue'] = [488, 546, 647, 'brightfield']
############################


###################
#### Constants ####
sleepSaver = 0.1 # the saver thread runs every sleepSaver second to check the savingQueue
liveGrabSleep = 0.1 # refresh cycle during the live visualization in s

# parameters of the cross displayed at the center of the field of view during live imaging
crossLength = 50
crossWidth = 5

stageInc = 1000 # displacement of the stage when the north/south/west/east buttons are pressed
#### End constants ####
#######################


import sys
sys.path.append(micromanagerFolder)
import MMCorePy

import os, time, datetime, shutil, pickle, argparse, tkFileDialog, subprocess, re, copy, random, json

from operator import itemgetter
import logging, colorlog # colorlog is not yet standard
#import logging

import threading
from threading import Thread
from Queue import Queue

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

import winsound

import numpy as np
from numpy import sin, pi, cos, arctan, tan, sqrt

from Tkinter import Label, LabelFrame, Button, Frame, Tk, LEFT, Canvas, Toplevel

import ctypes

import copy

import PIL # xxx does this not need tifffile ?
from PIL import Image, ImageTk

from scipy import fftpack
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.optimize import brent, minimize_scalar

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
		path = tkFileDialog.askopenfilename(title = text, initialdir = startingFolder)
	else:
		path = tkFileDialog.askopenfilename(title = text)
	logger.debug('Path chosen by user: ' + path)
	return path

def findFilesFromTags(folder,tags):
	filePaths = []
	for (dirpath, dirnames, filenames) in os.walk(folder):
		for filename in filenames:
			if (all(map(lambda x:x in filename,tags)) == True):
				path = os.path.join(dirpath, filename)
				filePaths.append(path)
	filePaths = naturalSort(filePaths)
	return filePaths
	
def readPoints(path):
	x,y = [], []
	with open(path, 'r') as f:
		lines = f.readlines()
		for point in lines:
			x.append(float(point.split('\t')[0]))
			try:
				y.append(float(point.split('\t')[1]))
			except Exception, e:
				pass
	logger.debug('Points read' + str([x,y]))
	return np.array([x,y])

def writePoints(path, points):
	with open(path, 'w') as f:
		for point in points:
			line = str(point[0]) + '\t' +  str(point[1]) + '\n'
			f.write(line)
	logger.debug('The point coordinates have been written')


def readSectionCoordinates(path):
	with open(path, 'r') as f:
		lines = f.readlines()
		sections = []
		for line in lines:
			points = line.split('\t')
			points.pop()
			section = [ [int(float(point.split(',')[0])), int(float(point.split(',')[1]))] for point in points ]
			sections.append(section)
	return sections

def naturalSort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)
	
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

def durationToPrint(d):
	return str(round(d/60., 1)) + ' min = ' + str(round(d/3600., 1)) + ' hours = ' + str(round(d/(3600.*24), 1)) + ' days'

def saver(q):
	while True:
		if not q.empty():
			logger.debug('Saving queue not empty')
			toSave = q.get()
			if toSave == 'finished':
				logger.debug('Saver thread is going to terminate')
				return
			else:
				sectionIndex, channel, tileId, folder, im, name = toSave
				fileName = 'section_' + str(sectionIndex).zfill(4) + '_channel_' + str(channel) + '_tileId_' + str(tileId[0]).zfill(2) + '-' + str(tileId[1]).zfill(2) + (len(name)>0) * ('-' + str(name)) + '.tif'
				path = os.path.join(folder, fileName)
				logger.debug('Saving snapped image in ' + path)
				# imsave(path, im) # with tifffile
				
				# with PIL and saving as png
				# im = np.array(im)
				result = PIL.Image.fromarray((im).astype(np.uint16))
				result.save(path)
				
		time.sleep(sleepSaver)

###########################
### Geometric functions       ###
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
	x_sourceLandmarks, y_sourceLandmarks = np.array(sourceLandmarks).T[:len(targetLandmarks.T)].T #sourceLandmarks trimmed to the number of existing targetlandmarks
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
#    print('Absolute errors in target coordinates : (xError, yError)')
#    for i in range(len(x_sourceLandmarks)):
	  #print ("%f, %f" % (
	#    np.abs(c[1]*x_sourceLandmarks[i] - c[0]*y_sourceLandmarks[i] + c[2] - x_targetLandmarks[i]),
	#    np.abs(c[1]*y_sourceLandmarks[i] + c[0]*x_sourceLandmarks[i] + c[3] - y_targetLandmarks[i])))

	#computing the accuracy
	x_target_computed_landmarks, y_target_computed_landmarks = applyAffineT(sourceLandmarks, c)
	accuracy = 0
	for i in range(len(x_targetLandmarks)):
			accuracy = accuracy + np.sqrt( np.square( x_targetLandmarks[i] - x_target_computed_landmarks[i] ) + np.square( y_targetLandmarks[i] - y_target_computed_landmarks[i] ) )
	accuracy = accuracy/float(len(x_sourceLandmarks) + 1)
#    print 'The mean accuracy in target coordinates is', accuracy

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

def getZInPlane(x,y,abc): #Fitted plane function
	return float(abc[0]*x + abc[1]*y + abc[2])

def focusThePoints(focusedPoints, pointsToFocus):
	x_pointsToFocus, y_pointsToFocus = pointsToFocus[0], pointsToFocus[1] # works even if pointsToFocus has no z coordinates
	x_focusedPoints, y_focusedPoints, z_focusedPoints = focusedPoints # focusedPoints of course has 3 coordinates
	
	# logger.debug('focusedPoints = ' + str(focusedPoints))
	# logger.debug('pointsToFocus = ' + str(pointsToFocus))
	# remove outliers
	idInliers = getInlierIndices(z_focusedPoints)
	# logger.debug('idInliers = ' + str(idInliers) )
	# logger.debug('There are ' + str(idInliers.size) + ' inliers in ' + str(map(lambda x:round(x, 2), z_focusedPoints*1e6)) + ' um' )
	if idInliers.size == 3:
		logger.warning('One autofocus point has been removed for interpolative plane calculation')
		x_focusedPoints, y_focusedPoints, z_focusedPoints = focusedPoints.T[idInliers].T
	elif idInliers.size < 3: 
		logger.warning('There are only ' + str(idInliers.size) + ' inliers for the interpolative plane calculation. A strategy should be developed to address such an event.')
	
	A = np.column_stack([x_focusedPoints, y_focusedPoints, np.ones_like(x_focusedPoints)])
	abc,residuals,rank,s = np.linalg.lstsq(A, z_focusedPoints)
	z_pointsToFocus = map(lambda a: getZInPlane (a[0],a[1],abc), np.array([x_pointsToFocus.transpose(), y_pointsToFocus.transpose()]).transpose())
	
	# calculating the accuracy
	z_check = np.array(map(lambda a: getZInPlane (a[0],a[1],abc), np.array([x_focusedPoints.transpose(), y_focusedPoints.transpose()]).transpose()))
	diff = z_check - z_focusedPoints
	meanDiff = np.mean(np.sqrt(diff * diff))
	logger.debug('The plane difference is  ' + str(diff*1e6) + ' um')
	logger.info('The mean distance of focus points to the plane is ' + str(round(meanDiff*1e6, 3))    + ' um')
	
	return np.array([x_pointsToFocus, y_pointsToFocus, z_pointsToFocus])

def transformCoordinates(coordinates, center, angle):
	return (translate(rotate(coordinates.T, angle), center)).T

def getInlierIndices(data, m = 2.5):
	d = np.abs(data - np.median(data))
	# mdev = float(np.median(d))
	mdev = float(np.mean(d))
	s = d/mdev if mdev else np.array(len(data) * [0.])
	# print 'mdev', mdev
	# print 'd', d
	# print 's', s
	return np.where(s <= m)[0]

def getOutlierIndices(data, m = 2.5):
	d = np.abs(data - np.median(data))
	# mdev = float(np.median(d))
	mdev = float(np.mean(d))
	s = d/mdev if mdev else np.array(len(data) * [0.])
	# print 'mdev', mdev
	# print 'd', d
	# print 's', s
	return np.where(s > m)[0]
	
def bbox(points):
	minx, miny = 1e9, 1e9
	maxx, maxy = -1e9, -1e9
	for point in points:
		if point[0] > maxx:
			maxx = point[0]
		if point[0] < minx:
			minx = point[0]
		if point[1] > maxy:
			maxy = point[1]
		if point[1] < miny:
			miny = point[1]
	return minx, miny, maxx-minx, maxy-miny

def gridInBb(bb, gridLayout = None, gridSpacing = None):
	if gridLayout is not None:
		gridSpacing = np.array(bb[2:])/(np.array(gridLayout)-1)
	else:
		gridLayout = (np.array(bb[2:])/np.array(gridSpacing)).astype(int)
		gridSpacing = np.array(bb[2:])/(np.array(gridLayout)-1)
		
	topLeftCorner = bb[:2]
	gridPoints = []
	for x in range(gridLayout[0]):
		for y in range(gridLayout[1]):
			gridPoints.append( topLeftCorner + np.array([x,y]) * gridSpacing)
	return np.array(gridPoints), gridLayout, gridSpacing
	
	# # x = np.linspace(bb[0], bb[0] + bb[2], gridLayout[0])
	# # y = np.linspace(bb[1], bb[1] + bb[3], gridLayout[1])
	
def bestNeighbors(gridPoints, targetPoints):
	bestNeighbors = []
	targetPoints = np.array(targetPoints).T[:2].T
	for gridPoint in gridPoints:
		distances = ((np.array(targetPoints) - gridPoint)**2).sum(axis=1)
		sortedDistances = distances.argsort()
		bestNeighbors.append(targetPoints[sortedDistances[0]])
	return np.array(bestNeighbors)

def getNearestPoints(points, refPoint, nPoints = 1):
	'''
	Returns the nPoints indices of points indicating the closest points to the reference point refPoint
	'''
	# return (min((hypot(x2-refPoint[0],y2-refPoint[1]), x2, y2) for x2,y2 in points))[1:3]
	
	sortedDistances = sorted( [[np.linalg.norm(np.array([point[0],point[1]]) - np.array(refPoint)), id] for id, point in enumerate(points)], key = lambda x: x[0], reverse = False)[:nPoints]
	
	# print 'sortedDistances XXX UUU', sortedDistances
	# return np.array([x[1:3] for x in sortedDistances]), np.array([x[0] for x in sortedDistances]) # centers, indices
	return np.array([x[1] for x in sortedDistances]) # indices

def getPlaneMesh(array2D, array3D, grid):     
	'''
	array2D gives the x,y boundaries for the mesh
	array3D gives the interpolative plane
	'''
	# grid = 50
	x = np.linspace(np.min(array2D[0]), np.max(array2D[0]), grid)
	y = np.linspace(np.min(array2D[1]), np.max(array2D[1]), grid)

	xv, yv = np.meshgrid(x, y)

	xvFlat = [item for sublist in xv for item in sublist]
	yvFlat = [item for sublist in yv for item in sublist]

	planeMesh = focusThePoints( array3D, np.array([xvFlat, yvFlat]))
	return planeMesh

def getRBFMesh(array2D, array3D, grid):       
	'''
	array2D gives the x,y boundaries for the mesh
	array3D gives the interpolative plane
	'''
	# grid = 50
	x = np.linspace(np.min(array2D[0]), np.max(array2D[0]), grid)
	y = np.linspace(np.min(array2D[1]), np.max(array2D[1]), grid)

	xv, yv = np.meshgrid(x, y)

	xvFlat = np.array([item for sublist in xv for item in sublist])
	yvFlat = np.array([item for sublist in yv for item in sublist])

	rbf = Rbf(np.array(array3D[0], array3D[1], array3D[2], epsilon = 2, function = 'thin_plate'))
	
	autofocusedRBF = np.array([xvFlat, yvFlat, rbf(xvFlat, yvFlat)]).T
	
	return autofocusedRBF

#####################
### GUI Functions ###
dirtyCounter = 0

class App:
	global wafer
	def __init__(self, master):
		self.live = False
		self.acquisitionInterval = acquisitionIntervalBF
		
		self.frame = Frame(master)
		self.frame.pack()
		
		self.button1 = Button(self.frame, text='Acquire tissue *HAF*', command = self.tissueAcquireHAF)
		self.button1.pack(side=LEFT)

		self.button95 = Button(self.frame, text='Acq. manual mosaic *HAF*', command = self.tissueAcquireHAFFromManualSections)
		self.button95.pack(side=LEFT)

		self.button50 = Button(self.frame, text='Acquire tissue *manual*', command = self.tissueAcquireManual)
		self.button50.pack(side=LEFT)

		self.button2 = Button(self.frame, text='Add mosaic here', command = self.addMosaicHere)
		self.button2.pack(side=LEFT)

		self.button3 = Button(self.frame, text='Add lowres landmark', command = self.addLowResLandmark)
		self.button3.pack(side=LEFT)

		self.button22 = Button(self.frame, text='Add highres landmark', command = self.addHighResLandmark)
		self.button22.pack(side=LEFT)

		self.button4 = Button(self.frame, text='Load wafer', command = self.loadWafer)
		self.button4.pack(side=LEFT)
		
		self.button44 = Button(self.frame, text='ResetWaferKeepTargets', command = self.resetWaferKeepTargetCalibration)
		self.button44.pack(side=LEFT)         
		
		self.button5 = Button(self.frame, text='Load sections and landmarks from pipeline', command = self.loadSectionsAndLandmarksFromPipeline)
		self.button5.pack(side=LEFT)

		self.button6 = Button(self.frame, text='Acquire mag *HAF*', command = self.magAcquireHAF)
		self.button6.pack(side=LEFT)

		self.button24 = Button(self.frame, text='Acquire mag *manual*', command = self.magAcquireManual)
		self.button24.pack(side=LEFT)
		
		self.button7 = Button(self.frame, text='Save Wafer', command = self.saveWafer)
		self.button7.pack(side=LEFT)

		self.button9 = Button(self.frame, text='Stop live', command = self.stopLive)
		self.button9.pack(side=LEFT)
		
		# # # self.buttonN = Button(self.frame, text='N', command = self.north)
		# # # self.buttonN.pack(side=LEFT)

		# # # self.buttonS = Button(self.frame, text='S', command = self.south)
		# # # self.buttonS.pack(side=LEFT)

		# # # self.buttonE = Button(self.frame, text='E', command = self.east)
		# # # self.buttonE.pack(side=LEFT)

		# # # self.buttonW = Button(self.frame, text='W', command = self.west)
		# # # self.buttonW.pack(side=LEFT)
		
		# self.button11 = Button(self.frame, text='GoToNextMag', command = self.goToNextMag)
		# self.button11.pack(side=LEFT)

		# self.button12 = Button(self.frame, text='resetGreenFocus', command = self.resetGreenFocus)
		# self.button12.pack(side=LEFT)

		# # self.button13 = Button(self.frame, text='rbf', command = self.rbf)
		# # self.button13.pack(side=LEFT)

		# # self.button14 = Button(self.frame, text='plane', command = self.plane)
		# # self.button14.pack(side=LEFT)
		
		# self.button15 = Button(self.frame, text='goToNextMagFocus', command = self.goToNextMagFocus)
		# self.button15.pack(side=LEFT)

		self.button66 = Button(self.frame, text='Live Dapi', command = self.liveDapi)
		self.button66.pack(side=LEFT)

		self.button16 = Button(self.frame, text='Live Green', command = self.liveGreen)
		self.button16.pack(side=LEFT)

		self.button30 = Button(self.frame, text='Live Red', command = self.liveRed)
		self.button30.pack(side=LEFT)

		self.button8 = Button(self.frame, text='Live BF', command = self.liveBF)
		self.button8.pack(side=LEFT)

		self.button87 = Button(self.frame, text='Live 647', command = self.live647)
		self.button87.pack(side=LEFT)


		# self.button17 = Button(self.frame, text='AF In Place', command = self.afInPlace)
		# self.button17.pack(side=LEFT)

		self.button18 = Button(self.frame, text='Snap', command = self.snap)
		self.button18.pack(side=LEFT)
		
		# self.button19 = Button(self.frame, text='ManualRetakesMag', command = self.manualRetakesMag)
		# self.button19.pack(side=LEFT)

		self.button20 = Button(self.frame, text='logHAF', command = self.logHAF)
		self.button20.pack(side=LEFT)

		self.button23 = Button(self.frame, text='HAF', command = self.logHAF)
		self.button23.pack(side=LEFT)

		self.button27 = Button(self.frame, text='ToggleNikonHAF', command = self.toggleNikonAutofocus)
		self.button27.pack(side=LEFT)
		
		self.button21 = Button(self.frame, text='ResetImagedSections', command = self.resetImagedSections)
		self.button21.pack(side=LEFT)

		self.button81 = Button(self.frame, text='getXYZ', command = self.logXYZ)
		self.button81.pack(side=LEFT)

		
		self.buttonQuit = Button(self.frame, text='Quit', command = root.destroy)
		self.buttonQuit.pack(side=LEFT)

		photo = ImageTk.PhotoImage(PIL.Image.fromarray(np.zeros((int(imageSize_px[0]), int(imageSize_px[1])))))
		self.label = Label(master, image = photo)
		self.label.image = photo # keep a reference?
		self.label.pack()

		# self.canvas = Canvas(master, width = imageSize_px[0], height = imageSize_px[1])
		# self.imageCanvas = self.canvas.create_image(0, 0, image = photo)
		# self.canvas.grid(row = 0, column = 0)
		
	def snap(self):
		if mmc.isSequenceRunning():
			mmc.stopSequenceAcquisition()
		takeImage(0, [0,0], folderSave, str(int(time.time())))
		
	def north(self):
		setXY(getXY()[0], getXY()[1] + stageInc)
		
	def south(self):
		setXY(getXY()[0], getXY()[1] - stageInc)
		
	def east(self):
		setXY(getXY()[0] - stageInc, getXY()[1])
		
	def west(self):
		setXY(getXY()[0] + stageInc, getXY()[1])
		
	def acquireWaferButtonAction(self):
		self.stopLive()
		# # wafer.magSections = []
		# # wafer.createMagSectionsFromPipeline()
		# # self.frame.quit()
		self.acquireWafer()

	def resetGreenFocus(self):
		wafer.targetMagFocus = []
		
	def rbf(self):
		rbf = Rbf(np.array(wafer.targetMagFocus).T[0], np.array(wafer.targetMagFocus).T[1], np.array(wafer.targetMagFocus).T[2], epsilon = 2, function = 'thin_plate')
		wafer.targetMagCenters = np.array([np.array(wafer.targetMagCenters).T[0], np.array(wafer.targetMagCenters).T[1], rbf(np.array(wafer.targetMagCenters).T[0], np.array(wafer.targetMagCenters).T[1])    ]).T
		logger.debug('rbf')
	
	def plane(self):
		wafer.targetMagCenters = focusThePoints(np.array(wafer.targetMagFocus).T, np.array(wafer.targetMagCenters).T).T
		logger.debug('Plane')
		
	def liveGrab(self):
		logger.debug('Starting liveGrab ' + str(self.acquisitionInterval) )
		if not mmc.isSequenceRunning():
			mmc.startContinuousSequenceAcquisition(self.acquisitionInterval)
		time.sleep(0.2)
			
		currentChannel = getChannel()     
		logger.debug('** LIVEGRAB current channel ** ' +  str(currentChannel))
		try:
			while self.live:
				if (mmc.getRemainingImageCount() > 0):
					lastImage = mmc.getLastImage()
					if currentChannel == 'brightfield':
						im = np.uint8(lastImage/256)
	#                     im = mmc.getLastImage()
					else:
						# im = np.uint8(lastImage/256)
						# im = np.uint8(lastImage/256 + 50)
						im = lastImage
					for x in range(crossLength):
						for y in range(crossWidth):
							im[imageSize_px[1]/2 - crossWidth/2 + y][imageSize_px[0]/2 - int(crossLength/2.) + x] = 0
					for y in range(crossLength):
						for x in range(crossWidth):
							im[imageSize_px[1]/2 - int(crossLength/2.) + y ][imageSize_px[0]/2 - crossWidth/2 + x] = 0
				else:
					im = np.zeros((30, 30))
					
				im = im[::2, ::2]
				photo = ImageTk.PhotoImage(PIL.Image.fromarray(im))
				self.label.configure(image=photo)
				self.label.image = photo # keep a reference!
				self.label.pack()
				
				# self.canvas.itemconfig(self.imageCanvas, image = photo)
				
				time.sleep(self.acquisitionInterval/50.)
		except Exception, e:
			logger.error('In liveGrab: ' + str(e))
		logger.debug('liveGrab terminated')

	def liveGreen(self):
		# global live, liveThread
		self.acquisitionInterval = acquisitionIntervalMag
		self.live = True
		self.liveThread = Thread(target = self.liveGrab)
		
		##########################################
		# # # Setting fluo conditions # # #
		setChannel(488)
		if mic == 'Z2':
			setExposure(channels[488][5]['exposure']['live']['beads'])
		elif mic == 'Leica' or mic == 'Nikon' or mic == 'NikonPeter':
			setExposure(channels[488][currentObjectiveNumber]['exposure']['live']['general'])
		mmc.setAutoShutter(False)
		openShutter()
		self.liveThread.start()

	def liveRed(self):
		# global live, liveThread
		self.acquisitionInterval = acquisitionIntervalMag
		self.live = True
		self.liveThread = Thread(target = self.liveGrab)
		
		##########################################
		# # # Setting fluo conditions # # #
		setChannel(546)
		if mic == 'Z2':
			setExposure(channels[546][5]['exposure']['live']['beads'])
		elif mic == 'Leica' or mic == 'Nikon' or mic == 'NikonPeter':
			setExposure(channels[546][currentObjectiveNumber]['exposure']['live']['general'])
		mmc.setAutoShutter(False)
		openShutter()
		self.liveThread.start()

	def live647(self):
		# global live, liveThread
		self.acquisitionInterval = acquisitionIntervalMag
		self.live = True
		self.liveThread = Thread(target = self.liveGrab)
		
		##########################################
		# # # Setting fluo conditions # # #
		setChannel(647)
		if mic == 'Z2':
			setExposure(channels[546][5]['exposure']['live']['beads'])
		elif mic == 'Leica' or mic == 'Nikon' or mic == 'NikonPeter':
			setExposure(channels[546][currentObjectiveNumber]['exposure']['live']['general'])
		mmc.setAutoShutter(False)
		openShutter()
		self.liveThread.start()


	def liveDapi(self):
		# global live, liveThread
		self.acquisitionInterval = acquisitionIntervalMag
		self.live = True
		self.liveThread = Thread(target = self.liveGrab)
		
		##########################################
		# # # Setting fluo conditions # # #
		setChannel('dapi')
		if mic == 'Z2':
			setExposure(channels['dapi'][5]['exposure']['live']['beads'])
		elif mic == 'Leica' or mic == 'Nikon' or mic == 'NikonPeter':
			setExposure(channels['dapi'][currentObjectiveNumber]['exposure']['live']['general'])
		mmc.setAutoShutter(False)
		openShutter()
		self.liveThread.start()

	def liveBF(self):
		# global live, liveThread
		self.acquisitionInterval = acquisitionIntervalMag
		self.live = True
		self.liveThread = Thread(target = self.liveGrab)
		
		##########################################
		# # # Setting fluo conditions # # #
		setChannel('brightfield')
		if mic == 'Z2':
			setExposure(channels['brightfield'][5]['exposure']['live']['beads'])
		elif mic == 'Leica' or mic == 'Nikon' or mic == 'NikonPeter':
			setExposure(channels['brightfield'][currentObjectiveNumber]['exposure']['live']['general'])
		mmc.setAutoShutter(False)
		openShutter()
		self.liveThread.start()
		
	def stopLive(self):
		self.live = False
		time.sleep(0.2)
		if mmc.isSequenceRunning():
			mmc.stopSequenceAcquisition()
		closeShutter()
		# root.destroy() # no, it kills everything ...

	def addLowResLandmark(self):
		stageXY = getXY()
		logger.info('wafer.targetLowResLandmarks --- before --- ' + str(wafer.targetLowResLandmarks))
		wafer.targetLowResLandmarks.append([stageXY[0], stageXY[1], getZ()])
		logger.info('wafer.targetLowResLandmarks --- after --- ' + str(wafer.targetLowResLandmarks))

		nValidatedLandmarks = len(wafer.targetLowResLandmarks)
		if nValidatedLandmarks == len(wafer.sourceLandmarks.T): # all target landmarks have been identified
			logger.info('Good. All landmarks have been calibrated.')
			wafer.save()                
			writePoints(os.path.join(wafer.pipelineFolder, 'target_lowres_landmarks.txt'), wafer.targetLowResLandmarks)
			# # # # self.generateSections() # now done with the high res calibration
			# # # # wafer.save()
		elif nValidatedLandmarks > 1:
			logger.info('There are still ' + str(len(wafer.sourceLandmarks[0]) - nValidatedLandmarks ) + ' landmarks to calibrate. The stage has been moved to the next landmark to be calibrated')
			
			nextXY = affineT(wafer.sourceLandmarks, np.array(wafer.targetLowResLandmarks).T[:2], wafer.sourceLandmarks).T[nValidatedLandmarks] 
			logger.debug('Computing nextXY: wafer.sourceLandmarks - ' + str(wafer.sourceLandmarks) + ' np.array(wafer.targetLowResLandmarks).T[:2] - ' + str(np.array(wafer.targetLowResLandmarks).T[:2]) + ' nextXY = ' + str(nextXY))
			setXY(*nextXY)
			
			if nValidatedLandmarks > 3:
				nextZ = focusThePoints(np.array(wafer.targetLowResLandmarks).T, np.array([[nextXY[0]], [nextXY[1]]]))[2][0] # the interpolative plane is calculated on the fly
				print 'nextZ', nextZ
				setZ(nextZ)
		else:
			logger.info('Please go manually to the second landmark.')

	def addHighResLandmark(self):
		nHighRes = len(wafer.targetHighResLandmarks)
		if nHighRes == 0:
			setXY(*wafer.targetLowResLandmarks[0][:2])
			logger.info('Just moved to first landmark. Adjust this first landmark position with high resolution')
			wafer.targetHighResLandmarks.append('dummy')
		elif wafer.targetHighResLandmarks[0] == 'dummy':
			wafer.targetHighResLandmarks.pop()
			stageXY = getXY() 
			wafer.targetHighResLandmarks.append([stageXY[0], stageXY[1], getZ()])
			setXY(*wafer.targetLowResLandmarks[1][:2])
			logger.info('First high res landmark calibrated. Just moved to the second low res landmark: please adjust it.')
		elif nHighRes == len(wafer.targetLowResLandmarks):
			logger.info('HighRes target wafers had already been calibrated. Reinitializing calibration ...')
			wafer.targetHighResLandmarks = []
			setXY(*wafer.targetLowResLandmarks[0][:2])
			logger.info('Just moved to first landmark. Adjust this first landmark position with high resolution')
			wafer.targetHighResLandmarks.append('dummy')
		
		else:
			stageXY = getXY()
			wafer.targetHighResLandmarks.append([stageXY[0], stageXY[1], getZ()])

			if not (len(wafer.targetHighResLandmarks) == len(wafer.targetLowResLandmarks)):
				nextXY = affineT(wafer.sourceLandmarks, np.array(wafer.targetHighResLandmarks).T[:2], wafer.sourceLandmarks).T[len(wafer.targetHighResLandmarks)]
				setXY(*nextXY)
				logger.info(str(len(wafer.targetHighResLandmarks)) + ' high res landmarks calibrated.')
			else:
				logger.info('All high res landmarks calibrated. Generating all sections.')
				wafer.save()
				writePoints(os.path.join(wafer.pipelineFolder, 'target_highres_landmarks.txt'), wafer.targetHighResLandmarks)
				self.generateSections()
				wafer.save()                

			
	def resetWaferKeepTargetCalibration(self): # no need to make wafer glogal as it was already global, right ?
		newWafer = Wafer(waferName, ip)
		newWafer.targetLowResLandmarks = copy.deepcopy(wafer.targetLowResLandmarks)
		newWafer.targetHighResLandmarks = copy.deepcopy(wafer.targetHighResLandmarks)
		newWafer.targetMagFocus = copy.deepcopy(wafer.targetMagFocus)
		newWafer.sourceLandmarks = copy.deepcopy(wafer.sourceLandmarks)
		newWafer.sourceMagDescription = copy.deepcopy(wafer.sourceMagDescription)
		newWafer.sourceSectionsMagCoordinates = copy.deepcopy(wafer.sourceSectionsMagCoordinates)
		newWafer.sourceSectionsTissueCoordinates = copy.deepcopy(wafer.sourceSectionsTissueCoordinates)
		newWafer.sourceTissueDescription = copy.deepcopy(wafer.sourceTissueDescription)
		self = newWafer # wtf is that ?
		self.createSections()
		
	def addMosaicHere(self):
		wafer.addCurrentPosition()
	
	def generateSections(self):
		wafer.createSections()
						
	def acquireWafer(self):
		self.stopLive()
		wafer.acquire()
		
	def loadWafer(self):
		global wafer
		waferPath = getPath('Select the wafer pickle file', startingFolder = folderSave)
		f = open(waferPath, 'r')
		wafer = pickle.load(f)
		f.close()
		for magSection in wafer.magSections:
			magSection.localized = False
		
		
	def saveWafer(self):
		wafer.save()

	# def manualRetakesMag(self):
		# wafer.manualRetakes(mag = True)
	
	def loadSectionsAndLandmarksFromPipeline(self):
		pipelineFolder = getDirectory('Select the folder containing the sections and landmarks from the pipeline', startingFolder = folderSave)
		wafer.pipelineFolder = pipelineFolder # needed to write the target landmark coordinates for later proper orientation
		
		sourceSectionsMagPath = os.path.join(pipelineFolder, 'source_sections_mag.txt.')
		sourceSectionsTissuePath = os.path.join(pipelineFolder, 'source_sections_tissue.txt.')
		
		wafer.sourceSectionsMagCoordinates = readSectionCoordinates(sourceSectionsMagPath) # list of lists
		wafer.sourceSectionsTissueCoordinates = readSectionCoordinates(sourceSectionsTissuePath) # list of lists

		# # wafer.sourceSectionCenters = np.array([getCenter(np.array(sourceSectionTissueCoordinates).T) for sourceSectionTissueCoordinates in wafer.sourceSectionsCoordinates])
		
		sourceTissueMagDescriptionPath = os.path.join(pipelineFolder, 'source_tissue_mag_description.txt.') # 2 sections: template tissue and template mag
		if os.path.isfile(sourceTissueMagDescriptionPath):
			wafer.sourceTissueDescription, wafer.sourceMagDescription= readSectionCoordinates(sourceTissueMagDescriptionPath)
		else:
			logger.warning('There is no source_tissue_mag_description')
		
		sourceLandmarksPath = os.path.join(pipelineFolder, 'source_landmarks.txt.')
		wafer.sourceLandmarks = readPoints(sourceLandmarksPath)

		sourceROIDescriptionPath = os.path.join(pipelineFolder, 'source_ROI_description.txt')
		if os.path.isfile(sourceROIDescriptionPath):
			wafer.sourceROIDescription = readSectionCoordinates(sourceROIDescriptionPath)
		else:
			logger.warning('There is no source_ROI_description. The center of the section will be used.')

	def magAcquireHAF(self):
		self.stopLive()
		wafer.magSections = []
		wafer.createMagSectionsFromPipeline()
		wafer.magAcquire()

	def magAcquireManual(self):
		self.stopLive()
		if wafer.magSections == []:
			wafer.magSections = []
			wafer.createMagSectionsFromPipeline()
		wafer.magAcquire(manualFocus = True)
		self.liveGreen()
		time.sleep(0.2)
		self.liveGreen()
		winsound.Beep(440,100)

	def tissueAcquireHAF(self):
		self.stopLive()
		wafer.sections = []
		wafer.createSectionsFromPipeline()
		wafer.acquire()

	def tissueAcquireHAFFromManualSections(self):
		self.stopLive()
		wafer.acquire(manualFocus = False)

	def tissueAcquireManual(self):
		self.stopLive()
		if wafer.sections == []:
			wafer.sections = []
			wafer.createSectionsFromPipeline()
		wafer.acquire(manualFocus = True)
		self.liveRed()
		time.sleep(0.2)
		self.liveRed()
		time.sleep(0.2)
		self.liveRed()
		winsound.Beep(440,100)
		
	def afInPlace(self):
		# setZSnapAndGetFocusScore(getZ())
		beadAutofocus()
		
	def manualRetakesMag(self, mag = True):
		self.liveGreen()
		idsToRetake = readPoints(findFilesFromTags(folderSave, ['manualRetakes'])[0])[0]
		
		if not hasattr(self, 'counterRetake'):
			self.counterRetake = -1
			sectionToRetake = wafer.magSections[idsToRetake[0] - 1]
			x,y,z = sectionToRetake.center[0], sectionToRetake.center[1], sectionToRetake.startingZ
			setXY(x, y)
		elif self.counterRetake < len(idsToRetake):
			self.counterRetake = self.counterRetake + 1
			idToRetake = idsToRetake[self.counterRetake]
			logger.info('Retaking manually mag section number ' + str(idToRetake))
		
			sectionToRetake = wafer.magSections[idToRetake-1]
			sectionToRetake.focusedZ = getZ()
			# update the taken flag ?

			self.stopLive()
			
			for imageToDelete in os.listdir(sectionToRetake.folderSectionSave):
				os.remove(os.path.join(sectionToRetake.folderSectionSave, imageToDelete))
			time.sleep(1)
			openShutter()
			# retake
			
			for idTile, point in enumerate(sectionToRetake.imagingCoordinates['tiles']):
				setXY(*point[:2]) # Z has just been set
				for imagingChannel in channels['imaging']['beads']:
					setChannel(imagingChannel)
					setExposure(channels[imagingChannel][objectiveBeads]['exposure']['imaging']['beads'])
					logger.debug('Scanning tile ' + str(idTile) + ' with channel ' + str(imagingChannel))
					
					logger.info('Deleting in folder ' + str(sectionToRetake.folderSectionSave))
					
					takeImage(sectionToRetake.index, sectionToRetake.ip.idsMag[idTile], sectionToRetake.folderSectionSave, name = 'mag')

			# closeShutter()
			

			# get section and set x,y,z
			sectionToRetake = wafer.magSections[idsToRetake[self.counterRetake + 1] - 1]
			x,y,z = sectionToRetake.center[0], sectionToRetake.center[1], sectionToRetake.startingZ
			setXY(x, y)
			self.liveGreen()
		
		# # ask user for focus
		# logger.info('Please focus then click ok')
		
		# # # v = True
		# # # t = time.time()
		# # # while v:
			# # # if time.time() - t > 5:
				# # # v = False
		
		# ctypes.windll.user32.MessageBoxW(0,'Are you done with manual focusing ?','Do it',0)
		
		# wafer.save()
		# logger.info('MagSection number ' + str(idToRetake) + ' has been retaken')


	# def manualRetakesMag(self, mag = True):
		# self.liveGreen()
	
	def logHAF(self):
		logger.debug(str(mmc.getLastFocusScore()) + ' getLastFocusScore')
		logger.debug(str(mmc.getCurrentFocusScore()) + ' getCurrentFocusScore')
		logger.debug(str(mmc.isContinuousFocusEnabled()) + ' isContinuousFocusEnabled')
		logger.debug(str(mmc.isContinuousFocusLocked()) + ' isContinuousFocusLocked')
		# logger.debug(str(mmc.isContinuousFocusDrive(zStage)) + ' isContinuousFocusDrive') # xxx isContinuousFocusDrive (const char *stageLabel)
		if mic == 'Leica':
			logger.debug(str(mmc.getAutoFocusOffset()) + ' getAutoFocusOffset')
		if mic == 'Nikon' or mic == 'NikonPeter':
			logger.debug(str(mmc.getPosition('TIPFSOffset')) + ' autoFocusOffset')
		
		logger.debug(str(getZ()) + ' getZ')

	def HAF(self):
		autofocus()
		self.logHAF()
		
	def resetImagedSections():
		for section in wafer.sections:
			section.acquireFinished = False
		logger.info('The "imaged" tag has been reverted to False for all sections from current wafer (you need to save manually) ')
	
	def toggleNikonAutofocus(self):
		if mmc.isContinuousFocusEnabled():
			mmc.enableContinuousFocus(False)
		else:
			mmc.enableContinuousFocus(True)

	def logXYZ(self):
		x,y = getXY()
		z = getZ()
		logger.info(str(x) + ',' + str(y) + ',' + str(z))

############################
### Microscope functions ###
def getZ():
	return mmc.getPosition(zStage)

def getXY():
	sensorXY = mmc.getXPosition(stage), mmc.getYPosition(stage)
	logger.debug('Sensor read stage ' + str([round(sensorXY[0], 3),round(sensorXY[1], 3)]) + ' um' )
	if mic == 'Leica':
		x = sensorXY[0]
		y = -sensorXY[1]
	elif mic == 'Nikon':
		x = sensorXY[0]
		y = -sensorXY[1]/float(4) # there is an amazing 4x factor between x and y axes !
	elif mic == 'NikonPeter':
		x = sensorXY[0]
		y = sensorXY[1] # to calibrate xxx
		
	else:
		x = sensorXY[0]
		y = sensorXY[1]
	return np.array([x, y])

def setXY(x,y):
	logger.debug('Moving stage to ' + str([round(x, 3),round(y, 3)]) + ' um' )
#    mmc.waitForDevice(stage)
#    time.sleep(0.1)       

	if mic == 'Leica':
		mmc.setXYPosition(stage, x , -y) # the y axis is flipped
	elif mic == 'Nikon':
		try:
			mmc.setXYPosition(stage, x , -y * 4) # the y axis is  flipped and there is a factor between the two axes
		except Exception, e:
			logger.error('*** STAGE ERROR LEVEL 1 - TRYING AGAIN ***')
			try:
				mmc.setXYPosition(stage, x , -y * 4) # the y axis is  flipped and there is a factor between the two axes
			except Exception, e:
				logger.error('*** STAGE ERROR LEVEL 2 - TRYING AGAIN ***')
				mmc.setXYPosition(stage, x , -y * 4) # the y axis is  flipped and there is a factor between the two axes

	elif mic == 'NikonPeter':
		mmc.setXYPosition(stage, x , y) # xxx to calibrate
	else:
		mmc.setXYPosition(stage, x , y)
	mmc.waitForDevice(stage)
	newXY = getXY() 
	logger.debug('Moved stage to ' + str([round(newXY[0], 3),round(newXY[1], 3)]) + ' um' )
	
def setZ(z):
	mmc.setPosition(zStage,z)
	mmc.waitForDevice(zStage)
	logger.debug('zStage moved to ' + str(round(z,3)) + ' um')

def isShutterOpen():
	# return (int(mmc.getProperty('ZeissReflectedLightShutter', 'State')) == 1)
	return mmc.getShutterOpen()
	
def openShutter():
	if mic == 'Leica':
		if not isShutterOpen():
			mmc.setShutterOpen(True)
			mmc.waitForDevice(shutter)

def closeShutter():
	if mic == 'Nikon':
		logger.debug('close shutter on the nikon means disenabling the lumencor channels')
		mmc.setProperty('Spectra', 'White_Enable', 0)
		mmc.setProperty('Spectra', 'YG_Filter', 1)
	elif mic == 'NikonPeter':
		logger.debug('close shutter on the nikon means disenabling the lumencor channels')
		mmc.setProperty('SpectraLED', 'White_Enable', 0)
		mmc.setProperty('SpectraLED', 'YG_Filter', 1)

def autofocus():
	try:
		mmc.fullFocus()
		focusedZ = getZ()
		logger.debug('Hardware autofocus performed: z = ' + str(round(focusedZ,3)) + ' um')
	except Exception,e:
		logger.error('### AUTOFOCUS FAIL ###')
		logger.error(e)

		if mmc.isContinuousFocusEnabled(): # reactivating the autofocus, typical error is dichroic not in place
			mmc.enableContinuousFocus(False)
		else:
			mmc.enableContinuousFocus(True)

		time.sleep(1)
		mmc.fullFocus()
		focusedZ = getZ()
		logger.debug('Hardware autofocus performed AFTER ERROR: z = ' + str(round(focusedZ,3)) + ' um')
		
	return focusedZ

def setChannel(channel):
	if mic == 'Z2':
		microscopeChannelName = microscopeChannelNames[channelNames.index(channel)]
		mmc.setProperty('ZeissReflectorTurret', 'Label', microscopeChannelName)       
		mmc.waitForDevice('ZeissReflectorTurret')
		# mmc.setShutterOpen(False)
	elif mic == 'Leica':
		mmc.setProperty('FastFilterWheelEX', 'Label', channelNames[channel][0])
		mmc.setProperty('FastFilterWheelEM', 'Label', channelNames[channel][1])
		mmc.waitForDevice('FastFilterWheelEM') #xxx
		mmc.waitForDevice('FastFilterWheelEX') #xxx
	elif mic == 'Nikon':
		if len(channelNames[channel]) > 2:
			for additionalState in channelNames[channel][2:]:
				mmc.setProperty(additionalState[0], additionalState[1], additionalState[2])
		else:
			if mmc.getProperty('TIFilterBlock1', 'Label') != '2-Quad': # if the standard NikonBodyCube has been changed for brightfield. This could be smartly managed to have reduce the number of switches by half (would gain about 2 hours for 1000 sections x [3x3] mosaics) 
				mmc.setProperty('TIFilterBlock1', 'Label', '2-Quad')
#             print 'mmc.getProperty(Spectra,Green_Level)', mmc.getProperty('Spectra', 'Green_Level')
#             if mmc.getProperty('Spectra', 'Green_Level') != '100': #  does not work I do not understand why. The state is already read as being 100 but it is effectively still at 2 percent ...
			logger.error('Setting green level back to 100') # what is that ?
			mmc.setProperty('Spectra', 'Green_Level', '100')

		if channelNames[channel][0] == 'White':
			mmc.setProperty('Spectra', 'White_Enable', 1)
		else:
			mmc.setProperty('Spectra', 'White_Enable', 0) # closes all channels
			mmc.setProperty('Spectra', channelNames[channel][0] + '_Enable', 1)
		mmc.setProperty('CSUW1-Filter Wheel', 'Label', channelNames[channel][1])
		mmc.setProperty('Spectra', 'YG_Filter', 1) # later in case some delay is needed ...
	elif mic == 'NikonPeter':
		if len(channelNames[channel]) > 2: # happens only for brightfield actually
			for additionalState in channelNames[channel][2:]:
				mmc.setProperty(additionalState[0], additionalState[1], additionalState[2])
		else: 
			logger.debug('Setting white level back to 100') # because set at level 10 during brightfield
			mmc.setProperty('SpectraLED', 'White_Level', '100')

		mmc.setProperty('SpectraLED', 'White_Enable', 0) # closes all channels
		
		mmc.setProperty('SpectraLED', channelNames[channel][0] + '_Enable', 1)
		if mmc.getProperty('EmissionWheel', 'Label') != channelNames[channel][1]:
#            mmc.waitForDevice('EmissionWheel') # does that solve the wheel failures ?
#            mmc.waitForDevice('Core') # does that solve the wheel failures ?
			time.sleep(0.1)
#            mmc.waitForSystem() # does that solve the wheel failures ?

			try:
				mmc.setProperty('EmissionWheel', 'Label', channelNames[channel][1])
			except Exception, e:
				logger.error('### EMISSIONWHEEL ERROR ###')
				logger.error(e)
				time.sleep(1)
				try:
					mmc.setProperty('EmissionWheel', 'Label', channelNames[channel][1])
				except Exception, e:
					logger.error('###-### EMISSIONWHEEL ERROR LEVEL 2 ###-###')
					logger.error(e)
					time.sleep(1)
					mmc.setProperty('EmissionWheel', 'Label', channelNames[channel][1])

		mmc.setProperty('SpectraLED', 'YG_Filter', 1) # later in case some delay is needed ...
		
	logger.debug('Channel set to ' + str(channel))

def getChannel():
	if mic == 'Z2':
		return channelNames[microscopeChannelNames.index(mmc.getProperty('ZeissReflectorTurret', 'Label'))]
	elif mic == 'Leica':
		return channelNames.keys()[channelNames.values().index([mmc.getProperty('FastFilterWheelEX', 'Label'), mmc.getProperty('FastFilterWheelEM', 'Label')])]           
	elif mic == 'Nikon':
		if mmc.getProperty('Spectra', 'Violet_Enable') == '1':
			return 'dapi'
		elif mmc.getProperty('Spectra', 'Cyan_Enable') == '1':
			return 488
		elif mmc.getProperty('Spectra', 'Green_Enable') == '1':
			if mmc.getProperty('TIFilterBlock1', 'Label') == '3-FRAP':
				return 'brightfield'
			else:
				return 546
		elif mmc.getProperty('Spectra', 'Red_Enable') == '1':
			return 647
		else:
			return None
	elif mic == 'NikonPeter':
		if mmc.getProperty('SpectraLED', 'White_Enable') == '1':
			return 'brightfield'
		elif mmc.getProperty('SpectraLED', 'Violet_Enable') == '1':
			return 'dapi'
		elif mmc.getProperty('SpectraLED', 'Cyan_Enable') == '1':
			return 488
		elif mmc.getProperty('SpectraLED', 'Green_Enable') == '1':
			return 546
		elif mmc.getProperty('SpectraLED', 'Red_Enable') == '1':
			return 647
		else:
			return None
		
	else:
		return None

def setExposure(exposure):
	mmc.setExposure(exposure)
	logger.debug('Exposure set to ' + str(exposure) + ' ms')

def takeImage(sectionIndex, tileId, folder, name = ''):
	# global savingQueue
	mmc.snapImage()
	channel = getChannel()
	logger.debug('Image taken. Section: ' + str(sectionIndex) + '; channel: ' + str(channel) + '; tileId: ' + str(tileId) + '; name: ' + str(name))
	im = mmc.getImage()
	savingQueue.put([sectionIndex, channel, tileId, folder, im, name])

def getImageSize():
	return np.array([mmc.getImageWidth(), mmc.getImageHeight()])
	
def getMicState(saveName = 'micState'):
	d = {}
	for device in mmc.getLoadedDevices():
		d[device] = {}
		for property in mmc.getDevicePropertyNames(device):
			d[device][property] = {}
			d[device][property]['allowedValues'] = mmc.getAllowedPropertyValues(device, property)
			try:
				d[device][property]['currentValue'] = mmc.getProperty(device, property)
			except Exception, e:
				d[device][property]['currentValue'] = e
	print 'Microscope State Dictionary + \n', json.dumps(d, indent=4, sort_keys=True)
	with open(os.path.join(folderSave, saveName + '.txt'), 'w') as f:
		json.dump(d, f)       
	with open(os.path.join(folderSave, saveName + '_humanReadable.txt'), 'w') as f:
		f.write(json.dumps(d, indent=4, sort_keys=True))
	
	return d

def loadLowResLandmark():
	wafer.targetLowResLandmarks = readPoints(os.path.join(os.path.normpath(wafer.pipelineFolder), 'target_lowres_landmarks.txt'))
	wafer.targetLowResLandmarks = wafer.targetLowResLandmarks.T
	wafer.save()

def loadHighResLandmark():
	wafer.targetHighResLandmarks = readPoints(os.path.join(os.path.normpath(wafer.pipelineFolder), 'target_highres_landmarks.txt'))
	wafer.targetHighResLandmarks = wafer.targetHighResLandmarks.T
	wafer.save()
	wafer.createSections()
	wafer.save()  

def resetNikonStage():
	print mmc.getProperty('LudlController', 'Reset')
	mmc.setProperty('LudlController', 'Reset', 'Reset')
	print mmc.getProperty('LudlController', 'Reset')
	time.sleep(10)
	logger.debug('Stage controller has been reset')

def runImQualityFromFolder(folder):
	command = r'python D:\Images\Templier\pyimagequalityranking\pyimq\bin\main.py --mode=directory --working-directory=' + folder
	# command = r'python D:\Images\Templier\pyimagequalityranking\pyimq\bin\main.py --mode=directory --mode=analyze --mode=plot --result=fpw --working-directory=' + autofocusTestsFolder

	result = subprocess.call(command, shell=True)

def readImQuality(path): # read the .csv output from the pyimagequality library
	with open(path, 'r') as f:
		lines = f.readlines()
		val = []
		for line in lines:
			val.append(line.split(','))
	return val      

##########################
### Imaging parameters ###
class ImagingParameters(object):
	def __init__(self, *args):
		self.channels = args[0]
		self.tileGrid = np.array(args[1]) # 3x4 mosaic for example ...
		self.overlap_pct = args[2]
		self.objective = args[3]
		
		self.tileSize = np.array(magnificationImageSizes[self.objective][1:])
		self.pixelSize = 1/2. * (self.tileSize[0]/float(imageSize_px[0]) + self.tileSize[1]/float(imageSize_px[1])) # averaging on x and y to be more precise ?
		self.mosaicSize = np.array([0, 0]) # just initializing and showing how it looks
		self.ids = [] # indices of the successive tiles [[0,0],[1,0],...,[n,n]]
		self.layoutFigurePath = os.path.join(folderSave, 'mosaicLayout.png')
		self.templateTileCoordinates = self.getTemplateTileCoordinates(self.mosaicSize, self.tileSize, self.tileGrid, self.overlap_pct, self.ids, self.layoutFigurePath)

		self.tileGridMag = np.array(args[4]) # 3x4 mosaic for example ...
		self.overlap_pctMag = args[5]
		self.objectiveMag = args[6]
		self.tileSizeMag = np.array(magnificationImageSizes[self.objectiveMag][1:])
		self.pixelSizeMag = 1/2. * (self.tileSizeMag[0]/float(imageSize_px[0]) + self.tileSizeMag[1]/float(imageSize_px[1])) # averaging on x and y to be more precise ?
		self.mosaicSizeMag = np.array([0, 0]) # just initializing and showing how it looks
		self.idsMag = [] # indices of the successive tiles [[0,0],[1,0],...,[n,n]]
		self.layoutFigurePathMag = os.path.join(folderSave, 'mosaicLayoutMag.png')
		self.templateTileCoordinatesMag = self.getTemplateTileCoordinates(self.mosaicSizeMag, self.tileSizeMag, self.tileGridMag, self.overlap_pctMag, self.idsMag, self.layoutFigurePathMag)
		
		logger.debug('Imaging parameters initialized')
		
	def getTemplateTileCoordinates(self, mosaicSize, tileSize, tileGrid, overlap_pct, ids, layoutFigurePath):
		fig = plt.figure() # producing a figure of the mosaic and autofocus locations
		ax = fig.add_subplot(111)     
		
		# compute and plot mosaic size
		mosaicSize = tileSize * tileGrid - (tileGrid - 1) * (overlap_pct/100. * tileSize)
		logger.debug('The size of the mosaic is ' + str(mosaicSize[0] * 1e6) + ' um x ' + str(mosaicSize[1] * 1e6) + ' um')
		p = patches.Rectangle((-mosaicSize[0]/2., -mosaicSize[1]/2.), mosaicSize[0], mosaicSize[1], fill=False, clip_on=False, color = 'blue', linewidth = 3)
		ax.add_patch(p)

		# compute tile locations starting from the first on the top left (which is actually top right in the Merlin ...)
		topLeftCenter = (- mosaicSize + tileSize)/2.

		tilesCoordinates = []
		for idY in range(tileGrid[1]):
			for idX in range(tileGrid[0]):
				if mic == 'Leica': # warning: leica stage inverted on x-axis
					# id = np.array([tileGrid[0] - 1 - idX, idY])
					id = np.array([idX, idY])
				else:
					id = np.array([idX, idY])
				ids.append(id)
				tileCoordinates = (topLeftCenter + id * (1-overlap_pct/100.) * tileSize)
				tilesCoordinates.append(tileCoordinates)
				plt.plot(tileCoordinates[0], tileCoordinates[1], 'ro')
				p = patches.Rectangle((tileCoordinates[0] - tileSize[0]/2. , tileCoordinates[1] - tileSize[1]/2.), tileSize[0], tileSize[1], fill=False, clip_on=False, color = 'red')
				ax.add_patch(p)

		tilesCoordinates = np.array(tilesCoordinates)

		# compute autofocus locations (actually not used)
		autofocusCoordinates = mosaicSize/2. * (1 + self.autofocusOffsetFactor) * np.array([ [-1 , -1], [1, -1], [-1, 1], [1, 1]]) # 4 points focus: 
	
		# plot autofocus locations
		for point in autofocusCoordinates:
			plt.plot(point[0], point[1], 'bo')
		
		plt.savefig(layoutFigurePath)
		return tilesCoordinates, autofocusCoordinates

#################################
### Wafer and Section classes ###
class Wafer(object):
	def __init__(self, *args):
		self.name = args[0]
		self.ip = args[1] # ImagingParameters
		self.sections = []
		self.magSections = []
		self.folderWaferSave = mkdir_p(os.path.join(folderSave, self.name))
		self.waferPath = os.path.join(self.folderWaferSave, 'Wafer_' + self.name)
		shutil.copy(ip.layoutFigurePath, self.folderWaferSave)
		logger.info('Wafer ' + self.name + ' initiated.')
		self.startingTime = -1
		self.finishingTime = -1
		self.targetLowResLandmarks = []
		self.targetHighResLandmarks = []
		self.targetMagFocus = []
		self.timeEstimate = 0

	def createSections(self):
		currentZ = getZ()
		if hasattr(self, 'targetHighResLandmarks'):
			if len(self.targetHighResLandmarks) == len(self.sourceLandmarks.T): # all target landmarks have been identified
				# creating targetTissuesCoordinates
				self.targetTissues = []
				self.targetTissuesCoordinates = []
				for sourceSectionTissueCoordinates in self.sourceSectionsTissueCoordinates:
					if hasattr(self, 'sourceROIDescription'): # transform the sourceRoi to the ROI in the sourceSection using the transform sourceSectionRoi -> sourceSectionCoordinates
						sourceSectionTissueCoordinates = affineT(np.array(self.sourceROIDescription[0]).T, np.array(sourceSectionTissueCoordinates).T, np.array(self.sourceROIDescription[1]).T).T
					targetTissueCoordinates = affineT(self.sourceLandmarks, np.array(self.targetHighResLandmarks).T[:2], np.array(sourceSectionTissueCoordinates).T)
					self.targetTissuesCoordinates.append(targetTissueCoordinates)
					targetTissue = getCenter(targetTissueCoordinates)
					try:
						targetTissueCenterZ = focusThePoints(np.array(self.targetHighResLandmarks).T, np.array([targetTissue]).T)[-1][0]
					except Exception, e:
						targetTissueCenterZ = currentZ
					self.targetTissues.append([targetTissue[0], targetTissue[1], targetTissueCenterZ])
					
					
				self.targetMagsCoordinates = []
				self.targetMagCenters = []
				for sourceSectionMagCoordinates in self.sourceSectionsMagCoordinates:
					targetMagCoordinates = affineT(self.sourceLandmarks, np.array(self.targetHighResLandmarks).T[:2], np.array(sourceSectionMagCoordinates).T)
					self.targetMagsCoordinates.append(targetMagCoordinates)
					targetMagCenter = getCenter(targetMagCoordinates)
					try:
						targetMagCenterZ = focusThePoints(np.array(self.targetHighResLandmarks).T, np.array([targetMagCenter]).T)[-1][0]
					except:
						targetMagCenterZ = currentZ

					self.targetMagCenters.append([targetMagCenter[0], targetMagCenter[1], targetMagCenterZ])
					
			# if hasattr(self, 'sourceMagDescription'):
				# self.targetMagCoordinates = []
				# for targetTissueCoordinates in self.targetTissuesCoordinates:
					# magCoord = affineT(np.array(self.sourceMagDescription[0]).T, targetTissueCoordinates, np.array(self.sourceMagDescription[1]).T )
					# self.targetMagCoordinates.append(magCoord)
					# self.targetMagCenters.append(getCenter(magCoord))
					
				###### Currently wrong, should simply copy the paragraph above
				# # self.targetROICoordinates = []
				# # self.targetROICenters = []
				# # for targetTissueCoordinates in self.targetTissuesCoordinates: # self.targetTissuesCoordinates seems to have been created earlier
					# # if hasattr(self, 'sourceROIDescription'):  # if no sourceRoiDescription provided, then simply use the center of the tissue sections as center of the ROI
						# # ROICoord = affineT(np.array(self.sourceROIDescription[0]).T, targetTissueCoordinates, np.array(self.sourceROIDescription[1]).T )
					# # else:
						# # ROICoord = targetTissueCoordinates
					# # self.targetROICoordinates.append(ROICoord)
					# # self.targetROICenters.append(getCenter(ROICoord))
		
		else:
			logger.error('Sections cannot be generated because there are currently no target landmarks')
		
		
	def addCurrentPosition(self):
		section = Section([len(self.sections), getXY(), getZ(), self.ip, self.folderWaferSave])
		self.sections.append(section)
		logger.info('New section number added with current position')

	def save(self):
		f = open(self.waferPath, 'w')
		pickle.dump(self, f)
		f.close()
		logger.debug('Wafer saved in ' + self.waferPath)
		
	def acquire(self, manualFocus = False):
		# write metadata
		imSize = getImageSize()
		metadataPath = os.path.join(self.folderWaferSave, 'LM_Metadata.txt')
		with open(metadataPath, 'w') as f:
			f.write('width = ' + str(imSize[0]) + '\n')
			f.write('height = '+ str(imSize[1]) + '\n')
			f.write('nChannels = '+ str(len(self.ip.channels['imaging']['tissue'])) + '\n')
			f.write('xGrid = ' + str(self.ip.tileGrid[0]) + '\n')
			f.write('yGrid = ' + str(self.ip.tileGrid[1]) + '\n')

			f.write('scaleX = ' + str(self.ip.pixelSize) + '\n')
			f.write('scaleY = ' + str(self.ip.pixelSize) + '\n')
			f.write('channels = [' + ','.join(map(str, self.ip.channels['imaging']['tissue'])) + ']')
	
	
		if (not manualFocus):
			logger.info('Acquire tissue with hardware autofocus')
			self.save()
			logger.info('Starting acquisition of wafer ' + str(self.name))
			self.startingTime = time.time()
			logger.info(str(len(filter(lambda x: x.acquireFinished, self.sections))) + ' sections have been already scanned before this start')
			
			nSectionsAcquired = sum([section.acquireFinished for section in self.sections] ) # for after interruptions
			sectionIndicesToAcquire = range(nSectionsAcquired, len(self.sections), 1)
			
			for currentSessionCounter, id in enumerate(sectionIndicesToAcquire):
				section = self.sections[id]
				logger.info('Starting acquisition of section ' + str(section.index) + ' (' +  str(id) + ') ' + ' in wafer ' + str(self.name) )
				section.acquire(mag = False, manualFocus = manualFocus)
				closeShutter()
				
				#logging some durations
				averageSectionDuration = (time.time()- self.startingTime)/float(currentSessionCounter + 1)
				timeRemaining = (len(self.sections) - (id + 1)) * averageSectionDuration
				logger.info(str(currentSessionCounter + 1) + ' sections have been scanned during this session, with an average of ' + str(round(averageSectionDuration/60., 1)) + ' min/section.' )
				logger.info('Time remaining estimated: ' + durationToPrint(timeRemaining) + ' for ' + str((len(self.sections) - (id + 1))) + ' sections' )
				self.save()
			self.finishingTime = time.time()
			elapsedTime = (self.finishingTime - self.startingTime)
			savingQueue.put('finished') # closing the saverThread
			time.sleep(2 * sleepSaver)
			# saverThread.join() # xxx is it ok ? Is it not going to return before ?
			
			logger.info('The current session for the wafer took ' + durationToPrint(elapsedTime))
			closeShutter()
		else:
			logger.info('Acquire tissue with manual focus')
			mmc.setAutoShutter(False)
			
			sectionsToAcquire = filter(lambda x: (not x.acquireFinished), self.sections) # I could have TSP ordered the sections earlier ...
			
			if len(sectionsToAcquire) == 0:
				logger.info('All sections have already been acquired.')
			else:
				logger.info(str(len(sectionsToAcquire)) + ' sections remaining to be acquired')
				nextTissueSectionToAcquire = filter(lambda x: x.localized, sectionsToAcquire)
				
				if len(nextTissueSectionToAcquire) == 0:
					logger.info('Currently not centered on a section. Moving to a section ...')
					logger.info('Moving to section number ' + str(sectionsToAcquire[0].index) )
					sectionsToAcquire[0].moveToSection()
					# updating the localization flag (false for all except for the current one)
					for tissueSection in wafer.sections:
						tissueSection.localized = False
					sectionsToAcquire[0].localized = True
					logger.info('Section is now localized. Manually adjust the focus then press ManualTissue button again.')
				elif len(nextTissueSectionToAcquire) == 1:
					theNextTissueSectionToAcquire = nextTissueSectionToAcquire[0]
					logger.info('Section ' + str(theNextTissueSectionToAcquire.index) + ' is localized. Acquiring ...')
					theNextTissueSectionToAcquire.acquire(mag = False, manualFocus = manualFocus) # it will update the acquireFinished flag
					logger.info('Section ' + str(theNextTissueSectionToAcquire.index) + ' acquired.')
					
					sectionsToAcquire = filter(lambda x: (not x.acquireFinished), self.sections) # this is the new list
					if len(sectionsToAcquire) == 0:
						logger.info('Good, all tissue sections have been acquired.')
					else:
						nextTissueSectionToAcquire = [sectionsToAcquire[0]] # enlisting for naming consistency
						
						nextTissueSectionToAcquire[0].moveToSection()
						theNextTissueSectionToAcquire.localized = False # delocalize the previous section
						nextTissueSectionToAcquire[0].localized = True
						logger.info('Moved to section ' + str(nextTissueSectionToAcquire[0].index) + '. Manually adjust the focus and press the button again.')
				else:
					logger.error('It cannot be. Go tell Thomas what a bozo he is.')
			
			# closeShutter()
			self.save()
			logger.debug('All tissue sections have been acquired. Looking for focus outliers ...') # outdated as localImaging will be better
			
	def createSectionsFromPipeline(self):
		self.sections = []
		for idSection, targetTissue in enumerate(self.targetTissues): # xxx it should be targetROICenters instead of targetTissues
			section = Section([idSection, [targetTissue[0], targetTissue[1]], targetTissue[2], self.ip, self.folderWaferSave])
			self.sections.append(section)
		logger.debug('Tissue sections have been created')

	def createMagSectionsFromPipeline(self):
		self.magSections = []
		for idSection, targetMagCenter in enumerate(self.targetMagCenters):
			magSection = Section([idSection, [targetMagCenter[0], targetMagCenter[1]], targetMagCenter[2], self.ip, self.folderWaferSave], mag = True) # targetMagCenter[2] comes from the interpolation of the targetMagFocus (manually focused beads)
			self.magSections.append(magSection)
		logger.debug('Mag sections have been created')
	
	def magAcquire(self, manualFocus = False):
		if not manualFocus:
			logger.debug('Starting acquisition of all mag sections with hardware autofocus for order retrieval')
			mmc.setAutoShutter(False)

			magSectionsToAcquire = filter(lambda x: (not x.acquireMagFinished), self.magSections) # I could have TSP ordered the sections earlier ...            
			for id, magSection in enumerate(magSectionsToAcquire): # xxx should add a filter for only sections not acquired
				logger.info('Ordering: moving to section number ' + str(magSection.index) + '(' + str(id) + ')')
				magSection.acquire(mag = True)
				self.save() # to keep track of which sections were scanned
				# closeShutter()
			self.save()
			logger.debug('All mag sections have been acquired.')
		else:
			logger.debug('Acquire mag with manual focus')
			mmc.setAutoShutter(False)
			magSectionsToAcquire = filter(lambda x: (not x.acquireMagFinished), self.magSections) # I could have TSP ordered the sections earlier ...
			
			# I am using 2 flags: localized (is the section currently centered) and acquireMagFinished (has the section already been acquired)
			if len(magSectionsToAcquire) == 0:
				logger.info('All mag sections have already been acquired.')
			else:
				logger.info(str(len(magSectionsToAcquire)) + ' mag sections remaining to be acquired')
				nextMagSectionToAcquire = filter(lambda x: x.localized, magSectionsToAcquire)
				
				if len(nextMagSectionToAcquire) == 0:
					logger.info('Currently not centered on a section. Moving to a section ...')
					logger.info('Moving to section number ' + str(magSectionsToAcquire[0].index) )
					magSectionsToAcquire[0].moveToSection()
					# updating the localization flag (false for all except for the current one)
					for magSection in wafer.magSections:
						magSection.localized = False
					magSectionsToAcquire[0].localized = True
					logger.info('Section is now localized. Manually adjust the focus then press ManualMag button again.')
				elif len(nextMagSectionToAcquire) == 1:
					logger.info('Section ' + str(nextMagSectionToAcquire[0].index) + ' is localized. Acquiring ...')
					nextMagSectionToAcquire[0].acquire(mag = True, HAF = False) # it will update the acquireMagFinished flag
					logger.info('Section ' + str(nextMagSectionToAcquire[0].index) + ' acquired.')
					self.save()
					
					magSectionsToAcquire = filter(lambda x: (not x.acquireMagFinished), self.magSections) # this is the new list
					if len(magSectionsToAcquire) == 0:
						logger.info('Good, all mag sections have been acquired.')
					else:
						nextMagSectionToAcquire = [magSectionsToAcquire[0]] # enlisting for naming consistency
						
						nextMagSectionToAcquire[0].moveToSection()
						nextMagSectionToAcquire[0].localized = True
						logger.info('Moved to section ' + str(nextMagSectionToAcquire[0].index) + '. Manually adjust the focus and press the button again.')
				else:
					logger.error('It cannot be. Go tell Thomas what a bozo he is.')
			
			# closeShutter()
			self.save()
			logger.debug('All mag sections have been acquired.')

class Section(object):
	def __init__(self, args, angle = 0, mag = False):
		self.index = args[0]
		self.center = args[1]
		self.startingZ = args[2] # given by the interpolative plane
		self.ip = args[3] # MosaicParameters
		self.folderWaferSave = args[4]
		self.angle = angle
		self.imagingCoordinates = {}
		if mag:
			self.imagingCoordinates['tiles'] = transformCoordinates(self.ip.templateTileCoordinatesMag[0], self.center, self.angle)
			self.imagingCoordinates['autofocus'] = transformCoordinates(self.ip.templateTileCoordinatesMag[1], self.center, self.angle)
		else:
			self.imagingCoordinates['tiles'] = transformCoordinates(self.ip.templateTileCoordinates[0], self.center, self.angle)
			self.imagingCoordinates['autofocus'] = transformCoordinates(self.ip.templateTileCoordinates[1], self.center, self.angle)
		
		self.focusedPoints = []
		self.acquireStarted = False
		self.acquireFinished = False
		self.acquireMagStarted = False
		self.acquireMagFinished = False
		self.currentZ = self.startingZ
		self.startingTile = 0 # for after interruptions
		self.folderSectionSave = os.path.join(self.folderWaferSave, 'section_' + str(self.index).zfill(4))
		self.startingTime = -1
		self.finishingTime = -1
		self.focusedMagZ = -1 
		self.focusScore = -99
		
		self.localized = False # for manual focus: flag that tells whether the stage is currently in position on that section
		
	def acquire(self, mag = False, manualFocus = False):
		self.startingTime = time.time()
		mkdir_p(self.folderSectionSave)
		logger.info('Scanning: Section ' + str(self.index))
		self.moveToSection()
		setZ(self.startingZ)
		closeShutter()
		if mag:
			self.acquireMagStarted = True
			# self.focusedMagZ, self.focusScore = beadAutofocus() # careful, beadAutofocus is changing the channel
			if not manualFocus:
				self.focusedMagZ = autofocus()
			else:
				self.focusedMagZ = getZ()
			for idTile, point in enumerate(self.imagingCoordinates['tiles']):
				logger.debug('Scanning tile ' + str(idTile))
				setXY(*point[:2])
				if not manualFocus:
					autofocus()
				tileZ = getZ()
#                 openShutter() # not necessary on Nikon
#                 time.sleep(0.1)
				for channel in channels['imaging']['beads']:
					setZ(tileZ + channels[channel][objectiveBeads]['offset']['imaging']['beads'])
					setChannel(channel)
					setExposure(channels[channel][objectiveBeads]['exposure']['imaging']['beads'])
#                     time.sleep(0.1)
					takeImage(self.index, self.ip.idsMag[idTile], self.folderSectionSave, name = 'mag')
					closeShutter()
				closeShutter()
			self.acquireMagFinished = True
				
		else:
			self.acquireStarted = True
			if not manualFocus:
				autofocus()
			self.focusedZ = getZ() # the focus of the section
			for idTile, point in enumerate(self.imagingCoordinates['tiles']):
				logger.debug('Scanning tile ' + str(idTile))
				setXY(*point[:2])
				if not manualFocus:
					autofocus()
				tileFocus = getZ() # the focus of the tile
				openShutter()
				# time.sleep(0.1)
				for channel in channels['imaging']['tissue']:
					setChannel(channel)
					setExposure(channels[channel][objectiveTissue]['exposure']['imaging']['tissue'])
					currentOffset = channels[channel][objectiveTissue]['offset']['imaging']['tissue']
					if currentOffset != 0:
						setZ(tileFocus + currentOffset)
					takeImage(self.index, self.ip.ids[idTile], self.folderSectionSave, name = 'tissue')
					if currentOffset != 0:
						setZ(tileFocus) # ideally would take into account what is the next offset, but annoying and not crucial ...
					
			self.finishingTime = time.time()
			logger.debug('Section ' + str(self.index) + ' acquired. It has taken ' + str((self.finishingTime - self.startingTime)/60.) + ' min.' )      
			self.acquireFinished = True
		logger.debug('Section ' + str(self.index) + ' has been acquired')


	def computeZPlane(self):
		self.focusedPoints = [] # this is needed for after interruptions: the focused points should be cleared
		setChannel('brightfield')
		
		for idPoint, autofocusPosition in enumerate(self.imagingCoordinates['autofocus']):
			logger.debug('Autofocusing of point number ' + str(idPoint) + ' in Tile number ' + str(self.index))
			setXY(autofocusPosition[0], autofocusPosition[1])
			openShutter()
			# # # if (idPoint == 0): # not possible because the filter wheels are not motorized ?
				# # # self.getRoughFocus()
			focusedZ = autofocus()
			self.focusedPoints.append([autofocusPosition[0], autofocusPosition[1], focusedZ])
		closeShutter()
		self.imagingCoordinates['tiles'] = focusThePoints(np.array(self.focusedPoints).T, self.imagingCoordinates['tiles'].T).T
		
		logger.debug('The imaging coordinates will be ' + str(self.imagingCoordinates['tiles']))
		logger.info('Interpolative plane calculated for Section number ' + str(self.index))

	def scanTiles(self):
		tilesToScan = range(self.startingTile, len(self.imagingCoordinates['tiles']), 1) # for restart after interruption
		for idTile in tilesToScan: 
			tileCoordinates = self.imagingCoordinates['tiles'][idTile]
			logger.debug('Scanning: Section ' + str(self.index) + ' - Tile ' + str(idTile) )
			setXY(tileCoordinates[0], tileCoordinates[1])
			if mic == 'Leica':
				mmc.fullfocus()
				
			for channel in channels['imaging']['tissue']:
				setChannel(channel)
				setExposure(self.ip.channels[channel][objectiveTissue]['exposure']['imaging']['tissue'])

				currentZ = getZ() # xxx this is wrong no !?
				setZ(currentZ + self.ip.channels[channel][objectiveTissue]['offset'])
				
				logger.info('Scanning: Section ' + str(self.index) + ' - Tile ' + str(idTile) + ' - Channel ' + str(channel))
				if idTile == tilesToScan[0]:
					openShutter()
				takeImage(self.index, self.ip.ids[idTile], self.folderSectionSave)
				self.startingTile = idTile + 1
				if idTile == tilesToScan[-1]:
					closeShutter()
		
	def moveToSection(self):
		setXY(self.center[0], self.center[1])
		logger.debug('Moved to center of section ' + str(self.index))

#################################
# Some focus function utils		
#################################
def testAutofocus():
	allAutofocus = []
	focusedZ = getZ()
	outOfFocusFactor = 30
	
	## for i in range(10):
	#     # outZ = focusedZ + outOfFocusFactor * random.random() * random.choice([1, -1])    
	#     # setZ(outZ)
	#     # autofocus()
	#     # allAutofocus.append([getZ(), outZ])
	#     
	## meanOffset = focusedZ - np.mean([f[0] for f in allAutofocus])
	
	for i in range(5):
		z = []
		elapsedTime = []
		outZ = focusedZ + outOfFocusFactor * random.random() * random.choice([1, -1])
		setZ(outZ)
		startingTime = time.time()
	#     time.sleep(0.5)
		autofocus()
		elapsedTime.append(int(time.time() - startingTime))
		z.append(getZ())
	#      if elapsedTime[0] > 4:
	#         setZ(z[0] + 15)
	#         startingTime = time.time()
	#         time.sleep(0.5)
	#         autofocus()
	#         elapsedTime.append(int(time.time() - startingTime))
	#         z.append(getZ())
	#         if elapsedTime[1] > 4:
	#             setZ(z[0] - 15)
	#             startingTime = time.time()
	#             time.sleep(0.5)
	#             autofocus()
	#             elapsedTime.append(int(time.time() - startingTime))
	#             z.append(getZ())
		allAutofocus.append([z, elapsedTime, outZ])
		# take and save image
	
		setChannel(488)
		setExposure(channels[488][objectiveBeads]['exposure']['imaging']['beads'])
		mmc.snapImage()
		im = mmc.getImage()
		result = PIL.Image.fromarray((im).astype(np.uint16))
		result.save(os.path.join(folderSave, 'autofocusTest_488_' + str(i).zfill(3)) + '.tif')
		
		setChannel(546)
		setExposure(channels[546][objectiveBeads]['exposure']['imaging']['beads'])
#         setZ(getZ() + channels[546][objectiveBeads]['offset']['imaging']['beads'])
		setZ(getZ() + channels[546][objectiveTissue]['offset']['imaging']['tissue'])
		mmc.snapImage()
		im = mmc.getImage()
		result = PIL.Image.fromarray((im).astype(np.uint16))
		result.save(os.path.join(folderSave, 'autofocusTest_546_' + str(i).zfill(3)) + '.tif')

		setChannel('brightfield')
		setExposure(channels['brightfield'][objectiveBeads]['exposure']['imaging']['beads'])
		setZ(getZ() + channels['brightfield'][objectiveTissue]['offset']['imaging']['tissue'])
		mmc.snapImage()
		im = mmc.getImage()
		result = PIL.Image.fromarray((im).astype(np.uint16))
		result.save(os.path.join(folderSave, 'autofocusTest_BF_' + str(i).zfill(3)) + '.tif')
		
	closeShutter()
	meanOffset = focusedZ - np.mean([f[0][-1] for f in allAutofocus])
	return allAutofocus

def findRedFocus():
	altitudes = np.arange( 1494, 1496 , 0.2)
	for altitude in altitudes:
		logger.info(str(altitude))
		setExposure(channels[647][objectiveTissue]['exposure']['imaging']['tissue'])
		setZ(altitude)
		setChannel(647)
		mmc.snapImage()
		closeShutter()

		im = mmc.getImage()
		result = PIL.Image.fromarray((im).astype(np.uint16))
		result.save(os.path.join(folderSave, 'findFocus_647_' + str(altitude).zfill(6)) + '.tif')

		setExposure(channels['brightfield'][objectiveTissue]['exposure']['imaging']['tissue'])
		setZ(altitude)
		setChannel('brightfield')
		mmc.snapImage()
		closeShutter()
		im = mmc.getImage()
		result = PIL.Image.fromarray((im).astype(np.uint16))
		result.save(os.path.join(folderSave, 'findFocus_brightfield_' + str(altitude).zfill(6)) + '.tif')  
	closeShutter()

#################################
# Some benchmarking of microscope components		
#################################
def testNikonBodyFilterSpeed():
	startingTime = time.time()
	
	for i in range(20):
		mmc.setProperty('TIFilterBlock1', 'Label', '3-FRAP')
		mmc.waitForDevice('TIFilterBlock1')
		mmc.setProperty('TIFilterBlock1', 'Label', '2-Quad')
		mmc.waitForDevice('TIFilterBlock1')
	
	print 'average cycle ', int(time.time() - startingTime)/float(i)
	
def testNikonPeterFilterWheel():
	channels = ['brightfield','dapi', 488, 546, 647]
	for channel in channels:
		setChannel(channel)	
	
	
if __name__ == '__main__':

	#######################
	### Initializations ###
	
	# starting the thread that saves the images that are inserted into the queue 
	savingQueue = Queue()
	saverThread = threading.Thread(target= saver, args= (savingQueue,) ) # the comma is important to make it a tuple
	saverThread.start()
	
	# initializing logger
	logPath = os.path.join(folderSave, 'log_' + waferName + '.txt')
	logger = initLogger(logPath)
	logger.info('Logger started.')
	logger.info('Wafer name : ' + str(waferName))
	logger.info('Saving path : ' + str(folderSave))
	
	# Micromanager core
	mmc = MMCorePy.CMMCore()  
	logger.info('Micromanager version and API version: ' + str(mmc.getVersionInfo()) + ' , ' + str(mmc.getAPIVersionInfo()))
	mmc.enableStderrLog(False)
	mmc.enableDebugLog(True)
	mmc.setPrimaryLogFile(os.path.join(folderSave, 'logMMC_' + waferName + '.txt'))
	mmc.loadSystemConfiguration(mmConfigFile)

	# all initial properties from a specific microscope
	for initialProperty in initialProperties:
		mmc.setProperty(initialProperty[0], initialProperty[1], initialProperty[2])

	# Initializing main microscope devices
	camera = mmc.getCameraDevice()
	shutter = mmc.getShutterDevice()
	zStage = mmc.getFocusDevice()
	stage = mmc.getXYStageDevice()
	imageSize_px = getImageSize()
	try:
		autofocusDevice = mmc.getAutoFocusDevice()
	except Exception, e:
		logger.error('Error: hardware autofocus could not be loaded')
	
	# deactivate autoshutter
	mmc.setAutoShutter(False)
	
	# what is the current objective
	currentObjectiveName = mmc.getProperty(objectiveProperties[0], objectiveProperties[1])

	if '20x' in currentObjectiveName:
		currentObjectiveNumber = 20
	elif '60x' in currentObjectiveName: # oil
		currentObjectiveNumber = 63

	### Create mosaic parameters ###
	overlap_pct = np.array([overlap, overlap])
	overlap_pctMag = np.array([overlapMag, overlapMag])
	ip = ImagingParameters(channels, tileGrid, overlap_pct, objectiveTissue, tileGridMag, overlap_pctMag, objectiveBeads)
	
	#############
	### Start ###
	root = Toplevel()
	wafer = Wafer(waferName, ip)
	app = App(root)
	root.mainloop()