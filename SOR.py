from __future__ import with_statement
import os, time, threading, pickle, shutil
from collections import Counter

import java
from java.util import ArrayList, HashSet, Stack
from java.util.concurrent.atomic import AtomicInteger
from java.awt import Polygon, Rectangle, Color, Point
from java.awt.geom import Area, AffineTransform
from java.lang import Math, Runtime, Exception # to catch java exceptions
from java.lang import Float
from java.lang.reflect import Array
from java.lang.Math import hypot, sqrt, atan2, PI, abs
from Jama import Matrix
import jarray
from jarray import zeros, array

import fijiCommon as fc

import ij
from ij import IJ, ImagePlus
from ij.gui import OvalRoi
from ij.gui import Roi
from ij.plugin import ImageCalculator
from ij.plugin.frame import RoiManager
from ij.plugin.filter import MaximumFinder
from ij.plugin.filter import ParticleAnalyzer
from ij.process import FloatProcessor
from ij.measure import Measurements

from java.lang import Double

from mpicbg.models import RigidModel2D, AffineModel2D, PointMatch, NotEnoughDataPointsException
from mpicbg.ij import SIFT, FeatureTransform
from mpicbg.ij.plugin import NormalizeLocalContrast
from mpicbg.imagefeatures import FloatArray2DSIFT
from mpicbg.ij.util import Util
from mpicbg.imglib.image import ImagePlusAdapter
from mpicbg.imglib.algorithm.scalespace import DifferenceOfGaussianPeak
from mpicbg.imglib.algorithm.scalespace.DifferenceOfGaussian import SpecialPoint
from mpicbg.imglib.algorithm.correlation import CrossCorrelation
from mpicbg.imglib.type.numeric.integer import IntType

from fiji.plugin.trackmate.detection import LogDetector


from net.imglib2.img.display.imagej import ImageJFunctions
# from net.imglib2 import Point
import net
from net.imglib2.algorithm.region.hypersphere import HyperSphere
from net.imglib2.img.imageplus import ImagePlusImgFactory
from net.imglib2.type.numeric.integer import UnsignedByteType

from ini.trakem2 import Project
from ini.trakem2.display import AreaList, Patch, Display
from ini.trakem2.imaging import StitchingTEM, Blending
from ini.trakem2.imaging.StitchingTEM import PhaseCorrelationParam

from process import Matching
from plugin import DescriptorParameters

from bunwarpj.bUnwarpJ_ import computeTransformationBatch, elasticTransformImageMacro
from bunwarpj import Param

####################
# I/O operations and geometric utils
####################
def matToList(m):
	l = []
	for column in m:
		l.append(list(column))
	return l

def pickleSave(a, path):
	f = open(path, 'w')
	pickle.dump(a, f)
	f.close()

def pickleLoad(path):
	f = open(path, 'r')
	a = pickle.load(f)
	f.close()
	return a

def inferTilingFromNames(names):
	n = len(names)
	maxX = max([getFieldFromName(os.path.basename(name), 'X') for name in names]) + 1
	return maxX, int(n/maxX)

def getFieldFromName(name, field):
	# patch_015_025_Edges.tif
	result = None
	if field == 'X':
		result = os.path.splitext(name)[0].split('_')[2]
		print name, field, os.path.splitext(name)[0].split('_')
	elif field == 'Y':
		result = os.path.splitext(name)[0].split('_')[3]
	print field, result
	return int(result)

def readPoints(path):
	points = []
	with open(path, 'r') as f:
		lines = f.readlines()
		for point in lines:
			points.append(map(lambda x: int(float(x)), point.split('\t') ))
	return points

def getLength(p1,p2):
	return sqrt((p1[0]-p2[0]) * (p1[0]-p2[0]) + (p1[1]-p2[1]) * (p1[1]-p2[1]))

def getAngle(line):
	diff = [line[0] - line[2], line[1] - line[3]]
	theta = atan2(diff[1], diff[0])
	return theta

def getEdgeLengths(corners):
	edgeLengths = []
	nCorners = len(corners)
	for i in range(nCorners):
		if (None in corners[i]) or (None in corners[(i+1) % nCorners]) :
			edgeLengths.append(None)
		else:
			edgeLengths.append(getLength(corners[i], corners[(i+1) % nCorners]))
	return edgeLengths

def getSectionsAngles(allSectionsCoordinates):
	angles = []
	for sectionCorners in allSectionsCoordinates:
		angle = getAngle([sectionCorners[0][0], sectionCorners[0][1], sectionCorners[1][0], sectionCorners[1][1]])
		angles.append(angle)
	return angles

def getNewMosaicSize():
	imPath = os.path.join(preprocessedFolder, filter(lambda x: os.path.splitext(x)[1] == '.tif', os.listdir(preprocessedFolder))[0])
	im = IJ.openImage(imPath)
	im.close()
	return im.getWidth(), im.getHeight()

def pointsToDogs(points):
	dogs = ArrayList()
	for point in points:
		dogs.add(DifferenceOfGaussianPeak( [int(point[0]), int(point[1]) ] , IntType(255), SpecialPoint.MAX ))
	return dogs

def pointListToList(pointList): # [[1,2],[5,8]] to [1,2,5,8]
	l = array(2 * len(pointList) * [0], 'd')
	for id, point in enumerate(pointList):
		l[2*id] = point[0]
		l[2*id+1] = point[1]
	return l

def listToPointList(l): # [1,2,5,8] to [[1,2],[5,8]]
	pointList = []
	for i in range(len(l)/2):
		pointList.append([l[2*i], l[2*i+1]])
	return pointList

def cropPeaks(peaks, cropParams):
	croppedPeaks = filter(lambda p: p[0]>cropParams[0] and p[0]<cropParams[1] and p[1]>cropParams[2] and p[1]<cropParams[3], peaks)
	IJ.log('cropping: ' + str(len(peaks)) + '** ' + str(len(croppedPeaks)))
	return croppedPeaks

def barycenter(points):
	xSum = 0
	ySum = 0
	for i,point in enumerate(points):
		xSum = xSum + point[0]
		ySum = ySum + point[1]
	x = int(xSum/float(i+1))
	y = int(ySum/float(i+1))
	return x,y

def crop(rectA, rectB):
	'''
	crops a rectB (width, height) in the center of rectA (width, height) and returns bounding box coordinates
	'''
	x1 = (rectA[0] - rectB[0])/2.
	x2 = (rectA[0] + rectB[0])/2.
	y1 = (rectA[1] - rectB[1])/2.
	y2 = (rectA[1] + rectB[1])/2.
	return [x1, x2, y1, y2]

def drawSphere(randomAccessible, center, r, val):
	smallSphere = HyperSphere( randomAccessible, center, r )
	for value in smallSphere:
		value.setReal(val)

def createBlobs(w, h, points):
	imDim = [w, h, 1]
	img    = ImagePlusImgFactory().create(imDim, UnsignedByteType())
	imp = img.getImagePlus()
	r = 3
	for a,b,c in points: # c is the size
		if (a-r > 0) and (a+r < w) and (b-r > 0) and (b+r < h):
			drawSphere(img, net.imglib2.Point([int(a),int(b)]), r, 255)
	return imp

def convertTo8Bit(atomicI, imagePaths, newImagePaths, minMax, downFactor = 1, vFlip = False, hFlip = False):
	while atomicI.get() < len(imagePaths):
		k = atomicI.getAndIncrement()
		if (k < len(imagePaths)):
			imagePath = imagePaths[k]
			newImagePath = newImagePaths[k]
			im = IJ.openImage(imagePath)
			IJ.setMinAndMax(im, minMax[0], minMax[1])
			IJ.run(im, '8-bit', '')
			if downFactor != 1:
				im = fc.resize(im, float(1/float(downFactor)))
			if 'rightfiel' in os.path.basename(imagePath):
				im = fc.localContrast(im)
			if vFlip:
				IJ.run(im, 'Flip Vertically', '')
			if hFlip:
				IJ.run(im, 'Flip Horizontally', '')
			IJ.save(im, newImagePath)
			IJ.log(str(k) + ' of ' + str(len(imagePaths)) + ' processed')
			im.close()


####################
# TSP operations
####################
def initMat(n, initValue = 0):
	a = Array.newInstance(java.lang.Float,[n, n])
	for i in range(n):
		for j in range(n):
			a[i][j] = initValue
	return a

def copySquareMat(m):
	a = initMat(len(m))
	for x, col in enumerate(m):
		for y, val in enumerate(col):
			a[x][y] = m[x][y]
	return a

def matSum(a,b):
	width = len(a)
	height = len(a[0])
	for x in range(width):
		for y in range(height):
			a[x][y] = a[x][y] + b[x][y]
	return a

def matAverage(ms):
	width = len(ms[0])
	a = initMat(width)
	for x in range(width):
		for y in range(width):
			l = [m[x][y] for m in ms]
			if sum(l) > 50000:
				a[x][y] = min(l)
				IJ.log('discrepancy ' + str(l) )
			else:
				a[x][y] = sum(l)/float(len(ms))
	return a

def orderFromMat(mat, rootFolder, solutionName = ''):
	tsplibPath = os.path.join(rootFolder, 'TSPMat.tsp')
	saveMatToTSPLIB(mat, tsplibPath)
	solutionPath = os.path.join(rootFolder, 'solution_' + solutionName + '.txt')
	pluginFolder = IJ.getDirectory('plugins')
	concordePath = os.path.join(pluginFolder, 'linkern.exe')
	IJ.log('concordePath is there: ' + str(os.path.isfile(concordePath)))
	# subprocess.call([concordePath , '-o', solutionPath , tsplibPath]) # I could specify more iterations

	# use os.system because subprocess currently broken
	# command = concordePath + ' ' + tsplibPath + ' -o ' + solutionPath
	command = concordePath + ' -o ' + solutionPath + ' ' + tsplibPath
	IJ.log('Command: ' + str(command))
	# os.system(command)
	# os.popen(command)
	
	process = Runtime.getRuntime().exec(command)
	# output = process.getInputStream()
	
	while not os.path.isfile(solutionPath):
		time.sleep(1)
		IJ.log('Computing TSP solution ...')
	time.sleep(1)
	
	with open(solutionPath, 'r') as f:
		lines = f.readlines()[1:]
	order = []
	for line in lines:
		order.append(int(line.split(' ')[0]))

	# remove the dummy city 0 and apply a -1 offset
	order.remove(0)
	for id, o in enumerate(order):
		order[id] = o-1

	# logging some info
	# IJ.log('The order is ' + str(order))
	costs = []
	for id, o in enumerate(order[:-1]):
		o1, o2 = sorted([order[id], order[id+1]]) # sorting because [8, 6] is not in the matrix, but [6,8] is
		cost = mat[o1][o2]
		IJ.log( 'order cost ' + str(order[id]) + '_' +  str(order[id+1]) + '_' + str(cost))
		costs.append(cost)
		# xxx if there are jumps, then they must be visible in the costs
	totalCost = sum(costs)
	IJ.log('The total cost of the retrieved order is ' + str(totalCost))
	IJ.log('The total cost of the incremental order is ' + str( sum( [ mat[t][t+1] for t in range(len(order) - 1)] )) )
	return order, costs

def saveMatToTSPLIB(mat, path):
	# the matrix is a distance matrix
	IJ.log('Entering saveMatToTSPLIB')
	n = len(mat)
	f = open(path, 'w')
	f.write('NAME: Section_Similarity_Data' + '\n')
	f.write('TYPE: TSP' + '\n')
	f.write('DIMENSION: ' + str(n + 1) + '\n')
	f.write('EDGE_WEIGHT_TYPE: EXPLICIT' + '\n')
	f.write('EDGE_WEIGHT_FORMAT: UPPER_ROW' + '\n')
	f.write('NODE_COORD_TYPE: NO_COORDS' + '\n')
	f.write('DISPLAY_DATA_TYPE: NO_DISPLAY' + '\n')
	f.write('EDGE_WEIGHT_SECTION' + '\n')

	distances = [0]*n #dummy city
	for i in range(n):
		for j in range(i+1, n, 1):
			distance = mat[i][j]
			distances.append(int(float(distance*1000)))

	for id, distance in enumerate(distances):
		f.write(str(distance))
		if (id + 1)%10 == 0:
			f.write('\n')
		else:
			f.write(' ')
	f.write('EOF' + '\n')
	f.close()

# Jama matrix operations
def pythonToJamaMatrix(m):
	a = Matrix(jarray.array([[0]*len(m) for id in range(len(m))], java.lang.Class.forName("[D")))
	for x, col in enumerate(m):
		for y, val in enumerate(col):
			a.set(x, y, m[x][y])
	return a

def perm(order): # permutation matrix
	n = len(order)
	rows = []
	for idRow in range(n):
		row = ([float(0)]*n)
		row[order.index(idRow)] = 1
		rows.append(row)
	# print '[row for row in rows]',[row for row in rows]
	m =  Matrix(jarray.array([row for row in rows], java.lang.Class.forName("[D")))
	# print 'permutation', m.getArrayCopy()[100], (m.getArrayCopy()[100]).index(1)
	return m

def reorderM(m, order):
	pm = perm(order)
	# pm = pm.transpose()

	# print 'inverse', pm.inverse().getArrayCopy()[0], (pm.inverse().getArrayCopy()[0]).index(1)
	# print '((pm.inverse()).times(m)).times(pm)', (((pm.inverse().times(m)).times(pm)).getArrayCopy())[0]
	return (pm.inverse().times(m)).times(pm)
	# return pm.times(m).times(pm.inverse())

####################
# Computing similarity
####################

def imToPeak(im, x, y, stdev, center, stretch, medianRadius, threshold = []):
	im = fc.normLocalContrast(im, x, y, stdev, center, stretch)
	
	# IJ.run(im, 'Invert', '')
	if threshold:
		im = fc.minMax(im, threshold[0], threshold[1])
	# IJ.run(im, 'Median...', 'radius=' + str(medianRadius)) # this median might a big effect actually ...
	# IJ.run(im, 'Invert', '') # invert should be run after the trakem Rotation to leave the beads bright and the background black
	return im

def preprocessImToPeak(imPaths, atomIndex, x, y, stdev, center, stretch, medianRadius, threshold):
	while atomIndex.get() < len(imPaths):
		index = atomIndex.getAndIncrement()
		if index < len(imPaths):
			IJ.log('Preprocessing tile ' + str(index))
			imPath = imPaths[index]
			im = IJ.openImage(imPath)
			im = imToPeak(im, x, y, stdev, center, stretch, medianRadius, threshold)
			# imInfo = im.getOriginalFileInfo()
			# IJ.save(im, os.path.join(imInfo.directory, imInfo.fileName))
			IJ.save(im, imPath)
			im.close()
			
def getPeaks(atom, paths):
	while atom.get() < len(paths) :
		k = atom.getAndIncrement()
		if k < len(paths):
			im = IJ.openImage(paths[k])
			ip = im.getProcessor()
			points = []
			MF = MaximumFinder()
			if ('A7' in inputFolder) or ('B6' in inputFolder) or ('C1' in inputFolder):
				poly = MF.getMaxima(ip, maximaNoiseTolerance, True) # noise tolerance, excludeOnEdges
				for x,y in zip(poly.xpoints, poly.ypoints):

					# not only append the location but also the size
					theMax = im.getPixel(x, y)[0]
					threshold = peakDecay * theMax
					
					intenseDisk = True
					d = 1
					while intenseDisk:# grow concentric disks and check the mean instensity relative to the intensity of the peak
						disk = OvalRoi(int(round(x-d/2.)), int(round(y-d/2.)), d, d)
						im.setRoi(disk)
						theMean = im.getStatistics(Measurements.MEAN).mean
						if theMean > threshold:
							d = d + 1
						else:
							intenseDisk = False
						im.killRoi()

					points.append([x, y, d])

					
			elif 'BIB' in inputFolder:
				############# /!\ Warning for BIB Manual to remove /!\ ###############
				IJ.run(im, 'Invert', '')
				im = fc.normLocalContrast(im, 10, 10, 3, True, True)
				IJ.run(im, 'Median...', 'radius=' + str(2))
				im = fc.minMax(im, 210, 255)
				im = fc.minMax(im, 200, 200)
				IJ.run(im, 'Invert', '')
				points = getConnectedComponents(im)
			
			IJ.log(str(k) + '-' + str(len(points)) + ' peaks')
			IJ.log(str(k) + '--' + str(points[:10]) + ' peaks')
			loader1.serialize(points, os.path.join(peaksFolder, 'peaks_channel_' + channel + '_' + str(k).zfill(4)) ) #xxx is there no other way to serialize in imagej without using a trakem2 loader ? I do not think so, see below ...
			im.close()

###################################
# # # # # # # deserializing attempt:
###################################
# # # # # from java.io import ObjectOutputStream, FileOutputStream, ObjectInputStream, FileInputStream
# # # # # from org.python.util import PythonObjectInputStream
# # # # # path = r'E:\Users\Thomas\Wafer_SFN_2016\OrderRetrieval_SFN_2016\sectionOutput\peaks_488-546_0102'
# # # # # #out = ObjectOutputStream(FileOutputStream(path))
# # # # # #out.writeObject(ob)
# # # # # #out.close()
# # # # # #print out
# # # # # a = []
# # # # # r = PythonObjectInputStream(FileInputStream(path))
# # # # # print r
# # # # # #ob = r.read()
# # # # # #print ob
# # # # # #ob = r.readFully(a)
# # # # # #print a
# # # # # print r.resolveObject(r)
# # # # # for t in range(1000):
	# # # # # print r.read()
# # # # # #ob = r.readObject()
# # # # # #r.close()
# # # # # #print ob


def getDogs(im, radius, threshold, doSubpixel, doMedian):
	points = []
	img = ImageJFunctions.wrap(im)
	interval = img
	cal = im.getCalibration()
	calibration = [cal.pixelWidth, cal.pixelHeight, cal.pixelDepth]
	detector = LogDetector(img, interval, calibration, radius, threshold, doSubpixel, doMedian)
	detector.process()
	peaks = detector.getResult()
	for peak in peaks:
		points.append([peak.getDoublePosition(0) / cal.pixelWidth, peak.getDoublePosition(1) / cal.pixelHeight])
	return points

def getConnectedComponents(im):
	points = []
	roim = RoiManager(True)
	pa = ParticleAnalyzer(ParticleAnalyzer.ADD_TO_MANAGER + ParticleAnalyzer.EXCLUDE_EDGE_PARTICLES, Measurements.AREA, None, 0, Double.POSITIVE_INFINITY, 0.0, 1.0)

	pa.setRoiManager(roim)

	pa.analyze(im)
	for roi in roim.getRoisAsArray():
		points.append(roi.getContourCentroid())
	roim.close()

	return points

def getDistance(atom, pairs, allDogs, corrMat, affineDict, matchPointsDict):
	dogsContainer = ArrayList()
	nPairs = len(pairs)
	if nPairs !=0:
		while atom.get() < len(pairs):
			k = atom.getAndIncrement()
			if k < len(pairs):
				IJ.log('Processing pair ' + str(k) )
				# if k%(int(nPairs/1000.)) == 0:
					# print str(int(k/float(nPairs) * 1000.)), '/1000 done'

				id1, id2 = pairs[k][0], pairs[k][1]
				dogs1 = allDogs[id1]
				dogs2 = allDogs[id2]

				dogsContainer.add(dogs1)
				dogsContainer.add(dogs2)

				comparePairs = Matching.descriptorMatching(dogsContainer, 2, dp, 0)

				distances = 0
				inliers = comparePairs[0].inliers

				if len(inliers)>0:
					for inlier in inliers:
						distances = distances + inlier.getDistance()
					meanDistance = distances/float(len(inliers))

					# # # # corrMat[id1][id2] = meanDistance
					corrMat[id1][id2] = len(dogs1) - len(inliers)

					affineDict[(id1, id2)] = comparePairs[0].model.createAffine()

					points1 = []
					points2 = []

					for inlier in inliers:
						p1 = inlier.getP1().getL()
						p2 = inlier.getP2().getL()

						points1.append([p1[0], p1[1]])
						points2.append([p2[0], p2[1]])

					matchPointsDict[(id1, id2)] = [points1, points2]
					IJ.log('dist value found ' + str(id1) + '-' + str(id2) + '--' + str(meanDistance))
				else:
					corrMat[id1][id2] = 500000
			dogsContainer.clear()
		IJ.log('getDistance has run')

def getMatchingCost(dogsContainer, dogs1, dogs2, sizes1, sizes2):
	# calculate a matching, this time the crop is probably larger and the pair of section has already matched
	dogsContainer.clear()
	dogsContainer.add(dogs1)
	dogsContainer.add(dogs2)
	comparePairs = Matching.descriptorMatching(dogsContainer, 2, dp, 0)
	inliers = comparePairs[0].inliers
	
	# transform the dogs to standard points in lists
	theDogs1 = [dog.getPosition() for dog in dogs1]
	theDogs2 = [dog.getPosition() for dog in dogs2]
	
	# get the size differences between matching beads and get the inlier indexes
	sizeCosts = []
	inlierIndexes1 = []
	inlierIndexes2 = []
	for inlier in inliers:
		# find p1 in theDogs1 to get the index and access its size
		p1 = inlier.getP1().getL()
		id1 = theDogs1.index(p1)
		inlierIndexes1.append(id1)
		size1 = sizes1[id1]
		
		p2 = inlier.getP2().getL()
		id2 = theDogs2.index(p2)
		inlierIndexes2.append(id2)
		size2 = sizes2[id2]
		
		sizeCosts.append(abs(size2-size1))
	
	# find the outliers and get their size: they are counted as cost too. A 5 pixel bead that disappears adds a cost equal to 5.
	outlierIndexes1 = set(inlierIndexes1) ^ set(range(len(theDogs1)))
	for outlierId in outlierIndexes1:
		sizeCosts.append(sizes1[outlierId]) # xxx warning: or is it sizes2 ? I do not think so ...

	outlierIndexes2 = set(inlierIndexes2) ^ set(range(len(theDogs2)))
	for outlierId in outlierIndexes2:
		sizeCosts.append(sizes2[outlierId])

	totalCost = sum(sizeCosts)
	
	return totalCost

# def countEvents(dogsContainer, dogs1, dogs2, sizes1, sizes2):
	# disappearingBeads = countEndings(dogsContainer, dogs1, dogs2, sizes1, sizes2)
	# # appearingBeads = countEndings(dogsContainer, dogs2, dogs1, sizes2, sizes1)
	# return disappearingBeads + appearingBeads

def getEvents(atom, pairs, allPeaks, allCropedDogs, cropSimilarity, corrMat, affineDict, matchPointsDict):
	'''
	allCropedDogs already croped with cropMatching
	'''
	dogsContainer = ArrayList()
	nPairs = len(pairs)
	if nPairs !=0:
		while atom.get() < len(pairs):
			k = atom.getAndIncrement()
			if k < len(pairs):
				IJ.log('Processing pair ' + str(k) )
				# if k%(int(nPairs/1000.)) == 0:
					# print str(int(k/float(nPairs) * 1000.)), '/1000 done'

				id1, id2 = pairs[k][0], pairs[k][1]
				dogs1, dogs2 = allCropedDogs[id1], allCropedDogs[id2]
				# print '*** LEN --- ', len(dogs2)

				# Pairwise screening
				dogsContainer = ArrayList()
				dogsContainer.add(dogs1)
				dogsContainer.add(dogs2)
				comparePairs = Matching.descriptorMatching(dogsContainer, 2, dp, 0)
				inliers = comparePairs[0].inliers

				if 'B6' in inputFolder:
					inlierThreshold = 15
				else:
					inlierThreshold = 5
					
				if len(inliers) > inlierThreshold: # this pair of sections is matching
					# saving then inverting the transform
					affineT = comparePairs[0].model.createAffine()
					affDeterminant = affineT.getDeterminant()
					
					if abs(1 - affDeterminant) > 0.1: # it sometimes happens that I get weird transforms, do not understand why yet ...
						print 'Error: could not create the inverse - ', str(affineT), 'for pair', str([id1,id2]), 'len(inliers)', len(inliers)
						IJ.log('Error: could not create the inverse - ' + str(affineT) + ' for pair ' + str([id1,id2]))
						corrMat[id1][id2] = 50000
					
					else:
						affInverse = affineT.createInverse()
						affineDict[(id1, id2)] = affineT

						# saving the match points
						points1, points2 = [], []
						for inlier in inliers:
							p1, p2 = inlier.getP1().getL(), inlier.getP2().getL()
							points1.append([p1[0], p1[1]])
							points2.append([p2[0], p2[1]])
						matchPointsDict[(id1, id2)] = [points1, points2]

						# should I take the convex hull of the total matches and use #events/area ?
							# a fake match may fail
								# really ? I am not sure why I wrote that ...
							# but by chance a fake inlier could give a wrong overlapping region

						# transform peaks1 into peaks2 (here these are all peaks, there was no crop)
						
						# keep the size information
						peaks1 = allPeaks[id1]
						peaks2 = allPeaks[id2]
						
						sizes1 = [peak[2] for peak in peaks1]
						sizes2 = [peak[2] for peak in peaks2]
						
						# because there is the size info in the third position
						peaks1 = [peak[:2] for peak in peaks1]
						peaks2 = [peak[:2] for peak in peaks2]
						
						transformedPeaksList1 = array(2 * len(peaks1) * [0], 'd')
						affInverse.transform(pointListToList(peaks1), 0, transformedPeaksList1, 0, len(peaks1))
						transformedPeaks1 = listToPointList(transformedPeaksList1)

						# put back the size information
						transformedPeaks1 = [peak + [size] for peak,size in zip(transformedPeaks1, sizes1)]
						peaks2 = [peak + [size] for peak,size in zip(peaks2, sizes2)]
						
						# crop the aligned peaks with the bounding box defined by cropSimilarity
						croppedTransformedPeaks1 = cropPeaks(transformedPeaks1, cropSimilarity)
						croppedPeaks2 = cropPeaks(peaks2, cropSimilarity)

						# getting once more the new sizes (cropping has occured)
						sizes1 = [peak[2] for peak in croppedTransformedPeaks1]
						sizes2 = [peak[2] for peak in croppedPeaks2]
						
						matchingCost = getMatchingCost(dogsContainer, pointsToDogs(croppedTransformedPeaks1), pointsToDogs(croppedPeaks2), sizes1, sizes2)
						corrMat[id1][id2] = matchingCost

						IJ.log(str(id1) + '-' + str(id2) + '--' + str(matchingCost) )
						# except Exception, e:
							# IJ.log('Did not succeed in inverting the affine transform in pair ' + str(id1) + '-' + str(id2))
				dogsContainer.clear()
		IJ.log('getDistance has run')

def getCC(im1,im2):
	im1, im2 = map(ImagePlusAdapter.wrap, [im1, im2])
	cc = CrossCorrelation(im1, im2)
	cc.process()
	return cc.getR()

def getHighResCorrMat(atom, pairs, affineDict, stitchedSectionPaths, corrMat):
	counter = 0

	if counter%400 ==399:
		fc.closeProject(p)
	p, loader, layerset, nLayers = fc.getProjectUtils( fc.initTrakem(ccCalculationFolder, 2) )
	# p.saveAs(os.path.join(ccCalculationFolder, 'pproject' + str(0) + '.xml'), True)
	layer1 = layerset.getLayers().get(0)
	layer2 = layerset.getLayers().get(1)

	# disp = Display(p, layerset.getLayers().get(0))
	# disp.showFront(layerset.getLayers().get(0))

	while atom.get() < len(pairs) :
		k = atom.getAndIncrement()
		if k < len(pairs):
			if counter%300 ==299:
				fc.closeProject(p)
				time.sleep(3)
				p, loader, layerset, nLayers = fc.getProjectUtils( fc.initTrakem(ccCalculationFolder, 2) )
				p.saveAs(os.path.join(ccCalculationFolder, 'pproject' + str(k) + '.xml'), True)
				layer1 = layerset.getLayers().get(0)
				layer2 = layerset.getLayers().get(1)

			pair = pairs[k]
			aff = affineDict[pair]

			path1 = stitchedSectionPaths[pair[0]]
			path2 = stitchedSectionPaths[pair[1]]

			patch1 = Patch.createPatch(p, path1)
			patch2 = Patch.createPatch(p, path2)

			layer1.add(patch1)
			layer2.add(patch2)

			patch1.setAffineTransform(aff)
			# patch2.updateBucket()

			fc.resizeDisplay(layerset)
			bb = layerset.get2DBounds()
			#shrink the BB ? 

			im1 = loader.getFlatImage(layer1, bb, 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layer1.getAll(Patch), True, Color.black, None)
			im2 = loader.getFlatImage(layer2, bb, 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layer2.getAll(Patch), True, Color.black, None)

			# im1 = imToPeak(im1, 10, 10, 3, True, True, [170,255])
			# im2 = imToPeak(im2, 10, 10, 3, True, True, [170,255])

			# im1 = imToPeak(im1, 20, 20, 10, True, False)
			# im2 = imToPeak(im2, 20, 20, 10, True, False)

			# # im1 = imToPeak(fc.blur(im1,1), 40, 40, 3, True, True)
			# # im2 = imToPeak(fc.blur(im2,1), 40, 40, 3, True, True)

			# im1 = imToPeak(im1, 100, 100, 2, True, True, [160,255])
			# im2 = imToPeak(im2, 100, 100, 2, True, True, [160,255])

			corr = Math.exp((4-getCC(im1,im2)) * 2)
			# corr = 1./getCC(im1,im2)
			corrMat[pair[0]][pair[1]] = corr
			IJ.log('Processing pair ' + str(k) + ' with correlation ' + str(corr))
			if k%100 == 0:
				print 'Processing pair ', str(k), ' with correlation ', str(corr)
			# print 'Processing pair ', str(k), ' with correlation '

			# return
			layer1.remove(patch1)
			layer2.remove(patch2)
			counter = counter + 1
	fc.closeProject(p)

def getCCCorrMat(atom, pairs, affineDict, stitchedSectionPaths, theMat):
	layer1 = layersetZ.getLayers().get(0)
	layer2 = layersetZ.getLayers().get(1)

	while atom.get() < len(pairs) :
		k = atom.getAndIncrement()
		if k < len(pairs):

			pair = pairs[k]
			aff = affineDict[pair]

			path1 = stitchedSectionPaths[pair[0]]
			path2 = stitchedSectionPaths[pair[1]]

			patch1 = Patch.createPatch(pZ, path1)
			patch2 = Patch.createPatch(pZ, path2)

			# try: # in case of a rare exception caught once, see below
			with lock:# projet operations
				layer1.add(patch1)
				layer2.add(patch2)
				patch1.setAffineTransform(aff)
				# patch2.updateBucket()

				fc.resizeDisplay(layersetZ)
				bb = layersetZ.get2DBounds()
				
				factor = 0.65
				
				# shrink the BB ? 
				bb = Rectangle(int(bb.width * (1 - factor)/2), int(bb.height * (1 - factor)/2), int(bb.width * factor), int(bb.height * factor))

				im1 = loaderZ.getFlatImage(layer1, bb, 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layer1.getAll(Patch), True, Color.black, None)
				im2 = loaderZ.getFlatImage(layer2, bb, 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layer2.getAll(Patch), True, Color.black, None)
				# im1.show()
				# im2.show()
				# 8/0
				
				layer1.remove(patch1)
				layer2.remove(patch2)

			IJ.run(im1, 'Invert', '')
			IJ.run(im2, 'Invert', '')
			
			# im1.show() # the first pair looked ok
			# im2.show()
			# 8/0
				

			# im1 = imToPeak(im1, 10, 10, 3, True, True, [170,255])
			# im2 = imToPeak(im2, 10, 10, 3, True, True, [170,255])

			# im1 = imToPeak(im1, 20, 20, 10, True, False)
			# im2 = imToPeak(im2, 20, 20, 10, True, False)

			# # im1 = imToPeak(fc.blur(im1,1), 40, 40, 3, True, True)
			# # im2 = imToPeak(fc.blur(im2,1), 40, 40, 3, True, True)

			# im1 = imToPeak(im1, 100, 100, 2, True, True, [160,255])
			# im2 = imToPeak(im2, 100, 100, 2, True, True, [160,255])

			corr = Math.exp( (4 - getCC(im1,im2)) * 2)
			# corr = 1./getCC(im1,im2)
			
			theMat[pair[0]][pair[1]] = corr
			IJ.log('Processing pair ' + str(k) + ' with correlation ' + str(corr))
		# except Exception, e:
			# IJ.log('Catching in case a rare exception that has occurred once :  im1 = loaderZ.getFlatImage(layer1, bb, 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layer1.getAll(Patch), True, Color.black, None)NullPointerException: ava.lang.NullPointerException')
			
			if k%100 == 0:
				print 'Processing pair ', str(k), ' with correlation ', str(corr)
				IJ.log('Processing pair ' + str(k) + ' with correlation ' + str(corr))
	
	
# def reorderProject(projectPath, reorderedProjectPath, order):
	# folder = os.path.dirname(os.path.normpath(projectPath))

	# pReordered, loaderReordered, layersetReordered, nLayers = fc.getProjectUtils( fc.initTrakem(folder, len(order)) )
	# pReordered.saveAs(reorderedProjectPath, True)

	# IJ.log('reorderedProjectPath ' + reorderedProjectPath)

	# project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)

	# for l,layer in enumerate(project.getRootLayerSet().getLayers()):
		# IJ.log('Inserting layer ' + str(l) + '...')
		# reorderedLayer = layersetReordered.getLayers().get(order.index(l))
		# # for ob in layer.getDisplayables():
			# # reorderedLayer.add(ob.clone(pReordered, False))
			# # xxx something missing to update the layer ?
		# for patch in layer.getDisplayables():
			# patchPath = loader.getAbsolutePath(patch)
			# patchTransform = patch.getAffineTransform()
			
			# newPatch = Patch.createPatch(pReordered, patchPath)
			# reorderedLayer.add(newPatch)
			# newPatch.setAffineTransform(patchTransform)        

	# fc.closeProject(project)
	# fc.resizeDisplay(layersetReordered) #I should check the size of the display of the reordered project, should be the same as the retrievalProject
	# pReordered.save()
	# fc.closeProject(pReordered)
	# IJ.log('Project reordering done')

def affineRealignProject(sourcePath, targetPath, SIFTMatchesPath, optionalMatchesPath = None):
	shutil.copyfile(sourcePath, targetPath)
	p, loader, layerset, nLayers = fc.openTrakemProject(targetPath)
	p.saveAs(targetPath, True)
	affineDict = loader.deserialize(SIFTMatchesPath)[1]
	if optionalMatchesPath != None:
		affineDictsOptional = [loader.deserialize(optionalMatchPath)[1] for optionalMatchPath in optionalMatchesPath]

	aff_0_To_N = AffineTransform()
	firstPair = True
	for l, layer1 in enumerate(layerset.getLayers()):
		if l < nLayers - 1:
		# if l < 100:
			IJ.log('Processing layer - ' +str(l))
			layer1 = layerset.getLayers().get(l)
			layer2 = layerset.getLayers().get(l+1)

			patch1 = layer1.getDisplayables(Patch)[0]
			patch2 = layer2.getDisplayables(Patch)[0]

			aff = AffineTransform()

			aff1 = patch1.getAffineTransform()
			aff2 = patch2.getAffineTransform()

			id1 = int(os.path.splitext(os.path.basename(patch1.getFilePath()))[0].split('_')[-1])
			id2 = int(os.path.splitext(os.path.basename(patch2.getFilePath()))[0].split('_')[-1])

			aff12 = AffineTransform()
			thereIsATransform = False
			
			IJ.log('a')            
			if ( (id1, id2) in affineDict and 
			abs(1 - affineDict[(id1, id2)].getDeterminant()) > 0.1):
				print 'determinant for pair', id1, id2, affineDict[(id1, id2)].getDeterminant()
				print 'Error: could not apply a non-invertible transform for pair ', str([id1, id2])
				IJ.log('Error: could not apply a non-invertible transform for pair ' + str([id1, id2]))
			if ( (id1, id2) in affineDict and 
			abs(1-affineDict[(id1, id2)].getDeterminant()) < 0.1):
				# print 'determinant for pair', id1, id2, affineDict[(id1, id2)].getDeterminant()
				# IJ.log(str((id1, id2)) + ' in dict')
				aff12.concatenate(affineDict[(id1, id2)].createInverse())
				thereIsATransform = True
			elif (id2, id1) in affineDict:
				# IJ.log(str((id2, id1)) + ' in dict')
				if abs(1 - affineDict[(id2, id1)].getDeterminant()) > 0.1:
					print 'Error with the forward transform'
				else:
					aff12.concatenate(affineDict[(id2, id1)])
					thereIsATransform = True
			else:
				if optionalMatchesPath != None:
					for optionalDict in affineDictsOptional:
						if not thereIsATransform:
							if ( (id1, id2) in optionalDict and
							abs(1 - optionalDict[(id1, id2)].getDeterminant()) > 0.1):
								print 'Error: could not apply a non-invertible transform for pair ', str([id1, id2])
								IJ.log('Error: could not apply a non-invertible transform for pair ' + str([id1, id2]))
							if ((id1, id2) in optionalDict and 
							abs(1 - optionalDict[(id1, id2)].getDeterminant() < 0.1)):
								thereIsATransform = True
								aff12.concatenate(optionalDict[(id1, id2)].createInverse())
							elif (id2, id1) in optionalDict:
								if abs(1 - optionalDict[(id2, id1)].getDeterminant()) > 0.1:
									print 'Error with the forward transform'
								else:
									thereIsATransform = True
									aff12.concatenate(optionalDict[(id2, id1)])
								
			if thereIsATransform:
				aff.concatenate(aff_0_To_N) # apply all the previous affine
				aff.concatenate(aff12) # apply the new affine from n to n+1
				patch2.setAffineTransform(aff)
				patch2.updateBucket()
			aff_0_To_N.concatenate(aff12)

	# disp = Display(p, layerset.getLayers().get(0))
	# disp.showFront(layerset.getLayers().get(0))
	IJ.log('d')
	# fc.resizeDisplay(layerset)
	IJ.log('e')
	p.save()
	IJ.log('f')
	fc.closeProject(p)
	IJ.log('g')

def sumAffineProject(projectPath, orderedImagePaths, consecAffineTransformPaths):
	p, loader, layerset, nLayers = fc.getProjectUtils( fc.initTrakem(baseFolder, nSections) )    
	p.saveAs(projectPath, True)

	aff_0_To_N = AffineTransform()
	for l, layer in enumerate(layerset.getLayers()):
		patch = Patch.createPatch(p, orderedImagePaths[l])
		layer.add(patch)
	for l, layer1 in enumerate(layerset.getLayers()):
		if l < nLayers - 1:
			IJ.log('Processing layer -- ' +str(l))
			layer2 = layerset.getLayers().get(l+1)

			patch1 = layer1.getDisplayables(Patch)[0]
			patch2 = layer2.getDisplayables(Patch)[0]

			aff = AffineTransform()

			aff1 = patch1.getAffineTransform()
			aff2 = patch2.getAffineTransform()

			aff12 = loader.deserialize(consecAffineTransformPaths[l])

			aff.concatenate(aff_0_To_N) # apply all the previous affine
			aff.concatenate(aff12) # apply the new affine from n to n+1
			patch2.setAffineTransform(aff)
			patch2.updateBucket()
			aff_0_To_N.concatenate(aff12)
	fc.resizeDisplay(layerset)
	p.save()
	time.sleep(3)
	fc.closeProject(p)    
	
def elasticRealignProject(sourcePath, targetPath):
	shutil.copyfile(sourcePath, targetPath)
	p, loader, layerset, nLayers = fc.openTrakemProject(targetPath)
	p.saveAs(targetPath, True)

	dogsContainer = ArrayList()
	newMosaicSize = getNewMosaicSize()

	for l, layer in enumerate(layerset.getLayers()):
		# if (l > 0) and (l < 100):
		if (l > 0):
		# if (l < nSections - 1):
		# if (l > - 1):
			IJ.log('Elastic aligning layer ' + str(l))
			bb = layerset.get2DBounds()

			layer1 = layerset.getLayers().get(l-1)
			layer2 = layerset.getLayers().get(l)

			# getting im1 to be transformed
			sectionIndex1 = layer1.getAll(Patch)[0]
			im1 = loader.getFlatImage(layer1, bb, 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layer1.getAll(Patch), True, Color.black, None)
			im1Path = os.path.join(calculationFolder, 'bunwarpTransform_finalAlignment_im1_' + channel + '_' + str(l).zfill(4) + '.tif')
			IJ.save(im1, im1Path)

			# getting the peaks from the preprocessed section1
			imPreprocessed1 = IJ.openImage(os.path.join(preprocessedFolder, 'stitchedRotatedSection_' + channel + '_' + str(sectionIndex1).zfill(4) + '.tif'))
			points1 = getConnectedComponents(imPreprocessed1)
			IJ.log('Layer '  + str(l-1) + ' has ' + str(len(points1)) + ' peaks')
			dogs1 = pointsToDogs(points1)
			imPreprocessed1.close()

			# getting im1 to be transformed
			# im2Path = os.path.join(sectionOutputFolder, 'bunwarpTransform_finalAlignment_im2_' + channel + '_' + str(l).zfill(4) + '.tif')
			im2Index = int(os.path.splitext(os.path.basename(im2Path))[0].split('_')[-1])
			im2Path = os.path.normpath(layer.getDisplayables(Patch)[0].getImageFilePath())

			# getting the peaks from the preprocessed section2
			# # # # For layer2, I can take the already calculated peaks because the image has sill not been warped yet.
			# # # imPreprocessed2 = IJ.openImage(os.path.join(preprocessedFolder, 'stitchedRotatedSection_' + channel + '_' + str(sectionIndex2).zfill(4) + '.tif'))
			# # # points2 = getConnectedComponents(imPreprocessed2)
			# # # IJ.log('Layer '  + str(l-1) + ' has ' + str(len(points1)) + ' peaks')
			# # # dogs2 = pointsToDogs(points2)
			# # # imPreprocessed2.close()
			dogs2 = pointsToDogs(loader.deserialize(os.path.join(peakFolder, 'peaks_' + channel + '_' + str(sectionIndex2).zfill(4))))

			dogsContainer.add(dogs2) # WARNING: order is inverted
			dogsContainer.add(dogs1)
			comparePairs = Matching.descriptorMatching(dogsContainer, 2, dp, 0)
			dogsContainer.clear()

			inliers = comparePairs[0].inliers
			if not (len(inliers)>0):
				IJ.log('ERROR: There should be inliers !')
				print '# # # WARNING # # # : there are no inliers in layer ', l
			else:
				stack1 = Stack()
				stack2 = Stack()

				for inlier in inliers:
					p1 = inlier.getP1().getL()
					p2 = inlier.getP2().getL()

					stack1.push(Point(int(p1[0]), int(p1[1])))
					stack2.push(Point(int(p2[0]), int(p2[1])))

				trans = computeTransformationBatch(im1.getWidth(), im1.getHeight(), newMosaicSize[0], newMosaicSize[1], stack1, stack2, unwarpParam)
				transPath = os.path.join(calculationFolder, 'bunwarpTransform_finalAlignment_' + channel + '_' + str(l).zfill(4))

				trans.saveDirectTransformation(transPath)
				# trans.saveInverseTransformation(transPath)

				transformedPath = os.path.join(warpFolder, 'bunwarpTransform_finalAlignment_elastiked_im2_' + channel + '_' + str(l).zfill(4) + '.tif')

				elasticTransformImageMacro(im1Path, im2Path, transPath, transformedPath) # targetPath, sourcePath, transPath, transformedPath

				im2Elastiked = IJ.openImage(transformedPath)
				IJ.run(im2Elastiked, '8-bit', '')
				im2Elastiked = fc.minMax(im2Elastiked, 100,100)
				IJ.save(im2Elastiked, transformedPath)

				layer.remove(layer.getDisplayables(Patch)[0])
				patch = Patch.createPatch(p, transformedPath)
				layer.add(patch)
				patch.updateBucket()
				fc.resizeDisplay(layerset)
	p.save()
	fc.closeProject(p)

##########
# Garbage
##########
def pointListToList(pointList): # [[1,2],[5,8]] to [1,2,5,8]
	l = array(2 * len(pointList) * [0], 'd')
	for id, point in enumerate(pointList):
		l[2*id] = point[0]
		l[2*id+1] = point[1]
	return l

def listToPointList(l): # [1,2,5,8] to [[1,2],[5,8]]
	pointList = []
	for i in range(len(l)/2):
		pointList.append([l[2*i], l[2*i+1]])
	return pointList

def pointListToDOGPs(points):
	DOGPs = ArrayList()
	for point in points:
		DOGPs.add(DifferenceOfGaussianPeak( [int(point[0]), int(point[1]) ] , IntType(255), SpecialPoint.MAX ))
	return DOGPs

def sectionToPoly(l):
	return Polygon( [int(a[0]) for a in l] , [int(a[1]) for a in l], len(l))

def getAffFromPoints(sourcePoints, targetPoints):
	sourceDOGPs = pointListToDOGPs(sourcePoints)
	targetDOGPs = pointListToDOGPs(targetPoints)

	dogpContainer = ArrayList()

	dogpContainer.add(sourceDOGPs)
	dogpContainer.add(targetDOGPs)

	comparePairs = Matching.descriptorMatching(dogpContainer, 2, dp, 0)
	aff = comparePairs[0].model.createAffine()
	return aff

def parallelWarpCC(pairs):
	nPairs = len(pairs)
	newMosaicSize = getNewMosaicSize()
	if nPairs !=0:
		while atom.get() < len(pairs):
			k = atom.getAndIncrement()
			if k < len(pairs):
				IJ.log('Processing pair ' + str(k) )

				pair = pairs[k][0]
				points1, points2 = pairs[k][1]

				stack1 = Stack()
				stack2 = Stack()
				for point1 in points1:
					# point = Point(2)
					# point.setPosition([int(point1[0]), int(point1[1])])
					# stack1.push(point)
					stack1.push(Point(int(point1[0]), int(point1[1])))
				for point2 in points2:
					# point = Point(2)
					# point.setPosition([int(point2[0]), int(point2[1])])
					# stack2.push(point)
					stack2.push(Point(int(point2[0]), int(point2[1])))
				trans = computeTransformationBatch(newMosaicSize[0], newMosaicSize[1], newMosaicSize[0], newMosaicSize[1], stack1, stack2, unwarpParam)

				transPath = os.path.join(calculationFolder, 'bunwarpTransform_' + channel + '_' + str(pair[0]) + '_' + str(pair[1]))

				trans.saveDirectTransformation(transPath)
				# trans.saveInverseTransformation(transPath)

				sourcePath = os.path.join(rawFolder, 'stitchedRotatedSection_' + channel + '_' + str(pair[0]).zfill(4) + '.tif')
				targetPath = os.path.join(rawFolder, 'stitchedRotatedSection_' + channel + '_' + str(pair[1]).zfill(4) + '.tif')

				transformedPath = os.path.join(warpFolder, 'transformed_' + channel + '_' + str(pair[0]) + '_' + str(pair[1]) + '.tif')
				elasticTransformImageMacro(targetPath, sourcePath, transPath, transformedPath)

				im1 = IJ.openImage(targetPath)
				im2 = IJ.openImage(transformedPath)
				IJ.run(im2, '8-bit', '')

				im1 = fc.minMax(im1, 5, 180)
				im2 = fc.minMax(im2, 5, 180)

				# corr = Math.exp((4-getCC(im1,im2)) * 2)
				corr = 1./getCC(im1,im2)
				corrMat[pair[0]][pair[1]] = corr

				# crop to the center (or to the barycenter of the matchpoints)
				# bary2 = barycenter(points2)
				bary2 = [int(newMosaicSize[0]/2.), int(newMosaicSize[1]/2.)] # trying with simply the center

				x1 = max(0, bary2[0] - cropForSimilarity[0]/2.)
				x2 = min(mosaicX, bary2[0] + cropForSimilarity[0]/2.)
				y1 = max(0, bary2[1] - cropForSimilarity[1]/2.)
				y2 = min(mosaicY, bary2[1] + cropForSimilarity[1]/2.)
				roi = Roi(x1, y1, x2 - x1, y2 - y1)

				im1 = fc.crop(im1,roi)
				im2 = fc.crop(im2,roi)

				# corr = Math.exp((4-getCC(im1,im2)) * 2)
				corr = 1./getCC(im1,im2)
				corrMat_Crop[pair[0]][pair[1]] = corr

				im1.close()
				im2.close()

				# # I can remove at the end
				# os.remove(transformedPath)
				# os.remove(transPath)


				logMessage = 'Pair number ' + str(k) + ' (' + str(pair[0]) + ',' + str(pair[1]) + ') has a CC value of ' + str(corr)
				IJ.log(logMessage)
				# print logMessage

def orderDistance(order1, order2):
	''' order1 is the reference '''
	allCosts = []
	for id, section in enumerate(order2[:-1]):
		nextSection = order2[id + 1]
		
		sectionPosition = order1.index(section)
		nextSectionPosition = order1.index(nextSection)
		
		distance = min(abs(nextSectionPosition - sectionPosition) - 1, 10)

		allCosts.append(distance)
	totalCost = sum(allCosts)
	print Counter(allCosts)
	return totalCost
				
#################################
###### Parameters to enter ######
#################################
# inputFolder = os.path.normpath(os.path.join(r'E:\Users\Thomas\Thesis\B6\B6_Wafer_203_Beads_WorkingFolder\AllBeads', ''))
inputFolder = os.path.normpath(os.path.join(r'E:\Users\Thomas\Thesis\C1\C1_Beads_Reordering\AllBeads', ''))

# twoStepsReordering = True
twoStepsReordering = False
peakDecay = 0.8

beadChannels = ['488', '546']
mosaicLayout = [1,1]
overlap = 50
# stitchingChannel = 'brightfield'
stitchingChannel = '488' # use one of the bead channels if mosaicLayout = [1,1] and that there is no brightfield

firstSectionFolder = os.path.join(inputFolder, os.walk(inputFolder).next()[1][0])
firstImagePath = os.path.join(firstSectionFolder, os.walk(firstSectionFolder).next()[2][0])
im0 = IJ.openImage(firstImagePath)
width = im0.getWidth()
height = im0.getHeight()
im0.close()

# width, height = 2048, 2048
# width, height = 1388, 1040

# refSectionOffset = -22.5 * PI / 180 # change this only if there is no target_highres_landmarks
refSectionOffset = 58 * PI / 180 # for A7_200


# The scaling factor between the magnification of the wafer overview and the magnification of the bead imaging
# beadsToWaferFactor = 13 #  Leica 630/52 and 1180/88
# beadsToWaferFactor = 630/float(155) #  (ZeissZ1, 5x) to (Nikon, 20x) 155 630
# beadsToWaferFactor = 630/float(155) * 1 #  (ZeissZ1, 5x) to (Z1, 20x)

# beadsResolution = 
# waferResolution = 1804000/1388.
# beadsToWaferFactor = beadsResolution/float(waferResolution)
beadsToWaferFactor = 4.01409 # from Visitron20x to ZeissZ1 5x

# parameter for maxima finder
maximaNoiseTolerance = 200
if 'C1' in inputFolder:
	noiseTolerance = 150


#################################
#################################

####### matching parameters, probably not optimal yet #######
dp = DescriptorParameters()
# dp.model = RigidModel2D() # old, not good, use affine now
dp.model = AffineModel2D()
dp.dimensionality = 2
dp.fuse = 2 # no overlay
dp.brightestNPoints = 2000
dp.redundancy = 1
# dp.ransacThreshold = 50 # 37 pairs with neighbors = 3
dp.ransacThreshold = 10 # 37 pairs with neighbors = 3
# dp.ransacThreshold = 1 # 29 pairs with 3 neighbors
# dp.ransacThreshold = 1000 # 42 pairs with 3 neighbors
dp.lookForMaxima = True

dp.minSimilarity = 100
# dp.numNeighbors = 6 # 10 pairs for 5 sections
# dp.numNeighbors = 4 # too few matches in A7_200
# dp.numNeighbors = 3 # 600 pairs for 200 sections in A7_200 ? 37 pairs for 5 sections
# dp.numNeighbors = 3 #  pairs for 5 sections
# dp.numNeighbors = 4 #  pairs for 5 sections
# dp.numNeighbors = 8
dp.numNeighbors = 3

print 'dp.minSimilarity', dp.minSimilarity
# dp.minSimilarity = 10
print 'minInlierFactor', dp.minInlierFactor
print 'sigma1', dp.sigma1
print 'sigma2', dp.sigma2
print 'threshold', dp.threshold
print 'filterRANSAC', dp.filterRANSAC
print 'redundancy', dp.redundancy # 1 is ok
print 'dp.significance', dp.significance
# dp.significance = 10
# 8/0


# Older parameters. Keep in case
# dp.numNeighbors = 3
# dp.brightestNPoints = 3
# dp.maxIterations = 1000
# dp.iterations = 1000
# dp.max = 255
# dp.ransacThreshold = 1000

# parameters for DoGs
radius = 10
threshold = 0.7
doSubpixel = True
doMedian = False

#################################
###### Folder initializations ###
#################################
# Experiment-279_b0s1c2x920-1388y360-1040m0


# # /!\TO COMMENT
# #####################################################
# #####################################################
# #####################################################
# # For Zeiss BIB manual experiment: preprocess the files to the right format
# wrongIndexes = [1,2,3,20,4,6,5,19,7,16,15,17,8,18,11,9,13,12,14,10]
# for idChannel, beadChannel in enumerate(beadChannels):
	# imageNames = filter(lambda x: ('c' + str(idChannel + 1) + 'x' in x) and (os.path.splitext(x)[1] =='.tif'), os.listdir(inputFolder))
	# for imageName in imageNames:
		# id = int(imageName.split('s')[1].split('c')[0])
		# sectionFolder = fc.mkdir_p(os.path.join(inputFolder, 'section_' + str(wrongIndexes[id]-1).zfill(4)))
		# shutil.copyfile(os.path.join(inputFolder, imageName), os.path.join(sectionFolder, 'section_' + str(wrongIndexes[id]-1).zfill(4) + '_channel_' + str(beadChannel) + '_tileId_00-00-mag.tif'))
# #####################################################
# #####################################################
# #####################################################

if stitchingChannel in beadChannels:
	allChannels = beadChannels
else:
	allChannels = beadChannels + [stitchingChannel]

baseFolder = os.path.dirname(inputFolder)
sectionsCoordinates = fc.readSectionCoordinates(os.path.join(baseFolder, 'preImaging', 'source_sections_mag.txt'))

nSections = len(os.walk(inputFolder).next()[1])
IJ.log('nSections: ' + str(nSections))

rawFolder = fc.mkdir_p(os.path.join(baseFolder, 'rawSections'))
preprocessedFolder = fc.mkdir_p(os.path.join(baseFolder, 'preprocessedSections'))
preprocessedMosaicsFolder = fc.mkdir_p(os.path.join(baseFolder, 'preprocessedMosaics'))
peaksFolder = fc.mkdir_p(os.path.join(baseFolder, 'peaks'))
blobizedFolder = fc.mkdir_p(os.path.join(baseFolder, 'blobizedSections'))
forCCFolder = fc.mkdir_p(os.path.join(baseFolder, 'forCCSections'))
calculationFolder = fc.mkdir_p(os.path.join(baseFolder, 'calculations'))
ccCalculationFolder = fc.mkdir_p(os.path.join(baseFolder, 'ccCalculations'))

# calculate the offset based on first section and section template
targetLandmarksPath = os.path.join(baseFolder, 'preImaging', 'target_highres_landmarks.txt')
if os.path.isfile(targetLandmarksPath):
	targetLandmarks = readPoints(targetLandmarksPath)
	# targetLandmarks = [ [-point[0], point[1]] for point in targetLandmarks ] # is flip necessary for Visitron ?
	sourceLandmarks = readPoints(os.path.join(baseFolder, 'preImaging', 'source_landmarks.txt'))
	affWaferOverviewToLeica = fc.getModelFromPoints(sourceLandmarks, targetLandmarks).createAffine()
	angleWaferOverviewToLeica = Math.atan2(affWaferOverviewToLeica.getShearY(), affWaferOverviewToLeica.getScaleY()) # in radian
	IJ.log('angleWaferOverviewToLeica ' + str(angleWaferOverviewToLeica))
	refSectionOffset = angleWaferOverviewToLeica # (14.5 + 35) * PI/float(180) # maybe a 35 offset remaining ?
print refSectionOffset
print refSectionOffset * 180/float(PI)
# 8/0

effectiveChannels = beadChannels + ['-'.join(beadChannels)] # the raw channels plus the merged channel (no need to make all possible merge configurations, I just take the max merger)

mosaicX = int(width * mosaicLayout[0] - (mosaicLayout[0] - 1) * (overlap/100. * width))
mosaicY = int(height * mosaicLayout[1] - (mosaicLayout[1] - 1) * (overlap/100. * height))
IJ.log('The mosaic dimensions is ' + str(mosaicX) + ' ; ' + str(mosaicY))

templateMag = fc.readSectionCoordinates(os.path.join(baseFolder, 'preImaging', 'source_tissue_mag_description.txt'))[1]

safetyFactor = 1.2 # factor to extend the cropping box further to make sure that the section is not overcropped in case the FOV is not well centered on the section

templateMagBBox = sectionToPoly(templateMag).getBounds()

widthTemplateMag = templateMagBBox.width # The template mag is already turned: I can simply take the width of the bounding box
heightTemplateMag = templateMagBBox.height

# section crop parameters used when exporting the stitched images
widthCropMag = int(min(beadsToWaferFactor * widthTemplateMag, mosaicX) * safetyFactor) # the min is used for when the mosaic is smaller than the mag area
heightCropMag = int(min(beadsToWaferFactor * heightTemplateMag, mosaicY) * safetyFactor)

# Crop parameter used during the pairwise screening
matchingShrinkFactor = 1 # the bounding box for matching will actually be originalSize * safetyFactor * matchingShrinkFactor, currently almost 1 ...


if 'B6' in inputFolder:
	cropForMatching = [400, 400]
	# cropForMatching = [800, 800]
else:
	cropForMatching = [int(beadsToWaferFactor * widthTemplateMag * matchingShrinkFactor), int(beadsToWaferFactor * heightTemplateMag * matchingShrinkFactor)]
IJ.log('cropForMatching: ' + str(cropForMatching))

# Crop parameter used after alignment of a match: this is the box in which the events are counted
# similarityShrinkFactor = 0.8
similarityShrinkFactor = 0.8

if 'B6' in inputFolder:
	cropForSimilarity = [400, 400]
	cropForSimilarity = [800, 800]
	cropForSimilarity = [1200, 1200]
else:
	cropForSimilarity = [int(beadsToWaferFactor * widthTemplateMag * similarityShrinkFactor), int(beadsToWaferFactor * heightTemplateMag * similarityShrinkFactor)]
IJ.log('cropForSimilarity: ' + str(cropForSimilarity))
	
#######################
# 8-biting everything (optionally flipping if x-axis inversion)
#######################
firstSectionFolder = os.path.join(inputFolder, os.walk(inputFolder).next()[1][0])
firstImagePath = os.path.join(firstSectionFolder, os.walk(firstSectionFolder).next()[2][0])
im0 = IJ.openImage(firstImagePath)
bitDepth = im0.getBitDepth()
im0.close()
if bitDepth != 8:
	IJ.log('8-biting (optionally flipping if x-axis inversion)')
	downFactor, vFlip, hFlip = 1, False, False
	for channel in allChannels:
		theMeanMin, theMeanMax = 0, 0
		counter = 0
		imagePaths = []
		for id, sectionFolderName in enumerate(os.walk(inputFolder).next()[1]): # not using shutil.copytree as it yields an uncatchable error 20047
			sectionIndex = int(sectionFolderName.split('_')[1])
			for tileName in os.walk(os.path.join(inputFolder, sectionFolderName)).next()[2]:
				if os.path.splitext(tileName)[1] == '.tif':
					if ('channel_' + channel + '_') in tileName:
						imagePath = os.path.join(inputFolder, sectionFolderName, tileName)
						imagePaths.append(imagePath)
						
						im = IJ.openImage(imagePath)
						stats = im.getStatistics(Measurements.MIN_MAX)
						theMeanMax = theMeanMax + stats.max
						theMeanMin = theMeanMin + stats.min
						counter = counter + 1
						im.close()
		theMeanMax = int(theMeanMax/counter * 1.1)
		theMeanMin = int(theMeanMin/counter * 0.9)
		IJ.log('MinMax for channel ' + str(channel) + ' is ' + str([theMeanMin, theMeanMax]))
		
		atomicI = AtomicInteger(0)
		fc.startThreads(convertTo8Bit, fractionCores = 1, wait = 0, arguments = (atomicI, imagePaths, imagePaths, [theMeanMin, theMeanMax], downFactor, vFlip, hFlip))
#######################

#######################
# Creating the merged channel
#######################
mergedChannel = effectiveChannels[-1]
ic = ImageCalculator()

if len(filter(lambda x: 'channel_' + mergedChannel + '_' in x, [filename for root, dirnames, filenames in os.walk(inputFolder) for filename in filenames])) != nSections * mosaicLayout[0] * mosaicLayout[1]:
	IJ.log('Creating the merged channel ...')
	for id, sectionFolderName in enumerate(os.walk(inputFolder).next()[1]):
		sectionIndex = int(sectionFolderName.split('_')[1])
		for idX in range(mosaicLayout[0]):
			for idY in range(mosaicLayout[1]):
				imagePaths = [] # the paths of the images to merge
				imsToMerge = [] # the images to merge
				tileTag = str(idX).zfill(2) + '-' + str(idY).zfill(2)
				for tileName in os.walk(os.path.join(inputFolder, sectionFolderName)).next()[2]:
					for chan in beadChannels:
						if ('channel_' + chan + '_tileId_' + tileTag) in tileName:
							imagePaths.append(os.path.join(inputFolder, sectionFolderName, tileName))
				for imagePath in imagePaths:
					im = IJ.openImage(imagePath)
					imsToMerge.append(im)
				mergedIm = imsToMerge[0]
				for imToMerge in imsToMerge:
					mergedIm = ic.run('Max create', mergedIm, imToMerge)

				mergedPath = imagePaths[0].replace('channel_' + beadChannels[0] + '_', 'channel_' + mergedChannel + '_')
				IJ.save(mergedIm, mergedPath)
	IJ.log('Merged channel created...')
#####################################
# Computing the stitching (and rotating based on section orientation) transforms for the mosaics based on the reference stitching channel (e.g. brightfield) and exporting the sections (section = stitched mosaic)
#####################################
sectionAngles = getSectionsAngles(sectionsCoordinates)
mosaicAffineTransformsPath = os.path.join(baseFolder, 'stitchingTransforms')

worldSize = 5 * width
IJ.log('worldSize ' + str(worldSize))
boxOffset = 2 * width
IJ.log('boxOffset ' + str(boxOffset))

if (not os.path.isfile(mosaicAffineTransformsPath)):
	IJ.log('Computing the stitching and rotation transforms for the mosaics based on the reference stitching channel and the preImaging, respectively.')
	p, loader, layerset, nLayers = fc.getProjectUtils(fc.initTrakem(baseFolder, nSections))
	layerset.setDimensions(0, 0, worldSize, worldSize)
	for sectionFolderName in fc.naturalSort(os.walk(inputFolder).next()[1]):
		sectionFolder = os.path.join(inputFolder, sectionFolderName)
		sectionIndex = int(sectionFolderName.split('_')[1])
		IJ.log('Stitching/Rotating section ' + str(sectionIndex) + ' ...')
		layer = layerset.getLayers().get(sectionIndex)
		# rotationAff = AffineTransform().getRotateInstance(- sectionAngles[sectionIndex] + refSectionOffset, boxOffset + mosaicX/2., boxOffset +  mosaicY/2.)
		rotationAff = AffineTransform().getRotateInstance(sectionAngles[sectionIndex] + refSectionOffset, boxOffset + mosaicX/2., boxOffset +  mosaicY/2.)
		# rotationAff = AffineTransform().getRotateInstance(0) # no angle for debug

		if mosaicLayout != [1,1]: # stitching the layer with the stitchingChannel patches
			tileConfigurationPath = os.path.join(sectionFolder, 'TileConfiguration.registered.txt')
			if not os.path.isfile(tileConfigurationPath):
				IJ.run('Grid/Collection stitching', 'type=[Filename defined position] order=[Defined by filename         ] grid_size_x=' + str(mosaicLayout[0]) + ' grid_size_y=' + str(str(mosaicLayout[0])) + ' tile_overlap=' + str(overlap) + ' first_file_index_x=0 first_file_index_y=0 directory=' + sectionFolder + ' file_names=section_' + str(sectionIndex).zfill(4) + '_channel_' + stitchingChannel + '_tileId_{xx}-{yy}-mag.tif output_textfile_name=TileConfiguration.txt fusion_method=[Do not fuse images (only write TileConfiguration)] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap subpixel_accuracy computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory=' + sectionFolder)

			f = open(tileConfigurationPath, 'r')
			lines = f.readlines()[4:] # trimm the heading
			f.close()

			for line in lines:
				imPath = os.path.join(sectionFolder, line.replace('\n', '').split(';')[0])
				x = int(float(line.replace('\n', '').split(';')[2].split(',')[0].split('(')[1]))
				y = int(float(line.replace('\n', '').split(';')[2].split(',')[1].split(')')[0]))

				IJ.log('Inserting patch ' + str(imPath))
				patch = Patch.createPatch(p, imPath)
				layer.add(patch)
				patch.updateBucket()
				patch.setLocation(x + boxOffset, y + boxOffset)
				patch.updateBucket()
		else: # simply inserting the [1,1] patch in the project
			IJ.log('The mosaicLayout is actually only [1,1]')
			imName = 'section_' + str(sectionIndex).zfill(4) + '_channel_' + stitchingChannel + '_tileId_00-00-mag.tif'
			imPath = os.path.join(sectionFolder, imName)
			IJ.log('Inserting patch ' + str(imPath))
			patch = Patch.createPatch(p, imPath)
			patch.setLocation(boxOffset, boxOffset)
			layer.add(patch)
			patch.updateBucket()

		for patch in layer.getDisplayables(Patch): # it should work both with [1,1] and other mosaics
			currentAff = patch.getAffineTransform()
			IJ.log('currentAff' + str(currentAff))

			currentAff.preConcatenate(rotationAff)
			patch.setAffineTransform(currentAff)

	fc.writeAllAffineTransforms(p, mosaicAffineTransformsPath)
	p.saveAs(os.path.join(baseFolder, 'stitchingProject.xml'),True)
	fc.closeProject(p)

	with (open(mosaicAffineTransformsPath, 'r')) as f:
		transforms = f.readlines()
	
	if stitchingChannel in beadChannels:
		channelsToExport = effectiveChannels
	else:
		channelsToExport = effectiveChannels + [stitchingChannel]
	
	for channel in channelsToExport: # insert all channels with coordinates computed previously with the stitching and export
		p, loader, layerset, _ = fc.getProjectUtils(fc.initTrakem(baseFolder, nSections))
		layerset.setDimensions(0, 0, worldSize, worldSize)
		paths = []
		locations = []
		layers = []

		# for i in range(0, len(transforms), 8)[:5 * mosaicLayout[0] * mosaicLayout[1]]:
		for i in range(0, len(transforms), 8):
			alignedPatchPath = transforms[i]
			alignedPatchName = os.path.basename(alignedPatchPath)
			toAlignPatchName = alignedPatchName.replace('channel_' + stitchingChannel, 'channel_' + channel) # otherwise problem with section 488 ...
			toAlignPatchPath = os.path.join(os.path.dirname(alignedPatchPath), toAlignPatchName)
			toAlignPatchPath = toAlignPatchPath[:-1]  # why is there a trailing something !?
			IJ.log('toAlignPatchPath ' + toAlignPatchPath)

			l = int(transforms[i+1])
			paths.append(toAlignPatchPath)
			locations.append([0,0])
			layers.append(l)

		importFilePath = fc.createImportFile(baseFolder, paths, locations, layers = layers, name = 'channel_' + channel)
		IJ.log('Inserting all patches into a trakem project ...')
		task = loader.importImages(layerset.getLayers().get(0), importFilePath, '\t', 1, 1, False, 1, 0)
		task.join()
		layerset.setDimensions(0, 0, worldSize, worldSize)

		IJ.log('Applying the transforms to all patches ...')
		for i in range(0, len(transforms), 8):
			alignedPatchPath = transforms[i]
			alignedPatchName = os.path.basename(alignedPatchPath)
			toAlignPatchName = alignedPatchName.replace('channel_' + stitchingChannel, 'channel_' + channel)
			toAlignPatchPath = os.path.join(os.path.dirname(alignedPatchPath), toAlignPatchName)
			toAlignPatchPath = toAlignPatchPath[:-1]  # why is there a trailing something !?
			IJ.log('toAlignPatchPath ' + toAlignPatchPath)

			l = int(transforms[i+1])
			aff = AffineTransform([float(transforms[i+2]), float(transforms[i+3]), float(transforms[i+4]), float(transforms[i+5]), float(transforms[i+6]), float(transforms[i+7])])
			layer = layerset.getLayers().get(l)
			patches = layer.getDisplayables(Patch)
			thePatch = filter(lambda x: os.path.normpath(loader.getAbsolutePath(x)) == os.path.normpath(toAlignPatchPath), patches)[0]
			thePatch.setAffineTransform(aff)
			thePatch.updateBucket()

		fc.resizeDisplay(layerset)
		p.saveAs(os.path.join(baseFolder, 'exportingProject' + channel + '.xml'),True)
		Blending.blendLayerWise(layerset.getLayers(), True, None)

		for l, layer in enumerate(layerset.getLayers()):
			# # without cropping
			# cropRectangle = layerset.get2DBounds()

			# with cropping
			center = [layerset.get2DBounds().width/2., layerset.get2DBounds().height/2.]
			cropRectangle = Rectangle(int(center[0] - widthCropMag/2.), int(center[1] - heightCropMag/2.), widthCropMag, heightCropMag)

			im = loader.getFlatImage(layer, cropRectangle, 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layer.getAll(Patch), True, Color.black, None)
			# IJ.save(im, os.path.join(rawFolder, 'rawStitchedSection_' + channel + '_' + str(l).zfill(4) + '.tif'))
			IJ.save(im, os.path.join(rawFolder, 'stitchedRotatedSection_channel_' + channel + '_' + str(l).zfill(4) + '.tif'))
		fc.closeProject(p)

	# 8/0
	
#####################################
# Create the trakem project for large wafer overview
#####################################

sectionAngles = getSectionsAngles(sectionsCoordinates)
projectPath0 = os.path.join(baseFolder, 'waferOverviewProject_channel_' + beadChannels[0] + '.xml')

if not os.path.isfile(projectPath0):
	for channel in beadChannels:
		IJ.log('Creating the trakem project for large wafer overview')
		projectPath = os.path.join(baseFolder, 'waferOverviewProject_channel_' + channel + '.xml')
		
		p, loader, layerset, nLayers = fc.getProjectUtils(fc.initTrakem(baseFolder, 1))
		layer = layerset.getLayers().get(0)
		sectionPaths = [os.path.join(inputFolder, 'section_' + str(l).zfill(4), 'section_' + str(l).zfill(4) + '_channel_' + channel + '_tileId_00-00-mag.tif') for l in range(nSections)]

		maxX = 0
		maxY = 0
		
		for id, sectionPath in enumerate(sectionPaths):
			magCenterWafer = barycenter(sectionsCoordinates[id])
			aff = AffineTransform()
			
			rotationAff = AffineTransform().getRotateInstance(-refSectionOffset, width/2., height/2.)

			transX = magCenterWafer[0] * beadsToWaferFactor -width/2.
			transY = magCenterWafer[1] * beadsToWaferFactor -height/2.
			
			translationAff = AffineTransform().getTranslateInstance(transX, transY)

			maxX = max(maxX, transX)
			maxY = max(maxY, transY)
	
			aff.concatenate(translationAff)
			aff.concatenate(rotationAff)
			
			# a flip needs to be introduced because there is a flip between Visitron and ZeissZ1
			flippedSectionPath = os.path.splitext(sectionPath)[0] + '_flipped' + os.path.splitext(sectionPath)[1]
			
			im = IJ.openImage(sectionPath)
			IJ.run(im, 'Flip Horizontally', '')
			IJ.save(im, flippedSectionPath)
			im.close()
			
			IJ.log('Inserting patch ' + str(flippedSectionPath))
			patch = Patch.createPatch(p, flippedSectionPath)
			layer.add(patch)

			patch.setAffineTransform(aff)

			patch.updateBucket()
		
		layerset.setDimensions(0, 0, maxX * 1.1, maxY * 1.1)
		
		p.saveAs(projectPath, True)
		fc.closeProject(p)

8/0
#####################################
# Processing each channel
#####################################
for channel in effectiveChannels: # warning, channel is used as a global parameter to call functions
# for channel in [effectiveChannels[-1]]: # warning, channel is used as a global parameter to call functions
	IJ.log('Processing channel ' + channel)

	#######################
	# Defining paths
	#######################
	retrievalProjectPath = os.path.join(baseFolder, 'retrieval_Project_' + channel + '.xml') # trackem project

	SIFTMatchesPath = os.path.join(calculationFolder, 'SIFTMatches_' + channel)
	SIFTMatchesPicklePath = os.path.join(calculationFolder, 'SIFTMatchesPickle_' + channel)

	CCMatchesPath = os.path.join(calculationFolder, 'CCMatches_' + channel)
	CCMatchesPicklePath = os.path.join(calculationFolder, 'CCMatchesPickle_' + channel)

	CCCorrMatPath = os.path.join(calculationFolder, 'CCCorrMat_' + channel)
	CCCorrMatPicklePath = os.path.join(calculationFolder, 'CCCorrMatPickle_' + channel)

	CCCorrMatPathRaw = os.path.join(calculationFolder, 'highResCorrMatRaw_' + channel)
	CCCorrMatPicklePathRaw = os.path.join(calculationFolder, 'highResCorrMatPickleRaw_' + channel)

	SIFTOrderPath = os.path.join(calculationFolder, 'SIFTOrder_' + channel)
	SIFTOrderPicklePath = os.path.join(calculationFolder, 'SIFTOrderPickle_' + channel)

	sumSIFTOrderPath = os.path.join(calculationFolder, 'sumSIFTOrder_' + channel)
	sumSIFTOrderPicklePath = os.path.join(calculationFolder, 'sumSIFTOrderPickle_' + channel)

	CCOrderPath = os.path.join(calculationFolder, 'CCOrder_' + channel)
	CCOrderPicklePath = os.path.join(calculationFolder, 'CCOrderPickle_' + channel)

	CCOrderPathRaw = os.path.join(calculationFolder, 'CCOrderRaw_' + channel)
	CCOrderPicklePathRaw = os.path.join(calculationFolder, 'CCOrderPickleRaw_' + channel)

	SIFTReorderedProjectPath = os.path.join(baseFolder, 'SIFTReorderedProject_' + channel + '.xml')
	CCReorderedProjectPath = os.path.join(baseFolder, 'CCReorderedProject_' + channel + '.xml')

	affineAlignedSIFTReorderedProjectPath = os.path.normpath(os.path.join(baseFolder, 'affineAlignedSIFTReorderedProject_' + channel + '.xml'))
	affineAlignedCCReorderedProjectPath = os.path.join(baseFolder, 'affineAlignedCCReorderedProject_' + channel + '.xml')
	elasticAlignedCCReorderedProjectPath = os.path.join(baseFolder, 'elasticAlignedCCReorderedProject_' + channel + '.xml')

	# raw paths
	rawPaths = [os.path.join(rawFolder, 'stitchedRotatedSection_channel_' + str(channel) + '_' + str(id).zfill(4) + '.tif') for id in range(nSections)]
	rawProjectPath = os.path.join(baseFolder, 'raw_Project_' + channel + '.xml')
	SIFTReorderedRawProjectPath = os.path.join(baseFolder, 'SIFTReorderedRawProject_' + channel + '.xml')
	affineAlignedRawProjectPath = os.path.join(baseFolder, 'affineAlignedRawProject_' + channel + '.xml')

	# preprocessed paths
	preprocessedPaths = [os.path.join(preprocessedFolder, 'stitchedRotatedSection_channel_' + str(channel) + '_' + str(id).zfill(4) + '.tif') for id in range(nSections)]

	# blobized paths
	blobizedPaths = [os.path.join(blobizedFolder, 'stitchedRotatedSection_channel_' + str(channel) + '_' + str(id).zfill(4) + '.tif') for id in range(nSections)]
	blobizedProjectPath = os.path.join(baseFolder, 'blobized_Project_' + channel + '.xml')
	SIFTReorderedBlobizedProjectPath = os.path.join(baseFolder, 'SIFTReorderedBlobizedProject_' + channel + '.xml')
	affineAlignedBlobizedProjectPath = os.path.join(baseFolder, 'affineAlignedBlobizedProject_' + channel + '.xml')

	# forCC paths
	forCCPaths = [os.path.join(forCCFolder, 'stitchedRotatedSection_channel_' + str(channel) + '_' + str(id).zfill(4) + '.tif') for id in range(nSections)]


	# sum project paths
	sumProjectPath = os.path.join(baseFolder, 'sum_Project_' + channel + '.xml')
	SIFTReorderedSumProjectPath = os.path.join(baseFolder, 'SIFTReorderedSumProject_' + channel + '.xml')
	affineAlignedSumProjectPath = os.path.join(baseFolder, 'affineAlignedSumProject_' + channel + '.xml')

	# sum raw project paths
	sumRawProjectPath = os.path.join(baseFolder, 'sum_Raw_Project_' + channel + '.xml')
	SIFTReorderedSumRawProjectPath = os.path.join(baseFolder, 'SIFTReorderedSumRawProject_' + channel + '.xml')
	affineAlignedSumRawProjectPath = os.path.join(baseFolder, 'affineAlignedSumRawProject_' + channel + '.xml')

	# sum blobized project paths
	sumBlobizedProjectPath = os.path.join(baseFolder, 'sum_Blobized_Project_' + channel + '.xml')
	SIFTReorderedSumBlobizedProjectPath = os.path.join(baseFolder, 'SIFTReorderedSumBlobizedProject_' + channel + '.xml')
	affineAlignedSumBlobizedProjectPath = os.path.join(baseFolder, 'affineAlignedSumBlobizedProject_' + channel + '.xml')

	# getting list of paths to know whether parts of the script have already been executed
	# nRawCheckStitchedSections = len(filter(lambda x: 'rawStitchedSection_' + channel + '_' in x, os.listdir(rawCheckFolder))) # these sections are raw
	nStitchedSections = len(filter(lambda x: 'stitchedRotatedSection_channel_' + channel + '_' in x, os.listdir(preprocessedFolder))) # these sections are preprocessed
	nRawStitchedSections = len(filter(lambda x: 'stitchedRotatedSection_channel_' + channel + '_' in x, os.listdir(rawFolder))) # these sections are raw rotated
	nDogs = len(filter(lambda x: 'peaks_channel_' + channel + '_' in x, os.listdir(peaksFolder)))

	stitchedSectionPaths = [os.path.join(preprocessedFolder, 'stitchedRotatedSection_channel_' + channel + '_' + str(id).zfill(4) + '.tif') for id in range(nSections)]

	nBlobized = sum([os.path.isfile(blobizedPath) for blobizedPath in blobizedPaths])
	nforCC = sum([os.path.isfile(forCCPath) for forCCPath in forCCPaths])

	#######################
	# Copying current channel and preprocessing (imToPeak)
	#######################
	if len(filter(lambda x: 'channel_' + channel + '_' in x, [filename for root, dirnames, filenames in os.walk(inputFolder) for filename in filenames])) != len(filter(lambda x: 'channel_' + channel + '_' in x, [filename for root, dirnames, filenames in os.walk(preprocessedMosaicsFolder) for filename in filenames])): # is the number of copied files the same in the destination folder ?

		IJ.log('Copying input folder for preprocessing ...')
		imPaths = []
		for id, sectionFolderName in enumerate(os.walk(inputFolder).next()[1]): # not using shutil.copytree as it yields an uncatchable error 20047
			sectionIndex = int(sectionFolderName.split('_')[1])
			preprocessedSectionMosaicsFolder = fc.mkdir_p(os.path.join(preprocessedMosaicsFolder, sectionFolderName))
			IJ.log('B Copying section ' + str(id) + ' ...')
			for tileName in os.walk(os.path.join(inputFolder, sectionFolderName)).next()[2]:
				if 'channel_' + channel + '_' in tileName:
					sourcePath = os.path.join(inputFolder, sectionFolderName, tileName)
					targetPath = os.path.join(preprocessedSectionMosaicsFolder, tileName)
					shutil.copyfile(sourcePath, targetPath)
					imPaths.append(targetPath)

		IJ.log('Preprocessing the mosaics ...')
		atomicI = AtomicInteger(0)
		IJ.log('Preprocessing ...')
		fc.startThreads(preprocessImToPeak, fractionCores = 1, wait = 0, arguments = (imPaths, atomicI, 50, 50, 3, True, True, 2, [120,255]) ) #

	#######################
	# Stitching, rotating (according to section orientation), and exporting the stitched preprocessed sections to single files
	#######################
	# # My numbering is a bit inconsistent. Why would it happen that there are sections with names different from range(nSections) ?

	if nStitchedSections != nSections: #
		IJ.log('Stitching and rotating channel ' + channel)
		sectionAngles = getSectionsAngles(sectionsCoordinates)

		with (open(mosaicAffineTransformsPath, 'r')) as f:
			transforms = f.readlines()

		p, loader, layerset, _ = fc.getProjectUtils(fc.initTrakem(baseFolder, nSections))
		layerset.setDimensions(0, 0, worldSize, worldSize)
		paths = []
		locations = []
		layers = []

		for i in range(0, len(transforms), 8):
			alignedPatchPath = transforms[i] # D:\ThomasT\Thesis\A7\A7_REORDERING\A7_Beads_100\A7_100\section_0000\section_0000_channel_brightfield_tileId_00-00-mag.tif
			alignedPatchName = os.path.basename(alignedPatchPath) # section_0000_channel_brightfield_tileId_00-00-mag.tif
			sectionPrefix = alignedPatchName.split('_')[0] + '_' + alignedPatchName.split('_')[1]

			toAlignPatchPath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(alignedPatchPath))), 'preprocessedMosaics', sectionPrefix, alignedPatchName.replace('channel_' + stitchingChannel, 'channel_' + channel))
			toAlignPatchPath = toAlignPatchPath[:-1]  # why is there a trailing something !?
			IJ.log('toAlignPatchPath ' + toAlignPatchPath)

			l = int(transforms[i+1])
			paths.append(toAlignPatchPath)
			locations.append([0,0])
			layers.append(l)

		importFilePath = fc.createImportFile(baseFolder, paths, locations, layers = layers, name = 'channel_' + channel)
		IJ.log('Inserting all patches into a trakem project ...')
		task = loader.importImages(layerset.getLayers().get(0), importFilePath, '\t', 1, 1, False, 1, 0)
		task.join()
		layerset.setDimensions(0, 0, worldSize, worldSize)

		IJ.log('Applying the transforms to all patches ...')
		for i in range(0, len(transforms), 8):
			alignedPatchPath = transforms[i]
			alignedPatchName = os.path.basename(alignedPatchPath)
			sectionPrefix = alignedPatchName.split('_')[0] + '_' + alignedPatchName.split('_')[1]

			toAlignPatchPath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(alignedPatchPath))), 'preprocessedMosaics', sectionPrefix, alignedPatchName.replace('channel_' + stitchingChannel, 'channel_' + channel))
			toAlignPatchPath = toAlignPatchPath[:-1]  # why is there a trailing something !?
			IJ.log('os.path.normpath(toAlignPatchPath) ' + os.path.normpath(toAlignPatchPath))

			l = int(transforms[i+1])
			aff = AffineTransform([float(transforms[i+2]), float(transforms[i+3]), float(transforms[i+4]), float(transforms[i+5]), float(transforms[i+6]), float(transforms[i+7])])
			layer = layerset.getLayers().get(l)
			patches = layer.getDisplayables(Patch)
			IJ.log('list of patch names --- ' + str([os.path.normpath(loader.getAbsolutePath(x)) for x in patches]))
			thePatch = filter(lambda x: os.path.normpath(loader.getAbsolutePath(x)).replace(os.sep + os.sep, os.sep) == os.path.normpath(toAlignPatchPath), patches)[0]
			thePatch.setAffineTransform(aff)
			thePatch.updateBucket()

		fc.resizeDisplay(layerset)
		Blending.blendLayerWise(layerset.getLayers(), True, None)

		for l, layer in enumerate(layerset.getLayers()):
			# # without cropping
			# cropRectangle = layerset.get2DBounds()

			# with cropping
			center = [layerset.get2DBounds().width/2., layerset.get2DBounds().height/2.]
			cropRectangle = Rectangle(int(center[0] - widthCropMag/2.), int(center[1] - heightCropMag/2.), widthCropMag, heightCropMag)

			im = loader.getFlatImage(layer, cropRectangle, 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layer.getAll(Patch), True, Color.black, None)
			# # # # # IJ.run(im, 'Invert', '')
			# # # # # im = fc.minMax(im, 200, 200)
			IJ.save(im, os.path.join(preprocessedFolder, 'stitchedRotatedSection_channel_' + channel + '_' + str(l).zfill(4) + '.tif'))
		fc.closeProject(p)

	#######################
	# Creating base trakem project
	#######################
	if not os.path.isfile(retrievalProjectPath):
		IJ.log('Creating the trakEM project for channel ' + channel + ' with all stitched sections ...')
		p, loader, layerset, nLayers = fc.getProjectUtils(fc.initTrakem(baseFolder, nSections))
		p.saveAs(retrievalProjectPath, True)
		importFilePath = fc.createImportFile(baseFolder, stitchedSectionPaths, [[0,0]] * nSections, layers = range(nSections))
		loader.importImages(layerset.getLayers().get(0), importFilePath, '\t', 1, 1, False, 1, 0).join()
		p.save()
		time.sleep(5)
		fc.closeProject(p)
		time.sleep(5)
	#######################
	# Compute all peaks in parallel
	#######################
	if nDogs != nSections:
		p1, loader1, layerset1, nLayers1 = fc.getProjectUtils(fc.initTrakem(baseFolder, 1))
		if channel != effectiveChannels[-1]:
			IJ.log('Computing all dogs in parallel')
			atomicI = AtomicInteger(0)
			fc.startThreads(getPeaks, fractionCores = 0, arguments = (atomicI, stitchedSectionPaths)) # to check whether ok with more than 1 core ?
			threads = []
		else: # this is the merged channel, simply add the peaks from the single channels
			for i in range(nSections):
				allPeaks = []
				for beadChannel in beadChannels:
					peaks = loader1.deserialize(os.path.join(peaksFolder, 'peaks_channel_' + beadChannel + '_' + str(i).zfill(4)))			
					allPeaks = allPeaks + peaks
				loader1.serialize(allPeaks, os.path.join(peaksFolder, 'peaks_channel_' + channel + '_' + str(i).zfill(4)))
		fc.closeProject(p1)
		IJ.log('extracting dogs done')
	#######################
	# Create the blobized images
	#######################
	if nBlobized != nSections:

		newMosaicSize = getNewMosaicSize()

		### Load all peaks
		pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(baseFolder, 1))
		IJ.log('Loading all peaks ...')
		allPeaks = [ loaderZ.deserialize(os.path.join(peaksFolder, 'peaks_channel_' + channel + '_' + str(i).zfill(4))) for i in range(nSections)]
		# allPeaks = [loaderZ.deserialize(os.path.join(peaksFolder, name)) for name in fc.naturalSort([fileName for fileName in os.listdir(peaksFolder) if ('peaks_channel_' + channel + '_') in fileName])]
		IJ.log('All peaks have been loaded')
		fc.closeProject(pZ)
		IJ.log('allPeaks --- ' + str(len(allPeaks)))
		# IJ.log('allPeaks *** ' + str(allPeaks[:]))

		for id, sectionFolderName in enumerate(os.walk(preprocessedMosaicsFolder).next()[1]):
			sectionIndex = int(sectionFolderName.split('_')[1])
			blobizedPath = os.path.join(blobizedFolder, 'stitchedRotatedSection_channel_' + str(channel) + '_' + str(id).zfill(4) + '.tif')
			# IJ.log('newMosaicSize --- ' + str(newMosaicSize))
			# IJ.log('allPeaks[id]' + str(allPeaks[id]))
			blobizedIm = createBlobs(newMosaicSize[0], newMosaicSize[1], allPeaks[id])
			IJ.save(blobizedIm, blobizedPath)
			blobizedIm.close()
		del allPeaks
		
	# 8/0

	# #######################
	# # Create images for CC
	# #######################
	# IJ.log('forCC ?')
	# if nforCC != nSections:
		# newMosaicSize = getNewMosaicSize()
		# for id in range(nSections):
			# rawPath = os.path.join(rawFolder, 'stitchedRotatedSection_channel_' + str(channel) + '_' + str(id).zfill(4) + '.tif')
			# forCCPath = os.path.join(forCCFolder, 'stitchedRotatedSection_channel_' + str(channel) + '_' + str(id).zfill(4) + '.tif')
			# rawIm = IJ.openImage(rawPath)
			# forCCIm = fc.normLocalContrast(rawIm, 50, 50, 3, True, True)
			# forCCIm = fc.minMax(forCCIm, 150, 255)
			# # IJ.run(forCCIm, 'Median...', 'radius=' + str(2))
			# forCCIm = fc.blur(forCCIm, 3)
			# IJ.save(forCCIm, forCCPath)
			# forCCIm.close()
	
	#######################
	# Compute pairwise events
	#######################
	if not os.path.isfile(SIFTMatchesPath):
		IJ.log('Calculating low resolution matches')
		corrMat = initMat(nSections, initValue = 50000)
		affineDict = {} # writing to a dic is thread safe
		matchPointsDict = {}

		newMosaicSize = getNewMosaicSize()
		cropBBoxMatching = crop([newMosaicSize[0], newMosaicSize[1]], cropForMatching) # used for the pairwise initial screening
		cropBBoxSimilarity = crop([newMosaicSize[0], newMosaicSize[1]], cropForSimilarity) 

		pairs = []
		for id1 in range(nSections):
		# for id1 in [10,20,30,40,50]:
			for id2 in range(id1 + 1, nSections, 1):
				pairs.append([id1, id2])

		pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(baseFolder, 1))
		IJ.log('Loading allDogs ...')
		# allPeaks = [ loaderZ.deserialize(os.path.join(peaksFolder, 'peaks_' + channel + '_' + str(i).zfill(4))) for i in range(nSections)]
		allPeaks = [loaderZ.deserialize(os.path.join(peaksFolder, name)) for name in fc.naturalSort([fileName for fileName in os.listdir(peaksFolder) if ('peaks_channel_' + channel + '_') in fileName])] # sorry for whoever reads that
		IJ.log('All peaks have been loaded')

		allDogs = [pointsToDogs(peaks) for peaks in allPeaks]
		allCropedDogs = [pointsToDogs(cropPeaks(peaks, cropBBoxMatching)) for peaks in allPeaks]

		IJ.log('allDogs allCropedDogs allPeaks ' + str(len(allDogs)) + ', ' + str(len(allCropedDogs)) + ', ' + str(len(allPeaks)))

		atomN = AtomicInteger(0)
		fc.startThreads(getEvents, fractionCores = 0.9, arguments = [atomN, pairs, allPeaks, allCropedDogs, cropBBoxSimilarity, corrMat, affineDict, matchPointsDict])

		loaderZ.serialize([corrMat, affineDict, matchPointsDict], SIFTMatchesPath)
		fc.closeProject(pZ)
		pickleSave(matToList(corrMat), SIFTMatchesPicklePath)
	# 8/0
	# #######################
	# # Compute CC similarity matrix    
	# #######################
	
	# # with lock: # only one trakem working at a time
	
	# if not os.path.isfile(CCMatchesPath):
		# pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(baseFolder, 2))
		# lock = threading.Lock()    # for operations with the trakem project pZ
	
		# corrMat, affineDict, matchPointsDict = loaderZ.deserialize(SIFTMatchesPath)
		# IJ.log('The loaded SIFT corrMat: ' + str(corrMat))
		# ccMat = initMat(nSections, initValue = 50000)
		# atom = AtomicInteger(0)
		# print 'affineDict.keys()', affineDict.keys()
		# IJ.log('There are '  + str(len(affineDict.keys())) + ' pairs ')
		# # 8/0
		# # fc.startThreads(getCCCorrMat, fractionCores = 0, arguments = [atom, affineDict.keys(), affineDict, preprocessedPaths, ccMat])
		# # fc.startThreads(getCCCorrMat, fractionCores = 0, arguments = [atom, affineDict.keys(), affineDict, rawPaths, ccMat])
		# fc.startThreads(getCCCorrMat, fractionCores = 0, arguments = [atom, affineDict.keys(), affineDict, forCCPaths, ccMat])
		# # getHighResCorrMat(atom, affineDict.keys(), affineDict, preprocessedPaths, ccMat)        

		# IJ.log('The final ccMat: ' + str(ccMat))
		# loaderZ.serialize([ccMat, affineDict, matchPointsDict], CCMatchesPath)
		# pickleSave(matToList(ccMat), CCMatchesPicklePath)
		
		# loaderZ.serialize([ccMat, affineDict, matchPointsDict], SIFTMatchesPath)
		# pickleSave(matToList(ccMat), SIFTMatchesPicklePath)
		
		# fc.closeProject(pZ)
	# # 8/0
	#######################
	# Compute order
	#######################
	if not os.path.isfile(SIFTOrderPicklePath):
		IJ.log('Computing order ...')
		pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(baseFolder, 1))
		corrMat, affineDict, matchPointsDict = loaderZ.deserialize(SIFTMatchesPath)
		SIFTOrder, SIFTCosts = orderFromMat(corrMat, calculationFolder, solutionName = channel)
		pickleSave([SIFTOrder, SIFTCosts], SIFTOrderPicklePath)
		IJ.log('The SIFT order is: ' + str(SIFTOrder))
		fc.closeProject(pZ)
	# 8/0
	#######################
	# Save the corrMat as a 32-bit 0-1 image for Fiji 
	#######################
	corrMatImagePath = os.path.join(calculationFolder, 'corrMatImage_' + channel + '.tif')
	if not os.path.isfile(corrMatImagePath):	
		IJ.log('Saving the corrMat as a 32-bit 0-1 image for Fiji ...')
		pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(baseFolder, 1))        
		corrMat, affineDict, matchPointsDict = loaderZ.deserialize(SIFTMatchesPath)
		
		# symmetrize
		for a in range(nSections):
			for b in range(a, nSections):
				corrMat[b][a] = corrMat[a][b]

		SIFTOrder = pickleLoad(SIFTOrderPicklePath)[0]
		corrMat = reorderM(pythonToJamaMatrix(corrMat), SIFTOrder)

		im = ImagePlus('mm', FloatProcessor(nSections, nSections))
		ip = im.getProcessor()
		theArray = ip.getFloatArray()
		
		theMax = 0
		for a in range(nSections):
			for b in range(nSections):
				val = int(corrMat.get(a, b))
				if val != 50000:
					theMax = max(theMax, val)
		
		for x in range(nSections):
			for y in range(nSections):
				val = corrMat.get(x, y)
				if x == y:
					theArray[x][x] = 1
				elif int(val) == 50000:
					theArray[x][y] = Float.NaN
				else:
					theArray[x][y] = 1 - float(val/float(theMax))
					
		ip.setFloatArray(theArray)
		IJ.save(im, corrMatImagePath)
		fc.closeProject(pZ)
	
	#######################
	# Reorder and align project
	#######################
	if not os.path.isfile(SIFTReorderedProjectPath):
		IJ.log('Reordering the SIFT aligned project ...')
		SIFTOrder = pickleLoad(SIFTOrderPicklePath)[0]
		fc.reorderProject(retrievalProjectPath, SIFTReorderedProjectPath, SIFTOrder)
	if not os.path.isfile(affineAlignedSIFTReorderedProjectPath):
		IJ.log('Realigning the SIFT aligned project ...')
		affineRealignProject(SIFTReorderedProjectPath, affineAlignedSIFTReorderedProjectPath, SIFTMatchesPath)

	#######################
	# Create, reorder, and align the blobized project
	#######################
	if not os.path.isfile(blobizedProjectPath): # create
		IJ.log('Creating the trakEM blobized project for channel ' + channel + ' with all stitched blobized sections ...')
		p, loader, layerset, nLayers = fc.getProjectUtils(fc.initTrakem(baseFolder, nSections))
		p.saveAs(blobizedProjectPath, True)
		importFilePath = fc.createImportFile(baseFolder, blobizedPaths, [[0,0]] * nSections, layers = range(nSections))
		loader.importImages(layerset.getLayers().get(0), importFilePath, '\t', 1, 1, False, 1, 0).join()
		p.save()
		time.sleep(5)
		fc.closeProject(p)
		time.sleep(5)
	if not os.path.isfile(SIFTReorderedBlobizedProjectPath): # reorder
		IJ.log('Reordering the SIFT aligned blobized project ...')
		SIFTOrder = pickleLoad(SIFTOrderPicklePath)[0]
		fc.reorderProject(blobizedProjectPath, SIFTReorderedBlobizedProjectPath, SIFTOrder)
	if not os.path.isfile(affineAlignedBlobizedProjectPath): # align
		IJ.log('Realigning the SIFT aligned blobized project ...')
		affineRealignProject(SIFTReorderedBlobizedProjectPath, affineAlignedBlobizedProjectPath, SIFTMatchesPath)

	#######################
	# Create, reorder, and align the raw project
	#######################
	if not os.path.isfile(rawProjectPath): # create
		IJ.log('Creating the trakEM raw project for channel ' + channel + ' with all stitched raw sections ...')
		p, loader, layerset, nLayers = fc.getProjectUtils(fc.initTrakem(baseFolder, nSections))
		p.saveAs(rawProjectPath, True)
		importFilePath = fc.createImportFile(baseFolder, rawPaths, [[0,0]] * nSections, layers = range(nSections))
		loader.importImages(layerset.getLayers().get(0), importFilePath, '\t', 1, 1, False, 1, 0).join()
		p.save()
		time.sleep(5)
		fc.closeProject(p)
		time.sleep(5)
	if not os.path.isfile(SIFTReorderedRawProjectPath): # reorder
		IJ.log('Reordering the SIFT aligned raw project ...')
		SIFTOrder = pickleLoad(SIFTOrderPicklePath)[0]
		fc.reorderProject(rawProjectPath, SIFTReorderedRawProjectPath, SIFTOrder)
	if not os.path.isfile(affineAlignedRawProjectPath): # align
		IJ.log('Realigning the SIFT aligned raw project ...')
		affineRealignProject(SIFTReorderedRawProjectPath, affineAlignedRawProjectPath, SIFTMatchesPath)


#######################
# Calculate the sum order
#######################
# # # print 555, fc
# # # pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(baseFolder, 1))
# # # fc.closeProject(pZ)

if not os.path.isfile(sumSIFTOrderPicklePath):
# if True:
	IJ.log('Calculating the sumChannel event order')
	pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(baseFolder, 1))
	
	# Average of the matrices
	allMats = []
	for channel in beadChannels:
		SIFTMatchesPath = os.path.join(calculationFolder, 'SIFTMatches_' + channel)
		[corrMat, affineDict, matchPointsDict] = loaderZ.deserialize(SIFTMatchesPath)
		allMats.append(corrMat)

	sumMat = matAverage(allMats)
		
	sumOrder, sumCosts = orderFromMat(sumMat, calculationFolder, 'sum')
	pickleSave([sumOrder, sumCosts], sumSIFTOrderPicklePath)
	IJ.log('The sum order is: ' + str(sumOrder))
	fc.closeProject(pZ)
# 8/0
# # # # #######################
# # # # # Two-steps reordering: recalculate the order comparing only the 5-neighborhood with all peaks (large crop)
# # # # #######################	
# # # # if twoStepsReordering: #	
	# # # # [sumOrder, sumCosts] = pickleLoad(sumSIFTOrderPicklePath)	
	
	# # # # if 'B6' in inputFolder:
		# # # # channel = '488'
		# # # # SIFTMatchesPath = os.path.join(calculationFolder, 'SIFTMatches_2ndStep_' + channel)
		# # # # SIFTMatchesPicklePath = os.path.join(calculationFolder, 'SIFTMatchesPickle_2ndStep_' + channel)
		# # # # # SIFTOrderPicklePath = os.path.join(calculationFolder, 'SIFTOrderPickle_2ndStep_' + channel) # actually no, I can simply override the sum order
		
		# # # # cropBoxFactor = 3
		# # # # cropForMatching = [cropBoxFactor * cropForMatching[0], cropBoxFactor * cropForMatching[0]] # use a larger cropping window with many beads
		# # # # cropForSimilarity = [cropBoxFactor * cropForSimilarity[0], cropBoxFactor * cropForSimilarity[1]]
		# # # # neighborhood = 5
	
	# # # # corrMat = initMat(nSections, initValue = 50000)
	# # # # affineDict = {} # writing to a dic is thread safe
	# # # # matchPointsDict = {}

	# # # # newMosaicSize = getNewMosaicSize()
	# # # # cropBBoxMatching = crop([newMosaicSize[0], newMosaicSize[1]], cropForMatching) # used for the pairwise initial screening
	# # # # cropBBoxSimilarity = crop([newMosaicSize[0], newMosaicSize[1]], cropForSimilarity)

	# # # # pairs = []
	# # # # for id1 in range(nSections):
		# # # # for id2 in range(id1 + 1, min(id1 + 1 + neighborhood, nSections), 1): # /!\ make only pairs in the neighborhood
			# # # # pairs.append([sumOrder[id1], sumOrder[id2]])

	# # # # pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(baseFolder, 1))
	# # # # IJ.log('Loading allDogs ...')
	# # # # # allPeaks = [ loaderZ.deserialize(os.path.join(peaksFolder, 'peaks_' + channel + '_' + str(i).zfill(4))) for i in range(nSections)]
	# # # # allPeaks = [loaderZ.deserialize(os.path.join(peaksFolder, name)) for name in fc.naturalSort([fileName for fileName in os.listdir(peaksFolder) if ('peaks_channel_' + channel + '_') in fileName])] # sorry for whoever reads that
	# # # # IJ.log('All peaks have been loaded')

	# # # # allDogs = [pointsToDogs(peaks) for peaks in allPeaks]
	# # # # allCropedDogs = [pointsToDogs(cropPeaks(peaks, cropBBoxMatching)) for peaks in allPeaks]
	# # # # print 'cropForMatching', cropForMatching

	# # # # IJ.log('allDogs allCropedDogs allPeaks ' + str(len(allDogs)) + ', ' + str(len(allCropedDogs)) + ', ' + str(len(allPeaks)))

	# # # # atomN = AtomicInteger(0)
	# # # # # fc.startThreads(getEvents, fractionCores = 0.9, arguments = [atomN, pairs, allPeaks, allCropedDogs, cropBBoxSimilarity, corrMat, affineDict, matchPointsDict])

	# # # # # loaderZ.serialize([corrMat, affineDict, matchPointsDict], SIFTMatchesPath)
	# # # # fc.closeProject(pZ)
	# # # # # pickleSave(matToList(corrMat), SIFTMatchesPicklePath)

	# # # # #######################
	# # # # # Compute 2nd step order
	# # # # #######################
	# # # # # if os.path.isfile(sumSIFTOrderPicklePath):
	# # # # IJ.log('Computing order in 2nd step...')
	# # # # pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(baseFolder, 1))
	# # # # corrMat, affineDict, matchPointsDict = loaderZ.deserialize(SIFTMatchesPath)
	# # # # SIFTOrder, SIFTCosts = orderFromMat(corrMat, calculationFolder, solutionName = channel)
	# # # # pickleSave([SIFTOrder, SIFTCosts], sumSIFTOrderPicklePath)
	# # # # IJ.log('The SIFT order is: ' + str(SIFTOrder))
	# # # # fc.closeProject(pZ)

#######################
# Create, reorder, and align the sum project
#######################
if not os.path.isfile(affineAlignedSumProjectPath):
	IJ.log('Creating, reordering, and aligning the sum project')
	IJ.log('Creating ... ')
	shutil.copyfile(os.path.join(baseFolder, 'retrieval_Project_' + effectiveChannels[-1] + '.xml'), sumProjectPath)
	time.sleep(3)
	p, loader, layerset, nLayers = fc.openTrakemProject(sumProjectPath)
	time.sleep(3)
	print p.saveAs(sumProjectPath, True)
	time.sleep(3)
	fc.closeProject(p)
	
	IJ.log('Reordering ...')
	[sumOrder, sumCosts] = pickleLoad(sumSIFTOrderPicklePath)
	fc.reorderProject(sumProjectPath, SIFTReorderedSumProjectPath, sumOrder)

	IJ.log('Affine aligning ...')
	mergedChannelMatchesPath = os.path.join(calculationFolder, 'SIFTMatches_' + effectiveChannels[-1])
	singleChannelMatchesPaths = [os.path.join(calculationFolder, 'SIFTMatches_' + channel) for channel in beadChannels]

	affineRealignProject(SIFTReorderedSumProjectPath, affineAlignedSumProjectPath, mergedChannelMatchesPath, optionalMatchesPath = singleChannelMatchesPaths)
	
#######################
# Create, reorder, and align the sum raw project (using the affine transforms of the mergedChannel)
# WARNING: it seems to hang after Creating ...
#######################
if not os.path.isfile(affineAlignedSumRawProjectPath):
	IJ.log('Creating, reordering, and aligning the sum project')
	IJ.log('Creating ... ')

	shutil.copyfile(os.path.join(baseFolder, 'raw_Project_' + effectiveChannels[-1] + '.xml'), sumRawProjectPath)
	p, loader, layerset, nLayers = fc.openTrakemProject(sumRawProjectPath)
	time.sleep(2)
	p.saveAs(sumRawProjectPath, True)
	time.sleep(2)
	fc.closeProject(p)
	time.sleep(2)
	
	IJ.log('Reordering ...')
	[sumOrder, sumCosts] = pickleLoad(sumSIFTOrderPicklePath)
	fc.reorderProject(sumRawProjectPath, SIFTReorderedSumRawProjectPath, sumOrder)

	IJ.log('Affine aligning ...')
	mergedChannelMatchesPath = os.path.join(calculationFolder, 'SIFTMatches_' + effectiveChannels[-1])
	singleChannelMatchesPaths = [os.path.join(calculationFolder, 'SIFTMatches_' + channel) for channel in beadChannels]
	affineRealignProject(SIFTReorderedSumRawProjectPath, affineAlignedSumRawProjectPath, mergedChannelMatchesPath, optionalMatchesPath = singleChannelMatchesPaths)
	
#######################
# Create the blobized merged images (should look different compared to the merged blobized images)
#######################
if len(filter(lambda x: 'stitchedRotatedSection_Merged_' in x, os.listdir(blobizedFolder))) != nSections:
	IJ.log('Create the blobized merged images')
	newMosaicSize = getNewMosaicSize()
	pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(baseFolder, 1))
	for l in range(nSections):
		mergedBeads = []
		for channel in beadChannels:
			mergedBeads = mergedBeads + loaderZ.deserialize(os.path.join(peaksFolder, 'peaks_channel_' + str(channel) + '_' + str(l).zfill(4)))
		# IJ.log('mergedBeads --- ' + str(mergedBeads))

		blobizedPath = os.path.join(blobizedFolder, 'stitchedRotatedSection_Merged_' + str(effectiveChannels[-1]) + '_' + str(l).zfill(4) + '.tif')
		blobizedIm = createBlobs(newMosaicSize[0], newMosaicSize[1], mergedBeads)
		IJ.save(blobizedIm, blobizedPath)
		blobizedIm.close()
	fc.closeProject(pZ)
	del mergedBeads
	
#######################
# Compute the consecutive affine transforms on the sumReordered merged blobized images    
#######################
sumAffineFolder = fc.mkdir_p(os.path.join(baseFolder, 'sumAffine'))
if len(os.listdir(sumAffineFolder)) != (nSections-1):
	IJ.log('Compute the affine transforms on the sumReordered merged blobized images')
	[sumOrder, sumCosts] = pickleLoad(sumSIFTOrderPicklePath)
	pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(baseFolder, 1))

	for i, index1 in enumerate(sumOrder[:-1]):
		index2 = sumOrder[i+1]

		mergedBeads1 = []
		mergedBeads2 = []
		for channel in beadChannels:
			mergedBeads1 = mergedBeads1 + loaderZ.deserialize(os.path.join(peaksFolder, 'peaks_channel_' + str(channel) + '_' + str(index1).zfill(4)))
			mergedBeads2 = mergedBeads2 + loaderZ.deserialize(os.path.join(peaksFolder, 'peaks_channel_' + str(channel) + '_' + str(index2).zfill(4)))
		
		dogsContainer = ArrayList()
		dogs1 = pointsToDogs(mergedBeads1)
		dogs2 = pointsToDogs(mergedBeads2)
		dogsContainer.add(dogs1)
		dogsContainer.add(dogs2)
		comparePairs = Matching.descriptorMatching(dogsContainer, 2, dp, 0)
		inliers = comparePairs[0].inliers
		affineT = AffineTransform()
		affineT.setToIdentity()
		
		if len(inliers) > 0: # it should because it is sum reordered
			affineT = comparePairs[0].model.createAffine().createInverse()
		else:
			IJ.log('*** WARNING *** No matching found in the separately merged channel for pair (' + str(index1) + ',' + str(index2) + ')')
			# is there a matching in the single channels ?
				# lowResMatchesPickle_488-546 # the matches have been saved, should be easy to look it up
			# if not, try to loosen the matching parameters
			
			thereIsATransform = False
			for channel in reversed(effectiveChannels): # reversed to start with the merged channel which is likely to have a better transform
				if not thereIsATransform:
					SIFTMatchesPath = os.path.join(calculationFolder, 'SIFTMatches_' + channel)
					corrMat, affineDict, matchPointsDict = loaderZ.deserialize(SIFTMatchesPath)
					if (index1, index2) in affineDict:
						IJ.log(str((index1, index2)) + ' found in channel ' + str(channel))
						thereIsATransform = True
						affineT = affineDict[(index1, index2)].createInverse()
					elif (index2, index1) in affineDict:
						IJ.log(str((index2, index1)) + ' found in channel ' + str(channel))
						thereIsATransform = True
						affineT = affineDict[(index2, index1)]
			if not thereIsATransform: #still no transform found
				IJ.log('*** Urg *** Absolutely no matching found for pair (' + str(index1) + ',' + str(index2) + '). A manual affine would be needed.')
				
		loaderZ.serialize(affineT, os.path.join(sumAffineFolder, 'sumAffine_' + str(i).zfill(4))) # write a transform in any case, even if no match is found, in which case an identity is written
			
	fc.closeProject(pZ)
#######################
# Create sumAffineRealigned projects
#######################
[sumOrder, sumCosts] = pickleLoad(sumSIFTOrderPicklePath)
consecAffineTransformPaths = [os.path.join(sumAffineFolder, 'sumAffine_' + str(l).zfill(4)) for l in range(nSections-1)]
for channel in effectiveChannels:
	projectPath = os.path.join(baseFolder, 'sumAffineRealignedRaw_' + channel + '.xml')
	orderedImagePaths = [os.path.join(rawFolder, 'stitchedRotatedSection_channel_' + channel + '_' + str(sumOrder[l]).zfill(4) + '.tif') for l in range(nSections)]
	if not os.path.isfile(projectPath):
		sumAffineProject(projectPath, orderedImagePaths, consecAffineTransformPaths)

for channel in effectiveChannels:
	projectPath = os.path.join(baseFolder, 'sumAffineRealignedBlobized_' + channel + '.xml')
	orderedImagePaths = [os.path.join(blobizedFolder, 'stitchedRotatedSection_channel_' + channel + '_' + str(sumOrder[l]).zfill(4) + '.tif') for l in range(nSections)]
	if not os.path.isfile(projectPath):
		sumAffineProject(projectPath, orderedImagePaths, consecAffineTransformPaths)

for channel in effectiveChannels:
	projectPath = os.path.join(baseFolder, 'sumAffineRealigned_' + channel + '.xml')
	orderedImagePaths = [os.path.join(preprocessedFolder, 'stitchedRotatedSection_channel_' + channel + '_' + str(sumOrder[l]).zfill(4) + '.tif') for l in range(nSections)]
	if not os.path.isfile(projectPath):
		sumAffineProject(projectPath, orderedImagePaths, consecAffineTransformPaths)
		
IJ.log('SOR has run entirely.')