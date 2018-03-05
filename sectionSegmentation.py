# doing simple morpghological operations might be easier than WEKA ...

# What do I want exactly ?
	# approximate locations are ok
	# CCs are ok, simply need to fine tune a bit more
	
	# make a double CC on the two channels, and use a weight a.BF + (1-a).Fluo
	
	# --preprocess correctly at the beginning
	# --blend the BF channel
	# --get the edge channel yes, that is a useful channel
	
	# look for salient point in corner neighborhood ?
		# (Corners are too difficult, depends on block shape ?)

	
# Fiji script that aligns the patches of the silicon wafer overview. It outputs subpatches x_y_imageName (normal, thesholded, edges). The overlap of the subpatches depends on the size of the template.

# # ToDos 05/2017
# --user prompt to offset the channels
	# then insert the dapi channel too
# --activate automated montage
# manual section at the end should give the real section, not the mag ? tant qu'a faire, the user gives the exact section ...
	# but it is annoying, because the affine mechanism is in the preImaging script, so that either I should transfer the affine mechanism into this first script or I should keep track of which sections are mag sections and which sections are real sections

# /!\ -- actually I need the affine mechanism in this script because I need to assess the orientation of the sections
	# manually ?
		# the affine mechanism is almost there. I can already transform sections. I need to find an affine given source and target points. It most probably already exists.
	# asymmetric trimming ?
		# difficult because otherwise the section terminates with mag resin

# - understand why so many sections missing that should be easy to get
	# -- 0.95 was clearly too high for the area threshold
	# clustering issue also ?
# weka on the brightfield to find the orientation or manual entry with a key press ?
	# without systematic detachment it might not be easy to weka the orientation ...
	# compare the quantity of edges on the two sides ... should be rather robust
# final GUI: export the sections to landmarks, adjust the landmarks and add new ones 

from __future__ import with_statement
import os, time, pickle, shutil

# import subprocess # currently broken, use os.system instead
from java.lang import ProcessBuilder

import threading
import ij
from ij import IJ, ImagePlus, WindowManager
from ij.gui import Roi, PolygonRoi, PointRoi, WaitForUserDialog
from ij.process import ImageStatistics, ImageProcessor
from ij.measure import Measurements
from ij.plugin import ImageCalculator
from ij.plugin.frame import RoiManager
from ij.plugin.tool import RoiRotationTool

from java.awt import Frame
from java.lang import Double

import fijiCommon as fc

from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch, Display, AreaList, Displayable
from ini.trakem2.imaging import StitchingTEM
from ini.trakem2.imaging import Blending
from ini.trakem2.imaging.StitchingTEM import PhaseCorrelationParam
from ij.plugin.filter import ParticleAnalyzer

from mpicbg.imglib.algorithm.correlation import CrossCorrelation
from mpicbg.trakem2.align import Align, AlignTask
from mpicbg.imglib.image import ImagePlusAdapter

from java.awt import Rectangle, Color, Polygon
from java.awt.geom import Area, AffineTransform
from java.awt.event import MouseAdapter, KeyAdapter, KeyEvent
from java.lang import Math, Runtime
from java.lang.Math import hypot, sqrt, atan2, PI, abs
from java.util.concurrent.atomic import AtomicInteger

from jarray import zeros, array

from trainableSegmentation import WekaSegmentation

from xml.dom import minidom

from operator import itemgetter


#########################################################################	
# BEGIN README
#########################################################################	
# 1. Start Fiji
# 2. Run this script in the Script Editor of Fiji (setting language to python)

#########################################################################	
# END README
#########################################################################


def xlim(a, lbb):
	return max(min(a, lbb.width), 0)

def ylim(a, lbb):
	return max(min(a, lbb.height), 0)

colors = [Color.red, Color.blue, Color.green, Color.yellow, Color.cyan, Color.magenta, Color.orange]

def convertTo8BitAndResize(imagePaths, newImagePaths, downFactor, atomicI):
	while atomicI.get() < len(imagePaths):
		k = atomicI.getAndIncrement()
		if (k < len(imagePaths)):
			imagePath = imagePaths[k]
			newImagePath = newImagePaths[k]
			im = IJ.openImage(imagePath)
			IJ.run(im, '8-bit', '')
			# if 'BF' in os.path.basename(newImagePath): # normalize only the BF channel
				# fc.normLocalContrast(im, 500, 500, 3, True, True)
			im = fc.resize(im, float(1/float(downFactor)))
			IJ.save(im, newImagePath)
			IJ.log(str(k) + ' of ' + str(len(imagePaths)) + ' processed')
			im.close()

	
#########################
### TrakEM2 operations
#########################
def addLandmarkOverlays(project, landmarks):
	layerset = project.getRootLayerSet()
	layer = layerset.getLayers().get(0)
	layerId = layer.getId()
	arealists = []
	for l, landmark in enumerate(landmarks):
		ali = AreaList(project, 'landmark' + '_' + str(l), 0, 0)
		layerset.add(ali)
		lbb = layerset.get2DBounds()
		
		s = 500
		sw = 30
		
		poly = Polygon(map(lambda x: xlim(x, lbb), [landmark[0]-s, landmark[0]+s, landmark[0]+s, landmark[0]-s]), map(lambda x: ylim(x, lbb), [landmark[1]-sw, landmark[1]-sw, landmark[1]+sw, landmark[1]+sw]), 4)
		ali.addArea(layerId, Area(poly))
		poly = Polygon(map(lambda x: xlim(x, lbb), [landmark[0]-sw, landmark[0]+sw, landmark[0]+sw, landmark[0]-sw]), map(lambda x: ylim(x, lbb), [landmark[1]-s, landmark[1]-s, landmark[1]+s, landmark[1]+s]), 4)
		ali.addArea(layerId, Area(poly))

		ali.alpha = 0.5
		ali.color = colors[l%len(colors)]
		ali.visible = True
		ali.locked = False
		ali.calculateBoundingBox(layer)
		arealists.append(ali)
		ali.updateBucket()
		project.getProjectTree().insertSegmentations([ali])

		displays = Display.getDisplays()
		if displays.isEmpty():
			disp = Display(project, layer)
		else:
			disp = displays[0]
		disp.repaint()
		project.getLayerTree().updateList(layerset)
		layer.recreateBuckets()

		factor = 1
		canvas = disp.getCanvas()
		disp.repaint()

		# disp.show(layer, ali, False, False)
		Display.showCentered(layer, ali, False, False)
		disp.repaint()

		w = 500
		h = 500
		bb = Rectangle(int(round(landmark[0]-w/2)), int(round(landmark[1]-h/2)), w, h)
		bb = bb.createIntersection(layerset.get2DBounds())
		im = project.getLoader().getFlatImage(layer, bb, factor, 0x7fffffff, ImagePlus.COLOR_RGB, Displayable, True) 
		IJ.save(im, os.path.join(preImagingFolder, 'landmark_' + str(l) + '_zoom_1.png'))
	
		w = 5000
		h = 5000
		ali.alpha = 0.8
		bb = Rectangle(int(round(landmark[0]-w/2)), int(round(landmark[1]-h/2)), w, h)
		bb = bb.createIntersection(layerset.get2DBounds())
		disp.repaint()
		im = project.getLoader().getFlatImage(layer, bb, 0.5 * factor, 0x7fffffff, ImagePlus.COLOR_RGB, Displayable, True) 
		IJ.save(im, os.path.join(preImagingFolder, 'landmark_' + str(l) + '_zoom_2.png'))

		# # # w = 15000
		# # # h = 15000
		# # # ali.alpha = 1
		# # # bb = Rectangle(int(landmark[0]-w/2), int(landmark[1]-h/2), w, h)
		# # # bb = bb.createIntersection(layerset.get2DBounds())
		# # # disp.repaint()
		# # # im = project.getLoader().getFlatImage(layer, bb, 0.03 * factor, 0x7fffffff, ImagePlus.COLOR_RGB, Displayable, True) 
		# # # IJ.save(im, os.path.join(preImagingFolder, 'landmark_' + str(l) + '_zoom_3.png'))
		# # # ali.visible = False

	for ali in arealists:
		ali.visible = True
	im = project.getLoader().getFlatImage(layer, layerset.get2DBounds(), 0.5 * factor, 0x7fffffff, ImagePlus.COLOR_RGB, Displayable, True) 
	IJ.save(im, os.path.join(preImagingFolder, 'allLandmarks.png'))

	# project.save()
		
def forceAlphas(layerset):
	for ali in layerset.getZDisplayables(AreaList):
		if int(ali.getFirstLayer().getZ()) == 0:
			ali.alpha = 0.5
		elif int(ali.getFirstLayer().getZ()) == 1:
			ali.alpha = 1

# def addSectionOverlays(project, layers, sections, colors, alphas, name):
	# layerset = project.getRootLayerSet()
	# for l in layers:
		# layer = layerset.getLayers().get(l)
		# layerId = layer.getId()
		# segmentations = []
		# ali = AreaList(project, name + '_' + str(l), 0, 0)
		# layerset.add(ali)
		# for id, section in enumerate(sections):
			# ali.addArea(layerId, Area(sectionToPoly(section)))
		# ali.alpha = alphas[l]
		# ali.color = colors[l]
		# ali.visible = True
		# ali.locked = False
		# # ali.setColor(colors[l])
		# # ali.setAlpha(alphas[l])
		# ali.calculateBoundingBox(None)
		# print 'alpha', ali.getAlpha(), 'layer', l
		# ali.updateBucket()
		# segmentations.append(ali)
		# project.getProjectTree().insertSegmentations([ali])
	
		# displays = Display.getDisplays()
		# if displays.isEmpty():
			# disp = Display(project, layer)
		# else:
			# disp = displays[0]
		# disp.repaint()
		# # project.getProjectTree().insertSegmentations(segmentations)
			
		# project.getLayerTree().updateList(layerset)
		# layer.recreateBuckets()

def addSectionOverlays(project, layers, sections, colors, alphas, name):
	layerset = project.getRootLayerSet()
	for l in layers:
		layer = layerset.getLayers().get(l)
		layerId = layer.getId()
		segmentations = []
		for id, section in enumerate(sections):
			ali = AreaList(project, name + '_' + str(id), 0, 0)
			layerset.add(ali)
			ali.addArea(layerId, Area(sectionToPoly(section)))
			ali.alpha = alphas[l]
			ali.color = colors[l]
			ali.visible = True
			ali.locked = False
			# ali.setColor(colors[l])
			# ali.setAlpha(alphas[l])
			ali.calculateBoundingBox(None)
			print 'alpha', ali.getAlpha(), 'layer', l
			ali.updateBucket()
			segmentations.append(ali)
		project.getProjectTree().insertSegmentations(segmentations)
	
		displays = Display.getDisplays()
		if displays.isEmpty():
			disp = Display(project, layer)
		else:
			disp = displays[0]
		disp.repaint()
		# project.getProjectTree().insertSegmentations(segmentations)
			
		project.getLayerTree().updateList(layerset)
		layer.recreateBuckets()
		
		
		
def createImportFile(folder, paths, locations, factor = 1, layer = 0):
	importFilePath = os.path.join(folder, 'trakemImportFile.txt')
	with open(importFilePath, 'w') as f:
		for id, path in enumerate(paths):
				xLocation = int(round(locations[id][0] * factor))
				yLocation = int(round(locations[id][1] * factor))
				IJ.log('Inserting image ' + path + ' at (' + str(xLocation) + ' ; ' + str(yLocation) + ')' )
				f.write(str(path) + '\t' + str(xLocation) + '\t' + str(yLocation) + '\t' + str(layer) + '\n')
	return importFilePath

def getPointsFromUser(project, l, fov = None, text = 'Select points'):
	points = None
	layerset = project.getRootLayerSet()
	layer = layerset.getLayers().get(l)
	displays = Display.getDisplays()
	if displays.isEmpty():
		disp = Display(project, layer)
	else:
		disp = displays[0]
	disp.repaint()
	disp.showFront(layer)
	# print 'disp.getCanvas().getMagnification()', disp.getCanvas().getMagnification()
	# disp.getCanvas().center(Rectangle(int(round(fov[0])), int(round(fov[1])), int(round(effectivePatchSize)), int(round(effectivePatchSize))), 0.75)
	WaitForUserDialog(text).show()
	roi = disp.getRoi()
	if roi:
		poly = disp.getRoi().getPolygon()
		points = [list(a) for a in zip(poly.xpoints, poly.ypoints)]
	disp.getCanvas().getFakeImagePlus().deleteRoi()
	disp.update(layerset)
	return points

def getTemplates(project):
	layer = project.getRootLayerSet().getLayers().get(0) # the brightfield layer
	disp = Display(project, layer)
	disp.showFront(layer)

	WaitForUserDialog('Select in order : A. 4 corners of a section. B. 4 corners of the mag region of the same section. Then click OK.').show()
	
	poly = disp.getRoi().getPolygon()
	X = poly.xpoints
	Y = poly.ypoints
	nSections = len(X)/8 # the script is general enough to let the user give more sections, but only one used here ...
	
	print 'X', X
	print 'Y', Y
	print 'nSections', nSections

	# # # From the old pipeline, probably not useful any more
	# # 1. Determining the size of the subpatches
	# sectionExtent = longestDiagonal([ [X[0], Y[0]] , [X[1], Y[1]], [X[2], Y[2]], [X[3], Y[3]] ])
	# patchSize = int(round(3 * sectionExtent))
	# overlap = int(round(1 * sectionExtent))
	# effectivePatchSize = patchSize - overlap
	# print 'effectivePatchSize', effectivePatchSize
	# writePoints(patchSizeAndOverlapFullResPath, [[patchSize, overlap]])
	# writePoints(patchSizeAndOverlapLowResPath, [[patchSize/float(downsizingFactor), overlap/float(downsizingFactor)]])
	
	templateSections = []
	templateMags = []
	userTemplateInput = [] # the points the user gave: will be used to create template with real images
	
	for s in range(nSections): # the script is general enough to let the user give more sections, but only one used here ...
		# the user has to use the naming convention
		templateSection = [ [X[8*s +0], Y[8*s +0]] , [X[8*s +1], Y[8*s +1]], [X[8*s +2], Y[8*s +2]], [X[8*s +3], Y[8*s +3]] ]
		userTemplateInput.append(templateSection)
		templateMag = [ [X[8*s +4], Y[8*s +4]] , [X[8*s +5], Y[8*s +5]], [X[8*s +6], Y[8*s +6]], [X[8*s +7], Y[8*s +7]] ]
		userTemplateInput.append(templateMag)
		
		print 'templateSection', templateSection
		print 'templateMag', templateMag
		
		# calculate angle of the template section (based on the magnetic part)
		angle = getAngle([templateMag[0][0], templateMag[0][1], templateMag[1][0], templateMag[1][1]]) #the angle is calculated on the mag box
		rotTransform = AffineTransform.getRotateInstance(-angle)

		# rotate the template so that tissue is on the left, magnetic on the right (could be rotate 90deg actually ... but backward compatibility ...)
		templateSection = applyTransform(templateSection, rotTransform)
		templateMag = applyTransform(templateMag, rotTransform)
		
		# after rotation, the template points are likely negative, or at least far from (0,0): translate the topleft corner of the tissue bounding box (when magnetic pointing to right) to (100,100)
		bb = sectionToPoly(templateSection).getBounds() # the offset is calculated on the section box
		translateTransform = AffineTransform.getTranslateInstance(- bb.x + 100, - bb.y + 100)

		templateSection = applyTransform(templateSection, translateTransform)
		templateMag = applyTransform(templateMag, translateTransform)

		print 'templateSection', templateSection
		print 'templateMag', templateMag
		templateSections.append(templateSection)
		templateMags.append(templateMag)
	
	writeSections(templateSectionsPath, templateSections)
	writeSections(templateMagsPath, templateMags)
	writeSections(userTemplateInputPath, userTemplateInput) # to create a template image for template matching
	writeSections(sourceTissueMagDescriptionPath, [templateSections[0], templateMags[0]])


def getLandmarks(project, savePath, text):
	landmarks = []
	layer = project.getRootLayerSet().getLayers().get(0)
	disp = Display(project, layer)
	disp.showFront(layer)

	WaitForUserDialog(text).show()
	roi = disp.getRoi()
	IJ.log('ROI landmarks' + str(roi))
	if roi:
		poly = roi.getPolygon()
		landmarks = [list(a) for a in zip(poly.xpoints, poly.ypoints)]
		IJ.log('landmarks' + str(landmarks))
		
		writePoints(savePath, landmarks)
	return landmarks

def getROIDescription(project):
	layer = project.getRootLayerSet().getLayers().get(0)
	disp = Display(project, layer)
	disp.showFront(layer)

	WaitForUserDialog('Click on the 4 corners of a section. Then click on the 4 corners defining the ROI to be imaged. You can postpone that task to later').show()
	roi = disp.getRoi()
	if roi:
		poly = roi.getPolygon()
		section = [list(a) for a in zip(poly.xpoints[:4], poly.ypoints[:4])]
		ROI = [list(a) for a in zip(poly.xpoints[4:], poly.ypoints[4:])]
		
		writeSections(sourceROIDescriptionPath, [section, ROI])

#########################
### Section operations
#########################

def sectionToPoly(l):
	return Polygon( [int(round(a[0])) for a in l] , [int(round(a[1])) for a in l], len(l))

def writePoints(path, points):
	with open(path, 'w') as f:
		for point in points:
			line = str(int(round(point[0]))) + '\t' +  str(int(round(point[1]))) + '\n'
			IJ.log(line)
			f.write(line)
	IJ.log('The point coordinates have been written')

def readPoints(path):
	points = []
	with open(path, 'r') as f:
		lines = f.readlines()
		for point in lines:
			points.append(map(int,point.split('\t')))
	return points

def readSectionCoordinates(path, downFactor = 1):
	sections = []
	if os.path.isfile(path):
		f = open(path, 'r')
		lines = f.readlines()
		for line in lines:
			points = line.split('\t')
			points.pop()
			# print points
			section = [ [int(round(float(point.split(',')[0])/float(downFactor))), int(round(float(point.split(',')[1])/float(downFactor)))] for point in points]
			sections.append(section)
		f.close()
	return sections

def readMISTLocations(MISTPath):
	patchPaths = []
	patchLocations = []
	f = open(MISTPath, 'r')
	lines = f.readlines()
	for line in lines:
		patchPath = os.path.join(inputFolder8bit, line.split(';')[0].split(':')[1][1:])
		x = int(round(float(line.split(';')[2].split(':')[1].split('(')[1].split(',')[0])))
		y = int(round(float(line.split(';')[2].split(':')[1].split(',')[1][1:].split(')')[0])))
		patchPaths.append(patchPath)
		patchLocations.append([x,y])
	f.close()
	return patchPaths, patchLocations

def readStitchedLocations(path):
	f = open(path, 'r')
	lines = f.readlines()[4:] # trimm the heading
	f.close()

	patchPaths = []
	patchLocations = []
	
	for line in lines:
		patchPath = os.path.join(inputFolder8bit, line.replace('\n', '').split(';')[0])
		x = int(float(line.replace('\n', '').split(';')[2].split(',')[0].split('(')[1]))
		y = int(float(line.replace('\n', '').split(';')[2].split(',')[1].split(')')[0]))
		patchPaths.append(patchPath)
		patchLocations.append([x,y])
	return patchPaths, patchLocations

def sectionToList(pointList): # [[1,2],[5,8]] to [1,2,5,8]  
	l = array(2 * len(pointList) * [0], 'd')
	for id, point in enumerate(pointList):
		l[2*id] = point[0]
		l[2*id+1] = point[1]
	return l
	
def listToSection(l): # [1,2,5,8] to [[1,2],[5,8]]  
	pointList = []
	for i in range(len(l)/2):
		pointList.append([l[2*i], l[2*i+1]])
	return pointList

def offsetCorners(corners, xOffset, yOffset):
	for id, corner in enumerate(corners):
		corners[id] = [corner[0] + xOffset, corner[1] + yOffset ]
	return corners

def writeSections(path, sectionList):
	with open(path, 'w') as f:
		for section in sectionList:
			for corner in section:
				# print 'corner', corner
				f.write( str(int(round(corner[0]))) + ',' + str(int(round(corner[1]))) + '\t')
			f.write('\n')
	print 'The coordinates of', len(sectionList), 'sections have been written to', path

#########################
### Geometric operations
#########################
def barycenter(points):
	xSum = 0
	ySum = 0
	for i,point in enumerate(points):
		xSum = xSum + point[0]
		ySum = ySum + point[1]
	x = int(round(xSum/float(i+1)))
	y = int(round(ySum/float(i+1)))
	return x,y

def shrink(section, factor = 0):
	'''
	factor = 0 : nothing happens
	factor = 1 : complete shrinkage to center
	'''
	f = factor
	# center = [(section[0][0] + section[1][0] + section[2][0] + section[3][0])/4., (section[0][1] + section[1][1] + section[2][1] + section[3][1])/4.]
	center = barycenter(section)
	
	p0, p1, p2, p3 = section # the 4 points of the section
	
	p0 = [int(round((1-f) * p0[0] + f * center[0])) , int(round((1-f) * p0[1] + f * center[1]))]
	p1 = [int(round((1-f) * p1[0] + f * center[0])) , int(round((1-f) * p1[1] + f * center[1]))]
	p2 = [int(round((1-f) * p2[0] + f * center[0])) , int(round((1-f) * p2[1] + f * center[1]))]
	p3 = [int(round((1-f) * p3[0] + f * center[0])) , int(round((1-f) * p3[1] + f * center[1]))]

	return [p0, p1, p2, p3]

def getAngle(line):
	diff = [line[0] - line[2],  line[1] - line[3]]
	theta = Math.atan2(diff[1], diff[0])
	return theta
	
def longestDiagonal(corners):
	maxDiag = 0
	for corner1 in corners:
		for corner2 in corners:
			maxDiag = Math.max(Math.sqrt((corner2[0]-corner1[0]) * (corner2[0]-corner1[0]) + (corner2[1]-corner1[1]) * (corner2[1]-corner1[1])), maxDiag)
	return int(maxDiag)

def getArea(section):
	bb = sectionToPoly(section).getBounds()
	section = [[point[0] - bb.x, point[1]-bb.y] for point in section]
	
	im = IJ.createImage('', '8-bit', bb.width, bb.height, 1)
	ip = im.getProcessor()

	ip.setRoi(sectionToPoly(section))
	area = ImageStatistics.getStatistics(ip, Measurements.MEAN, im.getCalibration()).area
	im.close()
	return area

def getConnectedComponents(im, minSize = 0):
	IJ.run(im, 'Invert', '')
	points = []
	roim = RoiManager(True)
	
	pa = ParticleAnalyzer(ParticleAnalyzer.ADD_TO_MANAGER + ParticleAnalyzer.EXCLUDE_EDGE_PARTICLES, Measurements.AREA, None, 0, Double.POSITIVE_INFINITY, 0.0, 1.0)
	pa.setRoiManager(roim)
	pa.analyze(im)
	
	for roi in roim.getRoisAsArray():
		# IJ.log(str(len(roi.getContainedPoints())) + '-' + str(minSize))
		if len(roi.getContainedPoints()) > minSize:
			points.append(roi.getContourCentroid()) # center of mass instead ? There is nothing better apparently ...
	roim.close()

	return points	
	
def getFastCC(im1,im2):
	cc = CrossCorrelation(im1, im2)
	cc.process()
	return cc.getR()
	
def rotate(im, angleDegree):
	ip = im.getProcessor()
	ip.setInterpolationMethod(ImageProcessor.BILINEAR)
	ip.rotate(angleDegree)

def getCroppedRotatedWindow(im, rDegree, x, y): # warning: uses quite a few global parameters
	candidate = im.duplicate() # necessary, I cannot get it to work otherwise (the roi does not reset or something like this ...)

	wCandidate = candidate.getWidth()
	hCandidate = candidate.getHeight()	

	rotate(candidate, rDegree)
	
	candidate.setRoi(wCandidate/2 - wTemplate, hCandidate/2 - hTemplate, wTemplate * 2, hTemplate * 2)
	croppedRotatedCandidate = candidate.crop()

	# the top left corner of the sliding window: middle - template/2 - neighborhood/2 + advancement
	# xWindow = int(wTemplate - wTemplate/2. - neighborhood/2. + xStep * x)
	# yWindow = int(hTemplate - hTemplate/2. - neighborhood/2. + yStep * y)
	xWindow = x
	yWindow = y
			
	# dapi location of the sliding template patch: topleft corner + templateDapi 
	newTemplateDapiCenter = [templateDapiCenter[0] + xWindow, templateDapiCenter[1] + yWindow]

	# distance between the dapiCenter of the sliding template and of the candidate (in the middle of the candidate)
	dapiDistances = sqrt((newTemplateDapiCenter[0] - wTemplate)*(newTemplateDapiCenter[0] - wTemplate) + (newTemplateDapiCenter[1] - hTemplate)*(newTemplateDapiCenter[1] - hTemplate))

	# crop the candidate below the sliding template
	croppedRotatedCandidate.setRoi(xWindow, yWindow, wTemplate, hTemplate)
	return croppedRotatedCandidate.crop()
	
	
def templateMatchCandidate(atom, candidatePaths, templateMatchingPath, allResults):
	template = ImagePlusAdapter.wrap(IJ.openImage(templateMatchingPath))

	while atom.get() < len(candidatePaths):
		k = atom.getAndIncrement()
		if k < len(candidatePaths):
			IJ.log('Processing section ' + str(k))
			candidatePath = candidatePaths[k]
			cand = IJ.openImage(candidatePath)
			
			wCandidate = cand.getWidth()
			hCandidate = cand.getHeight()

			results = []
			for rotationId in range(rotations):
				candidate = cand.duplicate() # necessary, I cannot get it to work otherwise (the roi does not reset or something like this ...)
				
				# rotate the candidate
				rotationDegree = rotationStepDegree * rotationId
				rotate(candidate, rotationDegree)
				
				# ip = candidate.getProcessor()
				# ip.setInterpolationMethod(ImageProcessor.BILINEAR)
				# ip.rotate(rotationDegree)
				
				# extract a central region of the rotation candidate, size is template*2 - is that really necessary ?
				# candidate.setRoi(wCandidate/2 - wTemplate, hCandidate/2 - hTemplate, wTemplate * 2, hTemplate * 2)
				# croppedRotatedCandidate = candidate.crop()

				# loops for the brute force search
				for x in range(xMatchingGrid):
					for y in range(yMatchingGrid):
						# the top left corner of the sliding window: middle - template/2 - neighborhood/2 + advancement
						xWindow = int(wCandidate/2. - wTemplate/2. - neighborhood/2. + xStep * x)
						yWindow = int(hCandidate/2. - hTemplate/2. - neighborhood/2. + yStep * y)
						
						# dapi location of the sliding template patch: topleft corner + templateDapi 
						newTemplateDapiCenter = [templateDapiCenter[0] + xWindow, templateDapiCenter[1] + yWindow]
						# distance between the dapiCenter of the sliding template and of the candidate (in the middle of the candidate)
						dapiDistances = sqrt((newTemplateDapiCenter[0] - wTemplate)*(newTemplateDapiCenter[0] - wTemplate) + (newTemplateDapiCenter[1] - hTemplate)*(newTemplateDapiCenter[1] - hTemplate))
						# IJ.log(str(dapiDistances))
						if dapiDistances < dapiCenterDistanceThreshold:
						# if dapiDistances < 99999:
							# crop the candidate below the sliding template patch
							candidate.setRoi(xWindow, yWindow, wTemplate, hTemplate)
							croppedRotatedCandidate = candidate.crop()
							# IJ.log('-----' + str(croppedRotatedCandidate.getWidth()))
							croppedRotatedCandidate = ImagePlusAdapter.wrap(croppedRotatedCandidate)
							# compute CC and append result
							cc = getFastCC(template, croppedRotatedCandidate)
							results.append([cc , [rotationDegree, x, y]])

			# close open images
			cand.close()
			croppedRotatedCandidate.close()
			candidate.close()
			
			# sort results and append to total results
			sortedResults = sorted(results, key=itemgetter(0), reverse=True) # maybe sort with the Id instead ? 
			# sortedResults = sorted(results, key=itemgetter(1), reverse=True) # maybe sort with the Id instead ? 
			# IJ.log(str(sortedResults[:5]))
			bestId = sortedResults[0][1]
			allResults.append([k] + sortedResults)

			# # optional display of the best candidates
			# im = IJ.openImage(candidatePath)
			# rotationDegree = bestId[0]
			# rotate(im, rotationDegree)
			# im.setRoi(int(wCandidate/2. - wTemplate/2. - neighborhood/2. + xStep * bestId[1]) , int(hCandidate/2. - hTemplate/2. - neighborhood/2. + yStep * bestId[2]), wTemplate, hTemplate)
			# im = im.crop()
			# im.show()
			# 8/0
#########################
### Affine transform operations
#########################			
def applyTransform(section, aff):
	sourceList = sectionToList(section)
	targetList = array(len(sourceList) * [0], 'd')
	aff.transform(sourceList, 0, targetList, 0, len(section))
	targetSection = listToSection(targetList)
	return targetSection

def affineT(sourceLandmarks, targetLandmarks, sourcePoints):
	aff = fc.getModelFromPoints(sourceLandmarks, targetLandmarks).createAffine()
	return applyTransform(sourcePoints, aff)

#######################
# Parameters to provide
#######################
inputFolder = os.path.normpath(r'D:\ThomasT\Thesis\B6\B6_Wafer1_203_24_12\AllImages')


###################
# Get the mosaic configuration
###################
# mosaicMetadataPath = os.path.join(os.path.dirname(inputFolder), 'Mosaic_Metadata.xml')
try:
	# mosaicMetadataPath = os.path.join(os.path.dirname(inputFolder), filter(lambda x: 'metadata' in x, os.listdir(os.path.dirname(inputFolder)))[0]) # ugly ...
	mosaicMetadataPath = os.path.join(inputFolder, filter(lambda x: 'Mosaic_Metadata' in x, os.listdir(inputFolder))[0]) # ugly ...
	IJ.log('Using grid size from the ZEN metadate file')
	xmldoc = minidom.parse(mosaicMetadataPath)
	xGrid = int(float(xmldoc.getElementsByTagName('Columns')[0].childNodes[0].nodeValue))
	yGrid = int(float(xmldoc.getElementsByTagName('Rows')[0].childNodes[0].nodeValue))
except Exception, e:
	IJ.log('No metadata file found')
	IJ.log('Using manually entered grid size for the mosaic')
	xGrid = 1
	yGrid = 1
IJ.log('Mosaic size: (' + str(xGrid) + ', ' + str(yGrid) + ')')

channels = ['BF', 'DAPI', '488', '546']
nChannels = len(channels)

overlap = 0.1
downsizingFactor = 3
shrinkFactor = 2 # for orientation flipping

calibrationX = 1.302428227746592
calibrationY = 1.302910064239829


#########################
# Parameters for template matching
#########################
xMatchingGrid = 10 # number of locations tested with the template on the x axis
yMatchingGrid = 10
neighborhood = 25 # neighborhood around the center tested for matching
xStep = neighborhood/xMatchingGrid
yStep = neighborhood/yMatchingGrid

rotations = 180 # number of rotations tested within the total 360 degrees
rotationStepDegree = 360/float(rotations)
rotationStepRadian = rotationStepDegree * PI / 180.

dapiCenterDistanceThreshold = 9999 # during brute force matching, the DAPI of the template and the DAPI of the candidate should be close
tissueShrinkingForEdgeFreeRegion = 0.6

#######################
# Setting up
IJ.log('Setting up')
#######################

#######################
# Getting the template
#######################

ControlWindow.setGUIEnabled(False)

workingFolder = fc.mkdir_p(os.path.join(os.path.dirname(inputFolder), 'workingFolder'))

templatePath = os.path.join(workingFolder,'templateCoordinates.txt')
inputFolderContent = os.listdir(inputFolder)

inputFolder8bit = os.path.join(workingFolder, '8bit' + os.path.basename(inputFolder))
fc.mkdir_p(inputFolder8bit)
inputFolder8bitContent = os.listdir(inputFolder8bit)

inputFolder8bitFullRes = os.path.join(workingFolder, '8bitFullRes' + os.path.basename(inputFolder))
fc.mkdir_p(inputFolder8bitFullRes)
inputFolder8bitFullResContent = os.listdir(inputFolder8bitFullRes)

fluoOffsetPath = os.path.join(workingFolder, 'fluoOffset.txt')

templateSectionsPath = os.path.join(workingFolder, 'templateSections.txt')
templateMagsPath = os.path.join(workingFolder, 'templateMags.txt')

templateSectionsLowResPath = os.path.join(workingFolder, 'templateSectionsLowRes.txt')
templateMagsLowResPath = os.path.join(workingFolder, 'templateMagsLowRes.txt')

userTemplateInputPath = os.path.join(workingFolder, 'userTemplateInputPath.txt')

preImagingFolder = 	fc.mkdir_p(os.path.join(workingFolder, 'preImaging'))
landmarksPath = os.path.join(preImagingFolder, 'source_landmarks.txt')
sourceTissueMagDescriptionPath = os.path.join(preImagingFolder, 'source_tissue_mag_description.txt') #2 sections: template tissue and template mag
sourceROIDescriptionPath = os.path.join(preImagingFolder, 'source_ROI_description.txt') #2 sections: template tissue and ROI

patchSizeAndOverlapFullResPath = os.path.join(workingFolder, 'patchSizeAndOverlapFullRes.txt')
patchSizeAndOverlapLowResPath = os.path.join(workingFolder, 'patchSizeAndOverlapLowRes.txt')

# print "filter(lambda x: '_Edges' in x, os.listdir(workingFolder))", filter(lambda x: '_Edges' in x, os.listdir(workingFolder))

magSectionsLowResPath = os.path.join(workingFolder, 'allMagSectionsCoordinatesLowRes.txt')
tissueSectionsLowResPath = os.path.join(workingFolder, 'allTissueSectionsCoordinatesLowRes.txt')

magSectionsHighResPath = os.path.join(workingFolder, 'allMagSectionsCoordinatesFullRes.txt')
tissueSectionsHighResPath = os.path.join(workingFolder, 'allTissueSectionsCoordinatesFullRes.txt')

finalMagSectionsPath = os.path.join(workingFolder, 'finalMagSectionsCoordinates.txt')
finalTissueSectionsPath = os.path.join(workingFolder, 'finalTissueSectionsCoordinates.txt')

finalMagSectionsPreImagingPath = os.path.join(preImagingFolder, 'source_sections_mag.txt')
finalTissueSectionsPreImagingPath = os.path.join(preImagingFolder, 'source_sections_tissue.txt')

flipFlag = os.path.join(workingFolder, 'flipflag')

### Matching files and folders ### 
dapiCentersPath = os.path.join(workingFolder, 'dapiCenters')
sectionsSpecsPath = os.path.join(workingFolder, 'sectionsSpecs')

#######################
# 0.0 Converting all images to 8 bit and downsizing
#######################
if len(filter(lambda x: os.path.splitext(x)[1] == '.tif', inputFolder8bitContent)) < len(filter(lambda x: os.path.splitext(x)[1] == '.tif', inputFolderContent)):
	IJ.log('0. 8-bit conversion and downsizing')
	imageNames = filter(lambda x: os.path.splitext(x)[1] == '.tif', inputFolderContent)
	imagePaths = [os.path.join(inputFolder, imageName) for imageName in imageNames]
	newImagePaths = [os.path.join(inputFolder8bit, imageName) for imageName in imageNames]

	atomicI = AtomicInteger(0)
	fc.startThreads(convertTo8BitAndResize, fractionCores = 0.9, wait = 0, arguments = (imagePaths, newImagePaths, downsizingFactor, atomicI))
#######################
# 0.1 Converting all images to 8 bit, for full res
#######################
if len(filter(lambda x: os.path.splitext(x)[1] == '.tif', inputFolder8bitFullResContent)) < len(filter(lambda x: os.path.splitext(x)[1] == '.tif', inputFolderContent)):
	IJ.log('0. 8-bit conversion for full res')
	imageNames = filter(lambda x: os.path.splitext(x)[1] == '.tif', inputFolderContent)
	imagePaths = [os.path.join(inputFolder, imageName) for imageName in imageNames]
	newImagePaths = [os.path.join(inputFolder8bitFullRes, imageName) for imageName in imageNames]

	atomicI = AtomicInteger(0)
	fc.startThreads(convertTo8BitAndResize, fractionCores = 0.9, wait = 0, arguments = (imagePaths, newImagePaths, 1, atomicI))

#######################
# 1. Assembling the overview
#######################
# setting up trakem project
trakemFolder = workingFolder
projectPath = os.path.join(os.path.normpath(trakemFolder) , 'WaferLMProject.xml') # the project with 7 layers (low res I believe)
projectPathBis = os.path.join(os.path.normpath(trakemFolder) , 'WaferLMProjectBis.xml')
projectPathFullRes = os.path.join(os.path.normpath(trakemFolder) , 'WaferLMProjectFullRes.xml')
overlaysProjectPath = os.path.join(os.path.normpath(trakemFolder) , 'OverlaysWaferLMProject.xml')

# Image names from ZEN 
# Wafer_SFN_2016_b0s0c0x1249-1388y3744-1040m45.tif

imageNames = []

if not os.path.isfile(projectPath):
	IJ.log('1. Assembling the overview')
	imageNames = []
	for channelId, channel in enumerate(channels):
		# imageNames = fc.naturalSort(filter(lambda x: (os.path.splitext(x)[1] in ['.tif', '.TIF']) and  ('c' + str(channelId) + 'x') in x , os.listdir(inputFolder8bit)))
		if channel == 'BF': # BF and fluo channels have different naming schemes at the Z1 microscope because all fluo chanels imaged together and the BF imaged separately (because high intensity needed for fluo but low intensity needed for BF)
			imageNames = fc.naturalSort(filter(lambda x: (os.path.splitext(x)[1] in ['.tif', '.TIF']) and  (('c' + str(channelId) + 'x') in x) and ('BF' in x) , os.listdir(inputFolder8bit)))
		else:
			imageNames = fc.naturalSort(filter(lambda x: (os.path.splitext(x)[1] in ['.tif', '.TIF']) and  ('c' + str(channelId-1) + 'x') in x , os.listdir(inputFolder8bit)))
	# Wafer_210_BF_b0s0c0x3747-1388y0-1040m3
	# Wafer_210_DAPI_488_546_b0s0c0x4996-1388y11232-1040m112
	
		if channelId == 0:
			# # commented lines below probably to delete
			# im0 = IJ.openImage(os.path.join(inputFolder8bit, imageNames[0]))
			# width = im0.getWidth()
			# height = im0.getHeight()
			# im0.close()

			# widthEffective = int(round((1-overlap) * width))
			# heightEffective = int(round((1-overlap) * height))

			project, loader, layerset, nLayers = fc.getProjectUtils(fc.initTrakem(trakemFolder, nChannels + 3)) #channels + thresholded + edged + rawEdged
			loader.setMipMapsRegeneration(False)
			project.saveAs(projectPath, True)
		
			layer = layerset.getLayers().get(channelId)

			# inserting all patches
			patchPaths = []
			patchLocations = []
			# for x in range(xGrid):
				# for y in range(yGrid):
					# patchNumber = y * xGrid + x * (1 - (y%2)) + (xGrid - x - 1) * (y%2)
					# patchNumber = x * yGrid + y * (1 - (x%2)) + (yGrid - y - 1) * (x%2)
					# patchNumber = x * yGrid + y

			# renaming to e.g. DAPI_038.tif for stitching plugin instead of the ZEN names
			for imageId, imageName in enumerate(fc.naturalSort(imageNames)):
				sourcePatchPath = os.path.join(inputFolder8bit, imageName)
				targetPatchPath = os.path.join(inputFolder8bit, channel + '_' + str(imageId).zfill(3) + '.tif')
				os.rename(sourcePatchPath, targetPatchPath)

				# patchPaths.append()
				
				# xmldoc = minidom.parse(os.path.join(inputFolder, imageName + '_metadata.xml'))					
				# patchLocations.append([int(float(xmldoc.getElementsByTagName('StageXPosition')[0].childNodes[0].nodeValue)/float(calibrationX)/float(downsizingFactor)) + 10000, int(float(xmldoc.getElementsByTagName('StageYPosition')[0].childNodes[0].nodeValue)/float(calibrationY)/float(downsizingFactor)) + 10000]) 

			if (xGrid * yGrid) != 1:
				patchLocationsPath = os.path.join(inputFolder8bit, 'TileConfiguration.registered.txt') # patch locations calculated by the plugin
				if not os.path.isfile(patchLocationsPath):
					# Grid/Collection stitching plugin has issues with section-free areas
					command = 'type=[Grid: column-by-column] order=[Down & Right                ] grid_size_x=' + str(xGrid) + ' grid_size_y=' + str(yGrid) + ' tile_overlap=' + str(overlap * 100) + ' first_file_index_i=0 directory=' + inputFolder8bit + ' file_names=BF_{iii}.tif output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory=' + inputFolder8bit
					IJ.log('Stitching command - ' + command)
					IJ.run('Grid/Collection stitching', command)

					# # The MIST plugin does not seem to have problems with the section-free areas, but more inaccurate
					# IJ.run('MIST', 'gridwidth=' + str(xGrid) + ' gridheight=' + str(yGrid) + ' starttile=0 imagedir=' + inputFolder8bit + ' filenamepattern=' + channel + '_{ppp}.tif filenamepatterntype=SEQUENTIAL gridorigin=UL assemblefrommetadata=false globalpositionsfile=[] numberingpattern=VERTICALCOMBING startrow=0 startcol=0 extentwidth=' + str(xGrid) + ' extentheight=' + str(yGrid)  + ' timeslices=0 istimeslicesenabled=false issuppresssubgridwarningenabled=false outputpath=' + inputFolder8bit + ' displaystitching=false outputfullimage=false outputmeta=true outputimgpyramid=false blendingmode=OVERLAY blendingalpha=NaN outfileprefix=img- programtype=AUTO numcputhreads=' + str(Runtime.getRuntime().availableProcessors()) + ' loadfftwplan=true savefftwplan=true fftwplantype=MEASURE fftwlibraryname=libfftw3 fftwlibraryfilename=libfftw3.dll planpath=' + os.path.join(IJ.getDirectory('imagej'), 'lib', 'fftw', 'fftPlans') + ' fftwlibrarypath=' + os.path.join(IJ.getDirectory('imagej'), 'lib', 'fftw')  + ' stagerepeatability=0 horizontaloverlap=' + str(overlap * 100) + ' verticaloverlap=' + str(overlap * 100) + ' numfftpeaks=0 overlapuncertainty=NaN isusedoubleprecision=false isusebioformats=false isenablecudaexceptions=false translationrefinementmethod=SINGLE_HILL_CLIMB numtranslationrefinementstartpoints=16 headless=false loglevel=MANDATORY debuglevel=NONE')			
					# patchLocationsPath = os.path.join(inputFolder8bit, 'img-global-positions-0.txt')			
					# patchPaths, patchLocations = readMISTLocations(patchLocationsPath)
				
				patchPaths, patchLocations = readStitchedLocations(patchLocationsPath)
			else:
				patchLocations.append([0,0])
				patchPaths.append(os.path.join(inputFolder8bit, channel + '_' + str(0).zfill(3) + '.tif'))
	
			IJ.log('patchPaths ' + str(patchPaths))
			IJ.log('patchLocations ' + str(patchLocations))
			
			# import all patches in the trakEM project
			importFilePath = createImportFile(workingFolder, patchPaths, patchLocations)
			task = loader.importImages(layerset.getLayers().get(channelId), importFilePath, '\t', 1, 1, False, 1, 0)
			task.join()

			# # # # # # /!\ old version with the phase correlation TEM stitcher
			# # # #######################
			# # # # 2. Montaging the overview
			# # # IJ.log('2. Montaging the overview')
			# # # #######################
			# # # patchScale, hide_disconnected, remove_disconnected, mean_factor, min_R = 1, False, False, 2.5, 0.1
			# # # stitcher = StitchingTEM()
			# # # params = PhaseCorrelationParam(patchScale, overlap, hide_disconnected, remove_disconnected, mean_factor, min_R)
			# # # collectionPatches = layer.getDisplayables(Patch)
			# # # stitcher.montageWithPhaseCorrelation(collectionPatches, params)
			
		else: # stitching has already been done in the first channel, simply read the calculated stitching locations 
			
			# renaming to e.g. DAPI_038.tif for stitching plugin instead of the ZEN names			
			for imageId, imageName in enumerate(fc.naturalSort(imageNames)):
				sourcePatchPath = os.path.join(inputFolder8bit, imageName)
				targetPatchPath = os.path.join(inputFolder8bit, channel + '_' + str(imageId).zfill(3) + '.tif')
				os.rename(sourcePatchPath, targetPatchPath)
			
			# read the patch coordinates of stitched layer 0 of the trakem project
			patches0 = layerset.getLayers().get(0).getDisplayables(Patch)
			patchPaths = [os.path.join(os.path.dirname(patch.getFilePath()), os.path.basename(patch.getFilePath()).replace(channels[0], channel)) for patch in patches0]
			patchLocations = [ [patch.getX() , patch.getY()] for patch in patches0]
			
			# patchLocations = []
			# for patchPath in patchPaths:
				# xmldoc = minidom.parse(os.path.join(inputFolder, os.path.basename(patchPath) + '_metadata.xml'))								
				# patchLocations.append([int(float(xmldoc.getElementsByTagName('StageXPosition')[0].childNodes[0].nodeValue)/float(calibrationX)/float(downsizingFactor)) + 10000 , int(float(xmldoc.getElementsByTagName('StageYPosition')[0].childNodes[0].nodeValue)/float(calibrationY)/float(downsizingFactor)) + 10000]) 
			
			# import patches to the trakem project
			importFilePath = createImportFile(workingFolder, patchPaths, patchLocations)
			task = loader.importImages(layerset.getLayers().get(channelId), importFilePath, '\t', 1, 1, False, 1, 0)
			task.join()
			
	project.save()
	IJ.log('Assembling the channels done and saved into ' + projectPath)
	fc.resizeDisplay(layerset)
	fc.closeProject(project)
	
#######################
# 2. Assembling the full res overview
#######################

if not os.path.isfile(projectPathFullRes):
	IJ.log('Creating the full res project: ' + str(projectPathFullRes))
	project, loader, layerset, nLayers = fc.getProjectUtils(fc.initTrakem(trakemFolder, nChannels))
	layerset.setDimensions(0, 0, 100000, 100000)
	loader.setMipMapsRegeneration(False)
	project.saveAs(projectPathFullRes, True)
	
	layer = layerset.getLayers().get(0)

	# insterting from the calculated registration does not work because there are negative values that trigger an offset of the layerset (though I do not understand why the problem does not occur in the low res project)
	lowResproject, lowResloader, lowReslayerset, lowResnLayers = fc.openTrakemProject(projectPath)

	for l, layer in enumerate(layerset.getLayers()):
	
		# renaming to e.g. DAPI_038.tif for stitching plugin instead of the ZEN names			
		if l == 0:
			imageNames = fc.naturalSort(filter(lambda x: 
			(os.path.splitext(x)[1] in ['.tif', '.TIF']) and
			('c0x' in x) and 
			('BF' in x),
			os.listdir(inputFolder8bitFullRes)))
		else:
			imageNames = fc.naturalSort(filter(lambda x: 
			(os.path.splitext(x)[1] in ['.tif', '.TIF']) and 
			('c' + str(l-1) + 'x') in x,
			os.listdir(inputFolder8bitFullRes)))
		
		for imageId, imageName in enumerate(fc.naturalSort(imageNames)):
			sourcePatchPath = os.path.join(inputFolder8bitFullRes, imageName)
			targetPatchPath = os.path.join(inputFolder8bitFullRes, channels[l] + '_' + str(imageId).zfill(3) + '.tif')
			os.rename(sourcePatchPath, targetPatchPath)
		
		patchPaths = []
		patchLocations = []
		for patch in lowReslayerset.getLayers().get(l).getDisplayables(Patch):
			lowPatchPath = patch.getImageFilePath()
			# renaming the patchPaths to the full resolution ones (replacing the folder)
			highPatchPath = os.path.normpath(lowPatchPath).replace(os.path.normpath(inputFolder8bit), os.path.normpath(inputFolder8bitFullRes))
			print 'lowPatchPath', lowPatchPath
			print 'highPatchPath', highPatchPath
			# 8/0
			patchPaths.append(highPatchPath)
			# upscaling the locations for the full res
			patchLocations.append([int(round(patch.getX()*float(downsizingFactor))), int(round(patch.getY()*float(downsizingFactor)))])
		
		# trakEM import
		importFilePath = createImportFile(workingFolder, patchPaths, patchLocations)
		task = loader.importImages(layer, importFilePath, '\t', 1, 1, False, 1, 0)
		task.join()
		
	fc.resizeDisplay(layerset)
		
	project.save()
	time.sleep(2)
	fc.closeProject(project)
	fc.closeProject(lowResproject)
	
#######################
# 3. Asking the user to input an offset between the BF and fluo channels (they have been imaged in two different sessions, the stage might have moved)
#######################
if not os.path.isfile(fluoOffsetPath): #/!\ Warning: the fluo offset is on the lowRes project
	IJ.log('Asking user for offset between BF and fluo channels')
	project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)
	offsetPoints = getLandmarks(project, fluoOffsetPath, 'Place pairs of landmarks to assess the fluorescent channel offset {BF, Fluo}')
	meanOffset = [0,0]
	
	# if there are points from the user, take the average
	if len(offsetPoints)>0:
		offsets = []
		for k in range(len(offsetPoints)/2):
			offsets.append([offsetPoints[2*k+1][0] - offsetPoints[2*k][0], offsetPoints[2*k+1][1] - offsetPoints[2*k][1]])
		
		for offset in offsets:
			meanOffset[0] = meanOffset[0] + offset[0]	
			meanOffset[1] = meanOffset[1] + offset[1]	
		meanOffset[0] = meanOffset[0]/float(len(offsets))
		meanOffset[1] = meanOffset[1]/float(len(offsets))
	
	writePoints(fluoOffsetPath, [meanOffset])
	
	# offsetting the patches
	for id, patch in enumerate(layerset.getLayers().get(channels.index('BF')).getDisplayables(Patch)):
		patch.setLocation(patch.getX() + meanOffset[0], patch.getY() + meanOffset[1])

	# save and close
	fc.resizeDisplay(layerset)
	project.save()
	fc.closeProject(project)
		
	# Offsetting the fullResProject
	project, loader, layerset, nLayers = fc.openTrakemProject(projectPathFullRes)
	for id, patch in enumerate(layerset.getLayers().get(0).getDisplayables(Patch)):
		patch.setLocation(patch.getX() + meanOffset[0] * downsizingFactor, patch.getY() + meanOffset[1] * downsizingFactor)
	# fc.resizeDisplay(layerset) /!\ WARNING: Never resize the display of the fullresProject, otherwise the offset gets lost 
	project.save()
	fc.closeProject(project)

#######################
# 4. Asking user for the template
#######################	
if not os.path.isfile(templateMagsPath):
	IJ.log('3. Asking user for the template')
	project, loader, layerset, nLayers = fc.openTrakemProject(projectPathFullRes)
	getTemplates(project)
	project.save() # saving mipmaps
	fc.closeProject(project)

	# write the low res templates for python segmentation
	templateMagLowRes = readSectionCoordinates(templateMagsPath, downFactor = downsizingFactor)[0]
	writeSections(templateMagsLowResPath, [templateMagLowRes])

	templateTissueLowRes = readSectionCoordinates(templateSectionsPath, downFactor = downsizingFactor)[0]
	writeSections(templateSectionsLowResPath, [templateTissueLowRes])

#######################
# 5. Asking user for the wafer landmarks
#######################	
if not os.path.isfile(landmarksPath):
	project, loader, layerset, nLayers = fc.openTrakemProject(projectPathFullRes)
	landmarks = getLandmarks(project, landmarksPath, 'Select at least 4 landmarks for CLEM imaging (4 is usually ok). You can postpone that to another time.')
	addLandmarkOverlays(project, landmarks)
	project.save() # saving mipmaps
	fc.closeProject(project)

#######################
# 6. Asking user for the ROI in the tissue part
#######################	
if not os.path.isfile(sourceROIDescriptionPath):
	project, loader, layerset, nLayers = fc.openTrakemProject(projectPathFullRes)
	getROIDescription(project)
	project.save() # saving mipmaps
	fc.closeProject(project)

	
####################
# Optionally start the manual segmentation mode
####################
# if not fc.getOK('Manual segmentation mode ?'):
if True:
	#############################
	# WEKA process the DAPI sections
	#############################
	imageNames = fc.naturalSort(filter(lambda x: # the DAPI tiles
	(os.path.splitext(x)[1] in ['.tif', '.TIF']) and
	('DAPI' in x)
	, fc.naturalSort(os.listdir(inputFolder8bit))))
	imagePaths = [os.path.join(inputFolder8bit, imageName) for imageName in imageNames]
	
	clahedFolder = fc.mkdir_p(os.path.join(workingFolder, 'clahedFolder'))
	clahedPaths = [os.path.join(clahedFolder, os.path.splitext(imageName)[0] + '_CLAHED' + os.path.splitext(imageName)[1])
	for imageName in fc.naturalSort(imageNames)]

	wekaedFolder = fc.mkdir_p(os.path.join(workingFolder, 'wekaedFolder'))
	wekaedPaths = [os.path.join(wekaedFolder, os.path.splitext(imageName)[0] + '_WEKAED' + os.path.splitext(imageName)[1]) 
	for imageName in fc.naturalSort(imageNames)]
	
	wekaedStackPath = os.path.join(workingFolder, 'wekaedStack.tif')
	wekaModelPath = os.path.join(workingFolder, 'wekaClassifier.model')

	edgedFolder = fc.mkdir_p(os.path.join(workingFolder, 'edgedFolder'))
	edgedPaths = [os.path.join(edgedFolder, os.path.splitext(imageName)[0] + '_EDGED' + os.path.splitext(imageName)[1]) 
	for imageName in fc.naturalSort(imageNames)]

	if sum([os.path.isfile(wekaedPath) for wekaedPath in wekaedPaths]) != len(imageNames): # check whether files already created
		if sum([os.path.isfile(clahedPath) for clahedPath in clahedPaths]) != len(imageNames):
			# creating the DAPI clahed images
			for k, imagePath in enumerate(imagePaths): # (the list is sorted)
				im = IJ.openImage(imagePath)
				im = fc.localContrast(im)
				im = fc.localContrast(im)
				IJ.log('CLAHING ' + str(k) + ' out of ' + str(len(imagePaths)))
				IJ.save(im, clahedPaths[k])

		# getting the weka model from user on the clahed DAPI images
		if not os.path.isfile(wekaModelPath):
			WaitForUserDialog('Create a WEKA model on the CLAHED images and save it as "wekaClassifier.model" in "workingFolder" then click OK.').show()
		segmentator = WekaSegmentation()
		segmentator.loadClassifier(wekaModelPath);

		# apply the weka classifier and save the stack
		imageStack = fc.stackFromPaths(clahedPaths)
		if not os.path.isfile(wekaedStackPath):
			result = segmentator.applyClassifier(imageStack, 0, 1) # 0 indicates number of threads is auto-detected
			result.show()
			time.sleep(1)
			IJ.run('Grays') # or IJ.run(im, 'Grays', '')
			IJ.setMinAndMax(0, 1) # warning: relies on the image being the current one
			time.sleep(1)
			IJ.run(result, '8-bit', '')
			IJ.save(result, wekaedStackPath)
			
			# # small trick to ask an input from the user. Rerun the script then
			# WaitForUserDialog('Open the WEKA stack and find parameters for subsequent thresholding. Then rerun').show()
			# 8/0
			result.close()
			
		result = IJ.openImage(wekaedStackPath)
		stack = result.getImageStack()

		# loop through the tiles
		for k, wekaedPath in enumerate(wekaedPaths):
			# save the thresholded/eroded WEKA
			IJ.log('Saving WEKA ...')
			tileIndex = result.getStackIndex(0, 0, k + 1) # to access the slice in the stack
			wekaTile = ImagePlus('wekaed_' + str(k), stack.getProcessor(tileIndex).convertToByteProcessor())
			IJ.save(wekaTile, wekaedPath)
			
			# # save the edged weka: probably not necessary any more ...
			# IJ.run(wekaTile, 'Find Edges', '')
			# # edged = fc.blur(wekaed, 2)
			# # edged = fc.minMax(edged, 60, 60)
			# edged = fc.minMax(wekaed, 60, 60)
			# IJ.save(edged, edgedPaths[k])
		result.changes = False
		result.close()

	##############################################
	# Edge the raw images for section orientation flipping	
	##############################################
	imageNames = fc.naturalSort(filter(lambda x: (os.path.splitext(x)[1] in ['.tif', '.TIF']) and  ('BF' in x), os.listdir(inputFolder8bit)))
	imagePaths = [os.path.join(inputFolder8bit, imageName) for imageName in imageNames]
	rawEdgedFolder = fc.mkdir_p(os.path.join(workingFolder, 'rawEdgedFolder'))
	rawEdgedPaths = [os.path.join(rawEdgedFolder, os.path.splitext(imageName)[0] + '_rawEdged' + os.path.splitext(imageName)[1]) for imageName in imageNames]

	if sum([os.path.isfile(rawEdgedPath) for rawEdgedPath in rawEdgedPaths]) != len(imageNames):
		IJ.log('RawEdging ...')
		for k, imagePath in enumerate(imagePaths):
			im = IJ.openImage(imagePath)
			IJ.run(im, 'Median...', 'radius=2')
			im = fc.localContrast(im)
			IJ.run(im, 'Find Edges', '')
			IJ.log('RawEdging ' + str(k) + ' out of ' + str(len(imagePaths)))
			IJ.save(im, rawEdgedPaths[k])

	dapiWekaedPath = os.path.join(workingFolder, 'dapiWekaed.tif')
	if not os.path.isfile(projectPathBis):
		#######################
		# Insert the wekaed tiles in the last layers of projectBis
		IJ.log('Insert the wekaed tiles in the last layers of projectBis')
		#######################
		shutil.copyfile(projectPath, projectPathBis) # the low res project with all channels
		project, loader, layerset, nLayers = fc.openTrakemProject(projectPathBis)
		
		meanOffset = readPoints(fluoOffsetPath) # actually not needed here

		wekaedPatchPaths = []
		edgedPatchPaths = []
		
		# get patch coordinates from the dapi layer and convert the names of the patches to insert
		patchLocations = []
		for id, patch in enumerate(layerset.getLayers().get(1).getDisplayables(Patch)): # read patch coordinates in the dapi layer (no offset)
			IJ.log('Processing patch ' + str(id))
			patchLocations.append([patch.getX() , patch.getY()])
			
			patchPath = patch.getImageFilePath()
			patchName = os.path.basename(patchPath)
			# patchName = os.path.basename(patchPath).replace('BF', 'Fluo')
			
			# get the wekaedPath from the dapi path
			wekaedPatchName = os.path.splitext(patchName)[0] + '_WEKAED' + os.path.splitext(patchName)[1]
			wekaedPatchPath = os.path.join(wekaedFolder, wekaedPatchName)
			wekaedPatchPaths.append(wekaedPatchPath)
			
			# get the edgedPath from the dapi path
			edgedPatchName = os.path.splitext(patchName)[0] + '_EDGED' + os.path.splitext(patchName)[1]
			edgedPatchPath = os.path.join(edgedFolder, edgedPatchName)
			edgedPatchPaths.append(edgedPatchPath)
			
		# insert the patches in layer number (nChannels)
		importFile = createImportFile(workingFolder, wekaedPatchPaths, patchLocations, layer = nChannels)
		task = loader.importImages(layerset.getLayers().get(0), importFile, '\t', 1, 1, False, 1, 0)
		task.join()

		# insert the patches in layer number (nChannels + 1)
		importFile = createImportFile(workingFolder, edgedPatchPaths, patchLocations, layer = nChannels + 1)
		task = loader.importImages(layerset.getLayers().get(0), importFile, '\t', 1, 1, False, 1, 0)
		task.join()
		
		# Insert the raw edges, useful for flips
		patchLocations = [] # need to use new patchLocations because of the BF-fluo offset
		rawEdgedPatchPaths = []
		for id, patch in enumerate(layerset.getLayers().get(0).getDisplayables(Patch)):
			IJ.log('Inserting rawedged patch ' + str(id))
			patchLocations.append([patch.getX() , patch.getY()]) # already contains the offset
			
			patchPath = patch.getImageFilePath()
			patchName = os.path.basename(patchPath)
			
			rawEdgedPatchName = os.path.splitext(patchName)[0] + '_rawEdged' + os.path.splitext(patchName)[1]
			rawEdgedPatchPath = os.path.join(rawEdgedFolder, rawEdgedPatchName)
			rawEdgedPatchPaths.append(rawEdgedPatchPath)

		importFile = createImportFile(workingFolder, rawEdgedPatchPaths, patchLocations, layer = nChannels + 2)
		task = loader.importImages(layerset.getLayers().get(0), importFile, '\t', 1, 1, False, 1, 0)
		task.join()
		
		# blending the raw and the 3 last channels
		Blending.blendLayerWise(layerset.getLayers(0, 0), True, None) # layerRaw
		Blending.blendLayerWise(layerset.getLayers(nChannels, nChannels), True, None) # layerWekaed
		Blending.blendLayerWise(layerset.getLayers(nChannels + 1, nChannels + 1), True, None) # layerWekaed
		Blending.blendLayerWise(layerset.getLayers(nChannels + 2, nChannels + 2), True, None) # layerWekaed
		
		layerDapiWekaed = layerset.getLayers().get(nChannels)
		dapiWekaed = loader.getFlatImage(layerDapiWekaed, 
		# Rectangle(0, 0, layerDapiWekaed.getLayerWidth(), layerDapiWekaed.getLayerHeight()), # warning: should I take the bounding box of the layer, not of the layerset because of the fluo offset ?
		layerset.get2DBounds(),
		1, 0x7fffffff, ImagePlus.GRAY8, Patch, 
		layerDapiWekaed.getAll(Patch), True, Color.black, None)
		
		IJ.save(dapiWekaed, dapiWekaedPath)
		dapiWekaed.show()
		
		project.save()
		fc.closeProject(project)

		# small trick to ask an input from the user. Rerun the script then
		WaitForUserDialog('Open the wekaed wafer and find parameters for subsequent thresholding. Then rerun').show()
		dapiWekaed.close()
		8/0
	
	########################
	# Second round of templating: create the templates, calculate dapiCenter, calculate box in which there should not be any edge
	########################
	templateTissue = readSectionCoordinates(templateSectionsLowResPath)[0] # work with the low res templates
	templateMag = readSectionCoordinates(templateMagsLowResPath)[0]

	userTemplateInputLowRes = readSectionCoordinates(userTemplateInputPath, downFactor = downsizingFactor)
	userTemplateTissueLowRes = userTemplateInputLowRes[0]
	userTemplateMagLowRes = userTemplateInputLowRes[1]
	
	t0, t1, t2, t3 = userTemplateTissueLowRes
	m0, m1, m2, m3 = userTemplateMagLowRes
	
	print 'userTemplateInputLowRes', userTemplateInputLowRes
	print 'userTemplateTissueLowRes', userTemplateTissueLowRes
	print 'userTemplateMagLowRes', userTemplateMagLowRes
	
	completeSection = [m0, t1, t2, m3] # containing mag and tissue
	sectionExtent = longestDiagonal(completeSection)
	
	completeSectionCenter = barycenter(completeSection)
	
	extractingBoxSize = int(sectionExtent * 3)
	extractingBox = Rectangle(int(round(completeSectionCenter[0] - extractingBoxSize/2.)) ,  int(round(completeSectionCenter[1] - extractingBoxSize/2.)), extractingBoxSize, extractingBoxSize)
	templateOriginalAngle = getAngle([m0[0], m0[1], m1[0], m1[1]]) # radian
	
	
	# the important reference is the center of the completeSection: calculate positions relative to that center
	
	# rotate the corners around the center of the completeSection in trakem
	rotTransform = AffineTransform.getRotateInstance(-templateOriginalAngle, completeSectionCenter[0], completeSectionCenter[1])
	rotatedTissueInWaferCoordinates = applyTransform(userTemplateTissueLowRes, rotTransform)
	rotatedMagInWaferCoordinates = applyTransform(userTemplateMagLowRes, rotTransform)
	
	rtw0, rtw1, rtw2, rtw3 = rotatedTissueInWaferCoordinates # rtw stands for rotated tissue in wafer coordinates
	rmw0, rmw1, rmw2, rmw3 = rotatedMagInWaferCoordinates # rmw stands for rotated mag in wafer coordinates
	
	# get the locations of the rotated template corners relative to the center of the section
	rtc0 = [rtw0[0] - completeSectionCenter[0], rtw0[1] - completeSectionCenter[1]] # rtc stands for Rotated Tissue in Center coordinates (relative to the center of the completeSection)
	rtc1 = [rtw1[0] - completeSectionCenter[0], rtw1[1] - completeSectionCenter[1]]
	rtc2 = [rtw2[0] - completeSectionCenter[0], rtw2[1] - completeSectionCenter[1]]
	rtc3 = [rtw3[0] - completeSectionCenter[0], rtw3[1] - completeSectionCenter[1]]
	
	rmc0 = [rmw0[0] - completeSectionCenter[0], rmw0[1] - completeSectionCenter[1]] # rtc stands for Rotated mag (=dapi) in Center coordinates (relative to the center of the completeSection)
	rmc1 = [rmw1[0] - completeSectionCenter[0], rmw1[1] - completeSectionCenter[1]]
	rmc2 = [rmw2[0] - completeSectionCenter[0], rmw2[1] - completeSectionCenter[1]]
	rmc3 = [rmw3[0] - completeSectionCenter[0], rmw3[1] - completeSectionCenter[1]]

	# get the locations of the rotated template corners relative to the center of the *dapi center*
	dapiCenter = barycenter([rmc0, rmc1, rmc2, rmc3]) # in completeSection coordinates
	
	rtd0 = [rtc0[0] - dapiCenter[0], rtc0[1] - dapiCenter[1]] # rtd stands for Rotated Tissue in dapiCenter coordinates (relative to the center of the dapi center)
	rtd1 = [rtc1[0] - dapiCenter[0], rtc1[1] - dapiCenter[1]]
	rtd2 = [rtc2[0] - dapiCenter[0], rtc2[1] - dapiCenter[1]]
	rtd3 = [rtc3[0] - dapiCenter[0], rtc3[1] - dapiCenter[1]]
	
	rmd0 = [rmc0[0] - dapiCenter[0], rmc0[1] - dapiCenter[1]] # rtd stands for Rotated Tissue in dapiCenter coordinates (relative to the center of the dapi center)
	rmd1 = [rmc1[0] - dapiCenter[0], rmc1[1] - dapiCenter[1]]
	rmd2 = [rmc2[0] - dapiCenter[0], rmc2[1] - dapiCenter[1]]
	rmd3 = [rmc3[0] - dapiCenter[0], rmc3[1] - dapiCenter[1]]
	
	completeSectionInCenterCoordinates = [rmc0, rtc1, rtc2, rmc3] # centerCoordinates: the center is the center of the complete section
	expandedSection = shrink(completeSectionInCenterCoordinates, -0.2) # 20% bigger
	templateBoxInCenterCoordinates = sectionToPoly(expandedSection).getBounds()
	
	# final crop with the templateBox in the coordinates of the extracted image
	templateBox = Rectangle(int(round(templateBoxInCenterCoordinates.x + extractingBox.width/2.)),
	int(round(templateBoxInCenterCoordinates.y + extractingBox.height/2.)),
	int(round(templateBoxInCenterCoordinates.width)),
	int(round(templateBoxInCenterCoordinates.height)))
	
	# Get the corner coordinates in the new coordinate system of the extracted template
	tt0 = [int(round(rtc0[0] + templateBox.width/2.)), int(round(rtc0[1] + templateBox.height/2.))] # tt stands for tissue in template coordinates
	tt1 = [int(round(rtc1[0] + templateBox.width/2.)), int(round(rtc1[1] + templateBox.height/2.))]
	tt2 = [int(round(rtc2[0] + templateBox.width/2.)), int(round(rtc2[1] + templateBox.height/2.))]
	tt3 = [int(round(rtc3[0] + templateBox.width/2.)), int(round(rtc3[1] + templateBox.height/2.))]
	
	mt0 = [int(round(rmc0[0] + templateBox.width/2.)), int(round(rmc0[1] + templateBox.height/2.))] # mt stands for mag in template coordinates
	mt1 = [int(round(rmc1[0] + templateBox.width/2.)), int(round(rmc1[1] + templateBox.height/2.))]
	mt2 = [int(round(rmc2[0] + templateBox.width/2.)), int(round(rmc2[1] + templateBox.height/2.))]
	mt3 = [int(round(rmc3[0] + templateBox.width/2.)), int(round(rmc3[1] + templateBox.height/2.))]
	
	templateDapiCenter = barycenter([mt0, mt1, mt2, mt3]) # the center of dapi is the barycenter of the template mag section
	
	edgeFreeSectionTemplateCoordinates = shrink([tt0, tt1, tt2, tt3], 0.2) # 80% of the tissue box should be free of edges (a flipped section would have edges in this region)
	edgeFreeSectionCenterCoordinates = shrink([rtc0, rtc1, rtc2, rtc3], 0.2)
	
	magBox = sectionToPoly([mt0, mt1, mt2, mt3]).getBounds()
	
	project, loader, layerset, nLayers = fc.openTrakemProject(projectPathBis)
	layerNames = ['BF', 'DAPI', '488', '546', 'dapiWekaed', 'nothing', 'rawEdged']

	for l, layer in enumerate(layerset.getLayers()):
		finalTemplatePath = os.path.join(workingFolder, 'finalTemplate_' + layerNames[l] + '.tif')
		
		extractedTemplate = loader.getFlatImage(layer, extractingBox , 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layer.getAll(Patch), True, Color.black, None)
		rotate(extractedTemplate, -templateOriginalAngle * 180 /float(PI))

		if layerNames[l] == 'dapiWekaed': # the template is simply the white rectangle ... I could also make an artificial one
			extractedTemplate.setRoi(templateBox)
			extractedTemplate = extractedTemplate.crop()
			
			extractedTemplate.setRoi(magBox)
			finalTemplate = extractedTemplate.crop()
			IJ.save(finalTemplate, finalTemplatePath)
		else:
			extractedTemplate.setRoi(templateBox)
			finalTemplate = extractedTemplate.crop()
		
			IJ.save(finalTemplate, finalTemplatePath)
		
			allPoints = [tt0, tt1, tt2, tt3, mt0, mt1, mt2, mt3, templateDapiCenter] + edgeFreeSectionTemplateCoordinates
			print 'allPoints', allPoints
			poly = Polygon([point[0] for point in allPoints], [point[1] for point in allPoints], len(allPoints))

			finalTemplate.setRoi(PointRoi(poly.xpoints, poly.ypoints, poly.npoints))
			flattenedTemplate = finalTemplate.flatten()
			flattenedTemplatePath = os.path.join(workingFolder, 'finalTemplate_WithPoints_' + layerNames[l] + '.tif')
			
			IJ.save(flattenedTemplate, flattenedTemplatePath)
			# finalTemplate.show()
	
	fc.closeProject(project)
	# 8/0
	
	##############################################
	# Find the DAPI centers
	##############################################
	if not os.path.isfile(dapiCentersPath):
		dapiWekaed = IJ.openImage(dapiWekaedPath)
		
		# preprocess the dapiWekaed to create good separated components
		dapiWekaed = fc.minMax(dapiWekaed, 110, 255)
		dapiWekaed = fc.blur(dapiWekaed, 5)
		dapiWekaed = fc.minMax(dapiWekaed, 50, 160)
		dapiWekaed = fc.blur(dapiWekaed, 5)
		dapiWekaed = fc.minMax(dapiWekaed, 80, 120)
		IJ.run(dapiWekaed, 'Gray Morphology', 'radius=10 type=circle operator=erode')
		dapiWekaed = fc.minMax(dapiWekaed, 245, 245)

		templateMag = readSectionCoordinates(templateMagsLowResPath)[0]
		dapiCenters = getConnectedComponents(dapiWekaed, minSize = getArea(templateMag)/10 ) # to adjust probably
		IJ.log('There are ' + str(len(dapiCenters)) + ' dapiCenters')

		with open(dapiCentersPath, 'w') as f:
			pickle.dump(dapiCenters, f)

	candidateEdgedFolder = fc.mkdir_p(os.path.join(workingFolder, 'candidateEdgedFolder'))
	candidateErodedFolder = fc.mkdir_p(os.path.join(workingFolder, 'candidateErodedFolder'))
	candidateRawFolder = fc.mkdir_p(os.path.join(workingFolder, 'candidateRawFolder'))
	candidateRawEdgedFolder = fc.mkdir_p(os.path.join(workingFolder, 'candidateRawEdgedFolder'))
	
	candidateHighResRawFolder = fc.mkdir_p(os.path.join(workingFolder, 'candidateHighResRawFolder'))

	#######################
	# Exporting all candidates centered on the dapiCenters
	#######################
	if len(os.listdir(candidateRawFolder)) == 0:
		IJ.log('Exporting all candidates centered on the dapiCenters')
		project, loader, layerset, nLayers = fc.openTrakemProject(projectPathBis)
		dapiCenters = loader.deserialize(dapiCentersPath)
		with open(dapiCentersPath, 'r') as f:
			dapiCenters = pickle.load(f)
		
		# 1. Determining the size of the candidates based on the template size
		templateSections = readSectionCoordinates(templateSectionsPath, downFactor = downsizingFactor)[0] # why plural ?
		sectionExtent = int(round(longestDiagonal(templateSections)))
		IJ.log('Section extent is ' + str(sectionExtent) + ' pixels')
		
		candidateWidth = sectionExtent * 3 # *2 because the candidate will be rotated and cropped for matching
		candidateHeight = sectionExtent * 3
		
		bounds = layerset.get2DBounds()
		layerRaw = layerset.getLayers().get(0)
		layerWekaed = layerset.getLayers().get(nChannels)
		layerEdged = layerset.getLayers().get(nChannels + 1)
		layerRawEdged = layerset.getLayers().get(nChannels + 2)
		
		for idSection, dapiCenter in enumerate(dapiCenters):
			x, y = dapiCenter

			rawPath = os.path.join(candidateRawFolder, 'candidate_' + str(idSection).zfill(4) + '_Raw.tif')
			thresPath = os.path.join(candidateErodedFolder, 'candidate_' + str(idSection).zfill(4)  + '_Wekaed.tif')
			edgesPath = os.path.join(candidateEdgedFolder, 'candidate_' + str(idSection).zfill(4) + '_Edges.tif')
			rawEdgesPath = os.path.join(candidateRawEdgedFolder, 'candidate_' + str(idSection).zfill(4) + '_RawEdges.tif')

			roiExport = Rectangle(int(round(x - candidateWidth/2)), int(round(y - candidateHeight/2)), candidateWidth, candidateHeight)
			IJ.log('x ' + str(x) + '; y ' + str(y) + ' roiexport' + str(roiExport))

			# save the raw image
			rawPatch = loader.getFlatImage(layerRaw, roiExport , 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layerRaw.getAll(Patch), True, Color.black, None)
			IJ.save(rawPatch, rawPath)
			
			# save the thresholded image
			thresholdedPatch = loader.getFlatImage(layerWekaed, roiExport , 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layerWekaed.getAll(Patch), True, Color.black, None)
			IJ.save(thresholdedPatch, thresPath)

			# save the edges image
			patchEdges = loader.getFlatImage(layerEdged, roiExport , 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layerEdged.getAll(Patch), True, Color.black, None)
			IJ.save(patchEdges, edgesPath)

			# save the edges image
			patchRawEdges = loader.getFlatImage(layerRawEdged, roiExport , 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layerRawEdged.getAll(Patch), True, Color.black, None)
			IJ.save(patchRawEdges, rawEdgesPath)
			
		project.save()
		fc.closeProject(project)

	#######################
	# Exporting all candidates centered on the dapiCenters of the high res raw wafer
	#######################
	if len(os.listdir(candidateHighResRawFolder)) == 0:
		IJ.log('Exporting all candidates centered on the dapiCenters')
		project, loader, layerset, nLayers = fc.openTrakemProject(projectPathFullRes)
		dapiCenters = loader.deserialize(dapiCentersPath)
		with open(dapiCentersPath, 'r') as f:
			dapiCenters = pickle.load(f)

		dapiCentersHighRes = [ [dapiCenter[0]*float(downsizingFactor), dapiCenter[1]*float(downsizingFactor)] for dapiCenter in dapiCenters]
		

		# 1. Determining the size of the candidates based on the saved low res candidates
		candidate0 = IJ.openImage(os.path.join(candidateRawFolder, os.listdir(candidateRawFolder)[0]))
		candidateWidth = int(round(candidate0.getWidth() * downsizingFactor))
		candidateHeight = int(round(candidate0.getHeight() * downsizingFactor))
		candidate0.close()
		
		bounds = layerset.get2DBounds()
		layerRaw = layerset.getLayers().get(0)
		
		for idSection, dapiCenter in enumerate(dapiCentersHighRes):
			x, y = dapiCenter

			rawPath = os.path.join(candidateHighResRawFolder, 'candidate_' + str(idSection).zfill(4) + '_Raw.tif')

			roiExport = Rectangle(int(round(x - candidateWidth/2)), int(round(y - candidateHeight/2)), candidateWidth, candidateHeight)
			IJ.log('x ' + str(x) + '; y ' + str(y) + ' roiexport' + str(roiExport))

			# save the raw image
			rawPatch = loader.getFlatImage(layerRaw, roiExport , 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layerRaw.getAll(Patch), True, Color.black, None)
			IJ.save(rawPatch, rawPath)
			
		project.save()
		fc.closeProject(project)


	#####################
	# Template matching
	#####################

	if not os.path.isfile(sectionsSpecsPath):
		
		# #############################
		# # run matching for raw images
		# #############################
		# rawTemplateMatchingPath = os.path.join(workingFolder, 'finalTemplate_' + layerNames[0] + '.tif')
		# rawTemplate = IJ.openImage(rawTemplateMatchingPath)
		# wTemplate = rawTemplate.getWidth()
		# hTemplate = rawTemplate.getHeight()
		# rawTemplate.close()
		
		# atom = AtomicInteger(0)
		# rawSectionResults = []
		# candidatePaths = [os.path.join(candidateRawFolder, candidateName) for candidateName in fc.naturalSort(os.listdir(candidateRawFolder))]	
		# fc.startThreads(templateMatchCandidate, fractionCores = 1, wait = 0, arguments = (atom, candidatePaths, rawTemplateMatchingPath, rawSectionResults))
		# IJ.log(str(rawSectionResults))
		
		#############################
		# run matching for dapiWekaed images
		#############################
		
		# the dapi template has a different size, the w/hTemplate must be updated
		dapiTemplateMatchingPath = os.path.join(workingFolder, 'finalTemplate_' + layerNames[4] + '.tif')
		dapiTemplate = IJ.openImage(dapiTemplateMatchingPath)
		wTemplate = dapiTemplate.getWidth()
		hTemplate = dapiTemplate.getHeight()
		dapiTemplate.close()		
		
		atom = AtomicInteger(0)
		dapiSectionResults = []
		candidatePaths = [os.path.join(candidateErodedFolder, candidateName) for candidateName in fc.naturalSort(os.listdir(candidateErodedFolder))]
		candidateEdgedPaths = [os.path.join(candidateRawEdgedFolder, candidateName) for candidateName in fc.naturalSort(os.listdir(candidateRawEdgedFolder))]
		fc.startThreads(templateMatchCandidate, fractionCores = 0.8, wait = 0, arguments = (atom, candidatePaths, dapiTemplateMatchingPath, dapiSectionResults))
		IJ.log(str(dapiSectionResults))
		# 8/0
		
		sectionsSpecs = [] # list with nDapiCenter elements containing [sectionId, r, x, y]
		for id, dapiSectionResult in enumerate(dapiSectionResults):
			sectionId = dapiSectionResult[0]
			# candidate = IJ.openImage(candidatePaths[sectionId])
			cand = IJ.openImage(candidateEdgedPaths[sectionId])
			sectionResults = dapiSectionResult[1:]
			
			wCandidate = cand.getWidth()
			hCandidate = cand.getHeight()
			
			solutionRank = 0 # counter in the while loop to go through the solutions until a non-flipped section is found
			foundNonFlippedSection = False
			while (not foundNonFlippedSection) and (solutionRank<50) :
				candidate = cand.duplicate()
				bestResult = sectionResults[solutionRank] # because it has been sorted
				ccScore, [r, x, y] = bestResult

				
				rotate(candidate, r)
				
				# the center of the template sliding window
				xWindowCenter = wCandidate/2. - neighborhood/2. + xStep * x
				yWindowCenter = hCandidate/2. - neighborhood/2. + yStep * y

				# # show center of template window
				# poly = Polygon([int(round(xWindowCenter))], [int(round(yWindowCenter))], 1)
				# candidate.setRoi(PointRoi(poly.xpoints, poly.ypoints, poly.npoints))
				# candidate.show()
				
				tissueDapiCoordinates = [rtd0, rtd1, rtd2, rtd3] # tissue corners in dapiCenter coordinates
				edgeFreeDapiCoordinates = shrink(tissueDapiCoordinates, tissueShrinkingForEdgeFreeRegion)
				
				# edgeFreeDapiCoordinates is in relative dapiCenter coordinates, need to transform in coordinates of the current image
				translationToCandidateDapiCenter = AffineTransform.getTranslateInstance(int(round(wCandidate/2.)), int(round(hCandidate/2.)))
				edgeFreeSection = applyTransform(edgeFreeDapiCoordinates, translationToCandidateDapiCenter)
				
				# # show edgeFree section : the region that should be edge free when a section is not flipped
				edgeFreePoly = Polygon([int(round(point[0])) for point in edgeFreeSection], [ int(round(point[1])) for point in edgeFreeSection], len(edgeFreeSection))
				
				candidate = fc.minMax(candidate, 60, 200)
				
				edgeFreePoly = PolygonRoi(edgeFreePoly, Roi.POLYGON)
				candidate.setRoi(edgeFreePoly)
				edgeMeasure = candidate.getStatistics(Measurements.MEAN).mean
				print 'edgeMeasure', edgeMeasure
				candidate.setTitle( str(sectionId) + ' - ' + str(int(edgeMeasure)) )

				if edgeMeasure < 1:
					foundNonFlippedSection = True
					sectionsSpecs.append([sectionId, r, x, y, ccScore])
					# candidate.show()
				
				solutionRank = solutionRank + 1

		with open(sectionsSpecsPath, 'w') as f:
			pickle.dump(sectionsSpecs, f)

	####################
	# Write the sections in wafer coordinates		
	####################
	if not os.path.isfile(tissueSectionsHighResPath):
		with open(dapiCentersPath, 'r') as f:
			waferDapiCenters = pickle.load(f)
			
		with open(sectionsSpecsPath, 'r') as f:
			sectionsSpecs = pickle.load(f)

		candidate0 = IJ.openImage(os.path.join(candidateRawFolder, os.listdir(candidateRawFolder)[0]))
		wCandidate = candidate0.getWidth() # *2 because the candidate will be rotated and cropped for matching
		hCandidate = candidate0.getHeight()
		candidate0.close()

		magSectionsLowRes = []
		tissueSectionsLowRes = []
		
		magSectionsHighRes = []
		tissueSectionsHighRes = []

		completeSectionDapiCenter = [rtd0, rtd1, rtd2, rtd3, rmd0, rmd1, rmd2, rmd3] # relative to the dapi center
		
		for sectionSpecs in sectionsSpecs:
			sectionId, r, x, y, ccScore = sectionSpecs
			waferDapiCenterLowRes = waferDapiCenters[sectionId]
			# waferDapiCenterHighRes = [waferDapiCenterLowRes[0]*float(downsizingFactor), waferDapiCenterLowRes[1]*float(downsizingFactor)]

			
			# found dapiCenter during the matching search relative to the dapiCenter
			xWindowCenter = - neighborhood/2. + xStep * x
			yWindowCenter = - neighborhood/2. + yStep * y

			transform = AffineTransform()
			translate1 = AffineTransform.getTranslateInstance(xWindowCenter, yWindowCenter) # /!\ I believe correct, but hard to confirm ...
			translate2 = AffineTransform.getTranslateInstance(waferDapiCenterLowRes[0], waferDapiCenterLowRes[1])
			rotateTransform = AffineTransform.getRotateInstance(-r * PI/float(180))
			
			transform.concatenate(translate1)
			transform.concatenate(translate2)
			transform.concatenate(rotateTransform)
			completeSectionWafer = applyTransform(completeSectionDapiCenter, transform)
			
			# completeSectionWafer = map(lambda x: [int(round(x[0])), int(round(x[1]))], completeSectionWafer) # necessary ?
			
			magSectionLowRes = completeSectionWafer[4:]
			tissueSectionLowRes = completeSectionWafer[:4]
			
			magSectionsLowRes.append(magSectionLowRes)
			tissueSectionsLowRes.append(tissueSectionLowRes)

			magSectionHighRes = [ [point[0]*float(downsizingFactor), point[1]*float(downsizingFactor)] for point in magSectionLowRes]
			tissueSectionHighRes = [ [point[0]*float(downsizingFactor), point[1]*float(downsizingFactor)] for point in tissueSectionLowRes]

			magSectionsHighRes.append(magSectionHighRes)
			tissueSectionsHighRes.append(tissueSectionHighRes)
			
		writeSections(magSectionsLowResPath, magSectionsLowRes)
		writeSections(tissueSectionsLowResPath, tissueSectionsLowRes)

		writeSections(magSectionsHighResPath, magSectionsHighRes)
		writeSections(tissueSectionsHighResPath, tissueSectionsHighRes)

	# 8/0
			
	################################
	# GUI to adjust existing sections
	################################
	if fc.getOK('Do you want to manually adjust the sections ?'):
		with open(dapiCentersPath, 'r') as f:
			waferDapiCenters = pickle.load(f)
		with open(sectionsSpecsPath, 'r') as f:
			sectionsSpecs = pickle.load(f)
		
		# the following ensures that all manual adjustments are systematically saved after each adjustment
		if not os.path.isfile(finalTissueSectionsPath):
			shutil.copyfile(magSectionsHighResPath, finalMagSectionsPath)
			shutil.copyfile(tissueSectionsHighResPath, finalTissueSectionsPath)
		
		tissueSectionsHighRes = readSectionCoordinates(finalTissueSectionsPath)
		magSectionsHighRes = readSectionCoordinates(finalMagSectionsPath)
		
		candidate0 = IJ.openImage(os.path.join(candidateRawFolder, os.listdir(candidateRawFolder)[0]))
		wCandidate = candidate0.getWidth()
		hCandidate = candidate0.getHeight()
		candidate0.close()

		candidateHighResRawPaths = [os.path.join(candidateHighResRawFolder, name) for name in os.listdir(candidateHighResRawFolder)]
		sectionStart = fc.getNumber('At which section do you want to start ?', default = 0, decimals = 0)

		magSectionsLowRes = []
		tissueSectionsLowRes = []
		
		for id, sectionSpecs in enumerate(sectionsSpecs):
			if id > sectionStart-1:
				sectionId, r, x, y, ccScore = sectionSpecs
				IJ.log('Manually checking section ' + str(id) + ' out of ' + str(len(sectionSpecs)) + '(the real id of the section is ' + str(sectionId) + ')')
				waferDapiCenterLowRes = waferDapiCenters[sectionId]
				waferDapiCenterHighRes = [waferDapiCenterLowRes[0]*float(downsizingFactor), waferDapiCenterLowRes[1]*float(downsizingFactor)]

				completeSectionDapiCenter = [rtd0, rtd1, rtd2, rtd3, rmd0, rmd1, rmd2, rmd3] # relative to the dapi center

				# the center of the template sliding window
				xWindowCenter = wCandidate/2. - neighborhood/2. + xStep * x
				yWindowCenter = hCandidate/2. - neighborhood/2. + yStep * y
				
				# completeSectionDapiCenter is in relative dapiCenter coordinates, need to transform in coordinates of the candidate
				translationToCandidateCoordinates = AffineTransform.getTranslateInstance(int(round(wCandidate/2.)), int(round(hCandidate/2.)))
				completeSectionInLowResCandidate = applyTransform(completeSectionDapiCenter, translationToCandidateCoordinates)

				# in coordinates of the high res candidate
				completeSectionInHighResCandidate = [[point[0]*float(downsizingFactor), point[1]*float(downsizingFactor)] for point in completeSectionInLowResCandidate]
				
				# display the section points for user to adjust
				# create roi
				poly = Polygon([int(round(point[0])) for point in completeSectionInHighResCandidate], [int(round(point[1])) for point in completeSectionInHighResCandidate], len(completeSectionInHighResCandidate))
				
				# show roi and user dialog
				candidate = IJ.openImage(candidateHighResRawPaths[sectionId])
				rotate(candidate, r)
				candidate.setRoi(PointRoi(poly.xpoints, poly.ypoints, poly.npoints))
				candidate.show()
				zoomFactor = 0
				for repeat in range(zoomFactor):
					time.sleep(0.2)
					IJ.run('In [+]')
				w = candidate.getWindow()
				w.setLocation(0,0)
				rrt = RoiRotationTool()
				rrt.run('')
				WindowManager.setCurrentWindow(w)
				WaitForUserDialog('Adjust the section points then click Ok.').show()
				
				# get the user input
				adjustedSectionLocal = []
				for id, roi in enumerate(candidate.getRoi()):
					adjustedSectionLocal.append([roi.x, roi.y])
				candidate.close()
				# transform to dapiCenter coordinates
				translationToDapiCenter = AffineTransform.getTranslateInstance(-int(round(wCandidate*downsizingFactor/2.)), -int(round(hCandidate*downsizingFactor/2.)))
				userInputCompleteSectionDapiCenter = applyTransform(adjustedSectionLocal, translationToDapiCenter)
				
				# calculate transform to transform manual input to wafer coordinates (see details of the calculation earlier)
				# xWindowCenter = (- neighborhood/2. + xStep * x)*downsizingFactor
				# yWindowCenter = (- neighborhood/2. + yStep * y)*downsizingFactor
				transform = AffineTransform()
				# translate1 = AffineTransform.getTranslateInstance(xWindowCenter, yWindowCenter) # /!\ I believe correct, but hard to confirm ...
				translate2 = AffineTransform.getTranslateInstance(waferDapiCenterHighRes[0], waferDapiCenterHighRes[1])
				rotateTransform = AffineTransform.getRotateInstance(-r * PI/float(180))
				# transform.concatenate(translate1)
				transform.concatenate(translate2)
				transform.concatenate(rotateTransform)

				completeSectionWafer = applyTransform(userInputCompleteSectionDapiCenter, transform)
				
				magSectionHighRes = completeSectionWafer[4:]
				tissueSectionHighRes = completeSectionWafer[:4]
				
				magSectionsHighRes[sectionId] = magSectionHighRes
				tissueSectionsHighRes[sectionId] = tissueSectionHighRes

				# magSectionLowRes = [ [point[0]/float(downsizingFactor), point[1]/float(downsizingFactor)] for point in magSectionHighRes]
				# tissueSectionLowRes = [ [point[0]/float(downsizingFactor), point[1]/float(downsizingFactor)] for point in tissueSectionHighRes]
				
				# magSectionsLowRes.append(magSectionLowRes)
				# tissueSectionsLowRes.append(tissueSectionLowRes)
				
				# write after each section (an issue is that if you start this manual proofreading, you need to go through the whole process, could be avoided ...)
				# writeSections(magSectionsLowResPath, magSectionsLowRes)
				# writeSections(tissueSectionsLowResPath, tissueSectionsLowRes)

				writeSections(finalMagSectionsPath, magSectionsHighRes)
				writeSections(finalTissueSectionsPath, tissueSectionsHighRes)
		

		
# # # # # # # # # # # ### Manually adding an offset, 
# # # # # # # # # # # tissueSections = readSectionCoordinates(finalTissueSectionsPath)
# # # # # # # # # # # magSections = readSectionCoordinates(finalMagSectionsPath)
		
# # # # # # # # # # # translate = AffineTransform.getTranslateInstance(103, 128)	

# # # # # # # # # # # tissueSections = [applyTransform(tissueSection, translate) for tissueSection in tissueSections]

# # # # # # # # # # # magSections = [applyTransform(magSection, translate) for magSection in magSections]

# # # # # # # # # # # writeSections(finalTissueSectionsPath, tissueSections)
# # # # # # # # # # # writeSections(finalMagSectionsPath, magSections)

		
#############################
# GUI to catch missed sections
#############################
if fc.getOK('Do you want to catch missed sections ?'):

	shutil.copyfile(projectPathFullRes, overlaysProjectPath)
	p, loader, layerset, nLayers = fc.openTrakemProject(overlaysProjectPath)
	p.saveAs(overlaysProjectPath, True)

	if os.path.isfile(finalTissueSectionsPath):
		tissueSections = readSectionCoordinates(finalTissueSectionsPath)
		magSections = readSectionCoordinates(finalMagSectionsPath)
	else:
		tissueSections = readSectionCoordinates(tissueSectionsHighResPath)
		magSections = readSectionCoordinates(magSectionsHighResPath)
	
	counter = 0
	addSectionOverlays(p, [0], magSections, [Color.yellow], [0.5], 'allMagSectionsWith_' + str(counter) + '_manualSections')
	addSectionOverlays(p, [0], tissueSections, [Color.blue], [0.5], 'allTissueSectionsWith_' + str(counter) + '_manualSections')

	noSectionAdded = False
	while not noSectionAdded:
		counter = counter + 1
		newSections = getPointsFromUser(p, 0, text = 'Select 4 corners in the right order of 1. the tissue 2. the mag.')
		print 'newSections', newSections
		if newSections != None:
			newSectionsTissue = [newSections[8*k: 8*k + 4] for k in range(len(newSections)/8)]
			newSectionsMag = [newSections[8*k + 4: 8*k + 8] for k in range(len(newSections)/8)]
				
			magSections = magSections + newSectionsMag
			tissueSections = tissueSections + newSectionsTissue

			addSectionOverlays(p, [0], newSectionsMag, [Color.yellow], [0.5], 'manualMagSections_' + str(counter))
			addSectionOverlays(p, [0], newSectionsTissue, [Color.blue], [0.5], 'manualTissueSections_' + str(counter))
			# # # # # forceAlphas(layerset)
			# update the starting point
			p.save()
		else:
			noSectionAdded = True
			
		writeSections(finalTissueSectionsPath, tissueSections)
		writeSections(finalMagSectionsPath, magSections)

	IJ.log('Writing the final image coordinates in the preImaging folder')
	writeSections(finalTissueSectionsPreImagingPath, tissueSections)
	writeSections(finalMagSectionsPreImagingPath, magSections)
	
	p.save()
	fc.closeProject(p)
	# disp = Display(p, layerset.getLayers().get(0))
	# disp.showFront(layerset.getLayers().get(0))
	# fc.closeProject(p)


#######################
# Extract images for all sections 
#######################
finalExtractedSectionsFolder = fc.mkdir_p(os.path.join(workingFolder, 'finalExtractedSections'))
blendFlagPath = os.path.join(workingFolder, 'finalHighResBlendFlag')
if len(os.listdir(finalExtractedSectionsFolder)) == 0:
	magSections = readSectionCoordinates(finalMagSectionsPath)
	tissueSections = readSectionCoordinates(finalTissueSectionsPath)
	
	boxSize = longestDiagonal(magSections[0]) * 3
	IJ.log('Exporting all candidates centered on the dapiCenters')
	project, loader, layerset, nLayers = fc.openTrakemProject(projectPathFullRes)
	
	# for idSection, dapiCenter in enumerate(dapiCentersHighRes):
	for l, layer in enumerate(layerset.getLayers()):
		if not os.path.isfile(blendFlagPath):
			# blending: /!\ takes some time
			IJ.log('Blending the high res wafer: takes some time ... channel ' + str(channels[l]))
			Blending.blendLayerWise(layerset.getLayers(l, l), True, None)
		
		for sectionId, magSection in enumerate(magSections):
			x, y = barycenter(magSection + tissueSections[sectionId])
			angle = getAngle([magSection[0][0], magSection[0][1], magSection[1][0], magSection[1][1]])

			extractedSectionPath = os.path.join(finalExtractedSectionsFolder, 'finalSection_' + str(sectionId).zfill(4) + '_' + channels[l] + '.tif')

			roiExport = Rectangle(int(round(x - boxSize/2)), int(round(y - boxSize/2)), boxSize, boxSize)
			IJ.log('x ' + str(x) + '; y ' + str(y) + ' roiexport' + str(roiExport))

			# save the raw image
			extractedSection = loader.getFlatImage(layer, roiExport , 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layer.getAll(Patch), True, Color.black, None)
			rotate(extractedSection, -angle * 180/float(PI))
			
			IJ.save(extractedSection, extractedSectionPath)

	with open(blendFlagPath, 'w') as f:
		f.write('blended done')
	project.save()
	fc.closeProject(project)

#######################
# Sanity check
#######################
tissueSections = readSectionCoordinates(finalTissueSectionsPath)
magSections = readSectionCoordinates(finalMagSectionsPath)

# for id, tissueSection in enumerate(tissueSections):
	# IJ.log(str(id) + ' - ' + str(int(getArea(tissueSection)/100)))

for id, magSection in enumerate(magSections):
	IJ.log(str(id) + ' - ' + str(int(getArea(magSection)/100)))

# for id, [tissueSection, magSection] in enumerate(zip(tissueSections, magSections)): # distance between the barycenters of mag and tissue
	# b1 = barycenter(tissueSection)
	# b2 = barycenter(magSection)
	# IJ.log(str(id) + ' - ' + str(int(round(Math.sqrt((b1[0] - b2[0]) * (b1[0] - b2[0]) + (b1[1] - b2[1]) * (b1[1] - b2[1]))))))