from __future__ import with_statement
# Python imports
import os, re, errno, string, shutil, ntpath, sys, time, threading
from sets import Set

# Java imports
from java.util import ArrayList, HashSet
from java.awt import Rectangle, Color
from java.awt.geom import AffineTransform
from jarray import zeros, array
from java.lang import Math, Thread, Runtime
from java.util.concurrent.atomic import AtomicInteger

# Fiji imports
import ij
from ij import IJ, Macro, ImagePlus, WindowManager, ImageStack
from ij.gui import WaitForUserDialog
from ij.process import ImageStatistics as IS
from ij.process import ImageConverter
import ij.io.OpenDialog
from ij.io import DirectoryChooser, FileSaver
#from ij.gui import GenericDialog
#from ij.gui import NonBlockingGenericDialog #fails without display
from ij.plugin.filter import GaussianBlur as Blur
from ij.plugin.filter import Filters
from ij.process import ByteProcessor, FloatProcessor

from mpicbg.ij.clahe import Flat
from mpicbg.ij.plugin import NormalizeLocalContrast
from mpicbg.models import RigidModel2D, AffineModel2D, Point, PointMatch
from mpicbg.trakem2.transform import CoordinateTransformList

from fiji.tool import AbstractTool
from fiji.selection import Select_Bounding_Box

# TrakEM imports
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Display, Patch
from ini.trakem2.utils import Utils
from ini.trakem2.io import CoordinateTransformXML
from ini.trakem2.tree import LayerTree
from mpicbg.trakem2.align import Align, AlignTask

from register_virtual_stack import Register_Virtual_Stack_MT
from register_virtual_stack import Transform_Virtual_Stack_MT


################
# File and I/O operations
################

def folderFromPath(path): #folders have an ending os.sep
	head, tail = ntpath.split(path)
	return head + os.sep

def nameFromPath(path):
	head, tail = ntpath.split(path)
	return os.path.splitext(tail)[0]

def folderNameFromFolderPath(path):
	head, tail = ntpath.split(path)
	head, tail = ntpath.split(head)
	return tail

def mkdir_p(path):
	path = os.path.join(path, '')
	try:
		os.mkdir(path)
		IJ.log('Folder created: ' + path)
	except Exception, e:
		if e[0] == 20047:
			# IJ.log('Nothing done: folder already existing: ' + path)
			pass
		else:
			IJ.log('Exception during folder creation :' + str(e))
	return path

def promptDir(text):
	folder = DirectoryChooser(text).getDirectory()
	content = naturalSort(os.listdir(folder))
	IJ.log('Prompted for ' + text)
	IJ.log('Selected folder :'  + folder)
	return folder, content

def makeNeighborFolder(folder, name):
	neighborFolder = folderFromPath(folder.rstrip(os.sep)) + name + os.sep
	mkdir_p(neighborFolder)
	IJ.log('NeighborFolder created: ' + neighborFolder)
	return neighborFolder

def getPath(text):
	path = IJ.getFilePath(text)
	IJ.log('File selected: ' + path)
	return path

def naturalSort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def getOK(text):
	gd = GenericDialog('User prompt')
	gd.addMessage(text)
	gd.hideCancelButton()
	gd.enableYesNoCancel()
	gd.showDialog()
	return gd.wasOKed()

def getName(text, defaultName = ''):
	gd = GenericDialog(text)
	gd.addStringField(text, defaultName)
	gd.showDialog()
	if gd.wasCanceled():
		print 'User canceled dialog!'
		return
	return gd.getNextString()

def getNumber(text, default = 0, decimals = 3):
	gd = GenericDialog(text)
	gd.addNumericField(text, default, decimals)  # show 6 decimals
	gd.showDialog()
	if gd.wasCanceled():
		IJ.log('User canceled dialog!')
		return
	return gd.getNextNumber()

def getNamesFromFolderExt(folder, extension = '.tif'):
	list = os.listdir(folder)
	list = filter(lambda x: os.path.splitext(x)[1] == extension, list)
	list = naturalSort(list)
	# for i, name in enumerate(list):
		# list[i] = folder + name
	return list

def findFilesFromTags(folder,tags):
	IJ.log('Looking for files in ' + folder + ' that match the following tags: ' + str(tags))
	filePaths = []
	for (dirpath, dirnames, filenames) in os.walk(folder):
		for filename in filenames:
			if (all(map(lambda x:x in filename,tags)) == True):
				path = os.path.join(dirpath, filename)
				filePaths.append(path)
				IJ.log('Found this file: ' + path)
	filePaths = naturalSort(filePaths)
	return filePaths

def findFoldersFromTags(folder,tags):
	IJ.log('Looking for folders in ' + folder + ' that match the following tags: ' + str(tags))
	folderPaths = []
	for (dirpath, dirnames, filenames) in os.walk(folder):
		for dirname in dirnames:
			if (all(map(lambda x:x in dirname,tags)) == True):
				path = os.path.join(dirpath,dirname,'')
				folderPaths.append(path)
				IJ.log('Found this folder: ' + path)
	folderPaths = naturalSort(folderPaths)
	return folderPaths

#def displayInfoDialog(text, title = 'Info'):
#	global infoDialog
#	infoDialog = NonBlockingGenericDialog(title)
#	infoDialog.addMessage(text)
#	infoDialog.setLocation(0,0)
#	infoDialog.setSize(400,800)
#	infoDialog.show()
#	return infoDialog

def toString(*a):
	return ''.join(map(str,a))

def getRefChannel(channels, text = 'Reference Channel'): #dialog prompt to choose the reference channel
	gd = GenericDialog(text)
	gd.addChoice("output as", channels, channels[0])
	gd.showDialog()
	if gd.wasCanceled():
		print "User canceled dialog!"
		return
	return gd.getNextChoice()

def getMinMaxFor8Bit(minMaxs):
	gd = GenericDialog('Enter the min and max for each channel for the 8 bit transformation')
	for channel in minMaxs.keys():
		gd.addNumericField('Min ' + channel , minMaxs[channel][0], 2) # show 2 decimals
		gd.addNumericField('Max ' + channel, minMaxs[channel][1], 2)
	gd.showDialog()
	if gd.wasCanceled():
		IJ.log('User canceled dialog!')
		return
	for channel in minMaxs.keys():
		minMaxs[channel][0] = gd.getNextNumber()
		minMaxs[channel][1] = gd.getNextNumber()
	return minMaxs

def cleanLinuxPath(path):
	if path[:2] == os.sep + os.sep:
		returnPath = path[1:]
		IJ.log('Path cleaned from ' + path + '\n to \n' + returnPath)
	else:
		returnPath = path
	return returnPath

####################
# Affine Transforms
####################

def readWriteXMLTransform(projectPath,layerIndex,folder):
	'''
	1-Take the TrakEM project 'projectPath'
	2-Read the transformation of the first patch of layer 'layerIndex'
	3-Read the locations of the first patch of layer 'layerIndex' and layer '1-layerIndex'. This is used to calculate the offset of the EM and LM images in the initial transformation file
	4-Write as a simple text file in folder + title of the project + InitialTransform.txt
	'''
	project = Project.openFSProject(projectPath, False)
	layerset = project.getRootLayerSet()
	layers = layerset.getLayers()
	layer = layerset.getLayers().get(layerIndex)
	patches = layer.getDisplayables(Patch)
	t =  patches.get(0).getAffineTransform()

	transPath = folder + project.getTitle() + 'InitialTransform.txt.txt'
	f = open(transPath,'w')
	f.write( str(t.getScaleX()) + "\n")
	f.write( str(t.getShearY())+ "\n")
	f.write( str(t.getShearX())+ "\n")
	f.write( str(t.getScaleY())+ "\n")
	f.write( str(t.getTranslateX())+ "\n")
	f.write( str(t.getTranslateY())+ "\n")

	f.write ( str( layers.get(layerIndex).getDisplayables(Patch).get(0).getX() ) + '\n')
	f.write ( str( layers.get(layerIndex).getDisplayables(Patch).get(0).getY() ) + '\n')
	f.write ( str( layers.get(1-layerIndex).getDisplayables(Patch).get(0).getX() ) + '\n')
	f.write ( str( layers.get(1-layerIndex).getDisplayables(Patch).get(0).getY() ) + '\n')
	f.close()
	IJ.log('Transformation saved in: ' + transPath)
	# read the parameters of the transformations
	trans = []
	f = open(transPath,'r')
	while 1:
		line = f.readline()
		if not line: break
		IJ.log(line)
		trans.append(float(line))
	f.close
	IJ.log('Transformation: ' + str(trans))
	closeProject(project)
	return trans

def writeAffineTransforms(project,path):
	with open(path,'w') as f:
		layerset = project.getRootLayerSet()
		for k,layer in enumerate(layerset.getLayers()):
			patch = layer.getDisplayables(Patch).get(0)
			t = patch.getAffineTransform()
			f.write( str(t.getScaleX()) + "\n")
			f.write( str(t.getShearY())+ "\n")
			f.write( str(t.getShearX())+ "\n")
			f.write( str(t.getScaleY())+ "\n")
			f.write( str(t.getTranslateX())+ "\n")
			f.write( str(t.getTranslateY())+ "\n")
	return

def writeAllAffineTransforms(project,path):
	layerset = project.getRootLayerSet()
	loader = project.getLoader()
	with open(path,'w') as f:
		for l,layer in enumerate(layerset.getLayers()):
			for patch in layer.getDisplayables(Patch):
				f.write(os.path.normpath(loader.getAbsolutePath(patch)) + '\n')
				f.write(str(l) + '\n')
				t = patch.getAffineTransform()
				f.write( str(t.getScaleX()) + '\n')
				f.write( str(t.getShearY())+ '\n')
				f.write( str(t.getShearX())+ '\n')
				f.write( str(t.getScaleY())+ '\n')
				f.write( str(t.getTranslateX())+ '\n')
				f.write( str(t.getTranslateY())+ '\n')
	IJ.log('All affine Transforms saved in: ' + path)
	return

def pushTransformsToTopLeft(path, patchesPerSection):
    with open(path, 'r') as f:
        lines = f.readlines()
    # minX = min([float(lines[k].replace('\n',''))
            # for k in range(6, len(lines), 8)])
    # minY = min([float(lines[k].replace('\n',''))
            # for k in range(7, len(lines), 8)])

    minX = [min([float(lines[p].replace('\n',''))
            for p in range(6 + 8*s*patchesPerSection, 6 + 8*s*patchesPerSection + 8*patchesPerSection + 1, 8)])
            for s in range(len(lines)/8/patchesPerSection]
    minY = [min([float(lines[p].replace('\n',''))
            for p in range(7 + 8*s*patchesPerSection, 7 + 8*s*patchesPerSection + 8*patchesPerSection + 1, 8)])
            for s in range(len(lines)/8/patchesPerSection]

    with open(path, 'w') as f:
        for l, line in enumerate(lines):
            if l%8 == 6:
                newLine = float(line.replace('\n','')) - minX
                f.write(str(newLine) + '\n')
            elif l%8 == 7:
                newLine = float(line.replace('\n','')) - minY
                f.write(str(newLine) + '\n')
            else:
				f.write(str(line))

    
    
def readTransform(path):
	IJ.log('Reading transformation file: ' + path)
	trans = []
	f = open(path,'r')
	while 1:
		line = f.readline()
		if not line: break
		IJ.log(line)
		try:
			line = float(line)
		except Exception, e:
			pass
		trans.append(float(line))
	f.close
	return trans

def readCoordinates(folder,tags):
	content = os.listdir(folder)
	for i in content:
		if (all(map(lambda x:x in i,tags)) == True):
			path = folder + i
			IJ.log('This file matched the tag --' + str(tags) + '-- in the folder ' + folder + ' : ' + path)
	f = open(path,'r')
	x = []
	y = []
	for i, line in enumerate(f):
		x.append(int(line.split("\t")[0]))
		y.append(int(line.split("\t")[1]))
	#x,y = map(lambda u: np.array(u),[x,y]) #fiji version, no numpy
	f.close()
	IJ.log('x = ' + str(x))
	IJ.log('y = ' + str(y))
	return x,y

def readSectionCoordinates(path):
	sections = []
	try:
		f = open(path, 'r')
		lines = f.readlines()
		for line in lines:
			points = line.split('\t')
			points.pop()
			section = [ [int(float(point.split(',')[0])), int(float(point.split(',')[1]))] for point in points ]
			sections.append(section)
		f.close()
	except Exception, e:
		IJ.log('Section coordinates not found. It is probably a simple manual run.')
	return sections
	
def writeRectangle(rectangle,path):
	with open(path,'w') as f:
		f.write(str(rectangle.x) + '\n')
		f.write(str(rectangle.y) + '\n')
		f.write(str(rectangle.width) + '\n')
		f.write(str(rectangle.height) + '\n')

def readRectangle(path):
	with open(path,'r') as f:
		res = []
		for i, line in enumerate(f):
			res.append(int(line))
	return Rectangle(res[0],res[1],res[2],res[3])

#################
# Image operations
#################

def crop(im,roi):
	ip = im.getProcessor()
	ip.setRoi(roi)
	im = ImagePlus(im.getTitle() + '_Cropped', ip.crop())
	return im

def localContrast(im, block = 127, histobins = 256, maxslope = 3):
	ipMaskCLAHE = ByteProcessor(im.getWidth(),im.getHeight())
	ipMaskCLAHE.threshold(-1)
	bitDepth = im.getBitDepth()
	if bitDepth == 8:
		maxDisp = Math.pow(2,8) - 1
	else:
		maxDisp = Math.pow(2,12) - 1

	ip = im.getProcessor()
	ip.setMinAndMax(0,maxDisp)
	if bitDepth == 8:
		ip.applyLut()
	Flat.getFastInstance().run(im, block, histobins, maxslope, ipMaskCLAHE, False)
	del ipMaskCLAHE
	return im

def edges(im):
	filter = Filters()
	ip = im.getProcessor()
	filter.setup('edge',im)
	filter.run(ip)
	im = ImagePlus(os.path.splitext(im.getTitle())[0] + '_Edges',ip)
	return im

def blur(im,sigma):
	blur = Blur()
	ip = im.getProcessor()
	blur.blurGaussian(ip,sigma,sigma,0.0005)
	im = ImagePlus(os.path.splitext(im.getTitle())[0] + '_Blur',ip)
	return im

def normLocalContrast(im, x, y, stdev, center, stretch):
	NormalizeLocalContrast().run(im.getProcessor(), x, y, stdev, center, stretch) # something like repaint needed ?
	return im

def resize(im,factor):
	IJ.run(im, 'Size...', 'width=' + str(int(Math.floor(im.width * factor))) + ' height=' + str(int(Math.floor(im.height * factor))) + ' average interpolation=Bicubic')
	return im

def minMax(im, min, max):
	# ip = im.getProcessor()
	# ip.setMinAndMax(min,max)
	# ip.applyLut()
	IJ.setMinAndMax(im, min, max)
	IJ.run(im, 'Apply LUT', '')
	return im

def to8Bit(*args):
	im = args[0]
	if len(args)==1:
		min = 0
		max = 4095
	else:
		min = args[1]
		max = args[2]
	ip = im.getProcessor()
	ip.setMinAndMax(min,max)
	IJ.run(im, '8-bit', '')
	return im

def stackFromPaths(paths):
	firstIm = IJ.openImage(paths[0])
	width = firstIm.getWidth()
	height = firstIm.getHeight()
	firstIm.close()

	ims = ImageStack(width, height) # assemble the ImageStack of the channel
	for path in paths:
		ims.addSlice(IJ.openImage(path).getProcessor())
	imp = ImagePlus('Title', ims)
	imp.setDimensions(1, 1, len(paths)) # these have to be timeframes for trackmate
	return imp

def rawToPeakEnhanced(im, min = 200, max = 255):
	im.getProcessor().invert()
	IJ.run(im, "Normalize Local Contrast", "block_radius_x=15 block_radius_y=15 standard_deviations=3 center stretch")
	IJ.run(im, "Median...", "radius=1")
	minMax(im, 200, 255)
	return im

def getModelFromPoints(sourcePoints, targetPoints):
	rigidModel = RigidModel2D()
	pointMatches = HashSet()
	for a in zip(sourcePoints, targetPoints):
		pm = PointMatch(Point([a[0][0], a[0][1]]), Point([a[1][0], a[1][1]]))
		pointMatches.add(pm)
	rigidModel.fit(pointMatches)
	return rigidModel

##############
# Trakem utils
##############

def initTrakem(path, nbLayers, mipmaps = False): #initialize a project
	path = cleanLinuxPath(path)
	ControlWindow.setGUIEnabled(False)
	project = Project.newFSProject("blank", None, path)
	project.getLoader().setMipMapsRegeneration(mipmaps)
	layerset = project.getRootLayerSet()
	for i in range(nbLayers): # create the layers
		layerset.getLayer(i, 1, True)
	project.getLayerTree().updateList(layerset) #update the LayerTree
	Display.updateLayerScroller(layerset) # update the display slider
	IJ.log('TrakEM project initialized with ' + str(nbLayers) + ' layers and stored in ' + path + ' (but not saved yet)')
	return project

def initProject(path, nbLayers, mipmaps = False): #initialize a project
	path = cleanLinuxPath(path)
	ControlWindow.setGUIEnabled(False)
	project = Project.newFSProject("blank", None, path)
	loader = project.getLoader()
	loader.setMipMapsRegeneration(mipmaps)
	layerset = project.getRootLayerSet()
	for i in range(nbLayers): # create the layers
		layerset.getLayer(i, 1, True)
	project.getLayerTree().updateList(layerset) #update the LayerTree
	Display.updateLayerScroller(layerset) # update the display slider
	IJ.log('TrakEM project initialized with ' + str(nbLayers) + ' layers and stored in ' + path + ' (but not saved yet)')
	return project, loader, layerset

def exportFlat(project,outputFolder,scaleFactor, baseName = '', bitDepth = 8, layers = [], roi = ''):
	layerset = project.getRootLayerSet()
	loader = project.getLoader()
	for l,layer in enumerate(layerset.getLayers()):
		if (layers ==[] ) or (l in layers):
			IJ.log('Exporting layer ' + str(l))
			if roi == '':
				roiExport = layerset.get2DBounds()
			else:
				roiExport = roi
			if bitDepth == 8:
				imp = loader.getFlatImage(layer,roiExport,scaleFactor, 0x7fffffff, ImagePlus.GRAY8, Patch, layer.getAll(Patch), True, Color.black, None)
			elif bitDepth == 16:
				imp = loader.getFlatImage(layer,roiExport,scaleFactor, 0x7fffffff, ImagePlus.GRAY16, Patch, layer.getAll(Patch), True, Color.black, None)
			savePath = os.path.join(outputFolder, baseName + '_' + str(l).zfill(4) + '.tif')
			IJ.save(imp, savePath)
			IJ.log('Layer ' + str(l) +' flat exported to ' + savePath)
			imp.close()

def exportFlatForPresentations(project,outputFolder,scaleFactor,rectangle):
	layerset = project.getRootLayerSet()
	loader = project.getLoader()
	for l,layer in enumerate(layerset.getLayers()): # import the patches
		IJ.log('Exporting layer ' + str(l) + 'with rectangle ' + str(rectangle) + 'scale factor ' + str(scaleFactor))
		imp = loader.getFlatImage(layer,rectangle,scaleFactor, 0x7fffffff, ImagePlus.GRAY8, Patch, layer.getAll(Patch), True, Color.black, None)
		IJ.save(imp,outputFolder + os.sep + nameFromPath(outputFolder.rstrip(os.sep)) + '_' + str(l) + '.tif') #use the name of the outputFolder to name the images
		IJ.log('Layer ' + str(l)+' flat exported to ' + outputFolder + os.sep + nameFromPath(outputFolder.rstrip(os.sep)) + '_' + str(l) + '.tif')
		imp.close()

def exportFlatCloseFiji(project,outputFolder,scaleFactor):
	#todo: check whether the output file already exists. If yes, skip
	for l,layer in enumerate(layerset.getLayers()):
		savePath = outputFolder + os.sep + nameFromPath(outputFolder.rstrip(os.sep)) + '_' + str(l) + '.tif'
		savePathNext = outputFolder + os.sep + nameFromPath(outputFolder.rstrip(os.sep)) + '_' + str(l+1) + '.tif'
		if os.isfile(savePathNext):
			IJ.log('Skipping layer ' + str(l) + ': already processed')
		else:
			IJ.log('Exporting layer ' + str(layer) + '; layer number ' + str(l))
		layerset = project.getRootLayerSet()
		loader = project.getLoader()
		imp = loader.getFlatImage(layer,layerset.get2DBounds(),scaleFactor, 0x7fffffff, ImagePlus.GRAY8, Patch, layer.getAll(Patch), True, Color.black, None)
		IJ.save(imp, savePath)
		IJ.log('Layer ' + str(layerCurrent)+' flat exported to ' + savePath)
		imp.close()
	IJ.log('exportFlatCloseFiji has reached the end')

def exportFlatRoi(project, scaleFactor, x, y, w, h, layer, saveName):
	loader = project.getLoader()
	rectangle = Rectangle(x-int(w/2),y-int(h/2),w,h)
	patches = layer.find(Patch, x, y)
	print patches
	# IJ.log('patches' + str(patches))

	for p, patch in enumerate(patches):
		visible = patch.visible
		patch.visible = True
		tiles = ArrayList()
		tiles.add(patch)
		print 'tiles',tiles
		print 'rectangle',rectangle
		IJ.log('Patch ' + str(patch) + ' cropped with rectangle ' + str(rectangle) )
		imp = loader.getFlatImage(layer, rectangle, scaleFactor, 0x7fffffff, ImagePlus.GRAY8, Patch, tiles, True, Color.black, None)
		exportName = saveName + '_' + str(int(p)) + '.tif'
		IJ.save(imp, exportName)
		patch.visible = visible

def closeProject(project):
	try:
		project.getLoader().setChanged(False) #no dialog if there are changes
		project.destroy()
	except Exception, e:
		IJ.log('Was asked to close a project, but failed (probably no open project available)')
		pass

def resizeDisplay(layerset):
	# layerset.setDimensions(1,1,1,1)
	# layerset.enlargeToFit(layerset.getDisplayables(Patch))
	layerset.setMinimumDimensions()

def setChannelVisible(project,channel):
	layerset = project.getRootLayerSet()
	layers = layerset.getLayers()
	for l,layer in enumerate(layers):
		patches = layer.getDisplayables(Patch)
		for patch in patches:
			patchName = nameFromPath(patch.getImageFilePath())
			if patchName[0:len(channel)] == channel:
				patch.setVisible(True, True)
			# else:
				# patch.setVisible(False, True)
		try:
			Display.getFront().updateVisibleTabs()
		except Exception, e:
			IJ.log('Did not succeed in updating the visible tabs')
			pass

def setChannelInvisible(project,channel):
	layerset = project.getRootLayerSet()
	layers = layerset.getLayers()
	for l,layer in enumerate(layers):
		patches = layer.getDisplayables(Patch)
		for patch in patches:
			patchName = nameFromPath(patch.getImageFilePath())
			if patchName[0:len(channel)] == channel:
				patch.setVisible(False, True)
			Display.getFront().updateVisibleTabs()

def toggleChannel(project,channel):
	layerset = project.getRootLayerSet()
	layers = layerset.getLayers()
	for l,layer in enumerate(layers):
		patches = layer.getDisplayables(Patch)
		for patch in patches:
			# IJ.log('thepatch' + str(patch))
			patchName = nameFromPath(patch.getImageFilePath())
			# IJ.log(str(patchName[0:len(channel)]) + '=' + str(channel) + ' value' + str(patchName[0:len(channel)] == channel) )
			if patchName[0:len(channel)] == channel:
				# IJ.log(str(patch) + ' toggled')
				patch.setVisible((not patch.visible), True)
				Display.getFront().updateVisibleTabs()

def openTrakemProject(path, mipmap = False):
	project = Project.openFSProject(cleanLinuxPath(path), False)
	return getProjectUtils(project, mipmap)

def getProjectUtils(project, mipmap = False):
	loader = project.getLoader()
	loader.setMipMapsRegeneration(mipmap)
	layerset = project.getRootLayerSet()
	nLayers = len(layerset.getLayers())
	return project, loader, layerset, nLayers

def createImportFile(folder, paths, locations, factor = 1, layers = None, name = ''):
	importFilePath = os.path.join(folder, 'trakemImportFile' + name + '.txt')
	with open(importFilePath, 'w') as f:
		for id, path in enumerate(paths):
				xLocation = int(locations[id][0] * factor)
				yLocation = int(locations[id][1] * factor)
				path = cleanLinuxPath(path)
				if layers:
					IJ.log('Inserting image ' + path + ' at (' + str(xLocation) + ' ; ' + str(yLocation) + ' ; ' + str(layers[id]) + ')')
					f.write(str(path) + '\t' + str(xLocation) + '\t' + str(yLocation) + '\t' + str(layers[id]) + '\n')
				else:
					IJ.log('Inserting image ' + path + ' at (' + str(xLocation) + ' ; ' + str(yLocation) + ')' )
					f.write(str(path) + '\t' + str(xLocation) + '\t' + str(yLocation) + '\t' + str(0) + '\n')
	return importFilePath

def rigidAlignment(projectPath, params, name = '', boxFactor = 1):
	# rigid alignment outside trakem2 with register virtual stack plugin because bug in trakem2
	projectFolder = os.path.dirname(projectPath)
	projectName = os.path.splitext(os.path.basename(projectPath))[0]
	project, loader, layerset, nLayers = openTrakemProject(projectPath)

	exportForRigidAlignmentFolder = mkdir_p(os.path.join(projectFolder, 'exportForRigidAlignment'))
	resultRigidAlignmentFolder = mkdir_p(os.path.join(projectFolder, 'resultRigidAlignment'))
	
	bb = layerset.get2DBounds()	
	roi = Rectangle(int(bb.width/2 * (1 - boxFactor)), int(bb.height/2 * (1 - boxFactor)), int(bb.width*boxFactor), int(bb.height*boxFactor))	
	boxPath = os.path.join(resultRigidAlignmentFolder, 'alignmentBox.txt')
	writeRectangle(roi, boxPath)

	exportFlat(project, exportForRigidAlignmentFolder, 1, baseName = 'exportForRigidAlignment', roi = roi)

	referenceName = naturalSort(os.listdir(exportForRigidAlignmentFolder))[0]
	use_shrinking_constraint = 0
	IJ.log('Rigid alignment with register virtual stack')
	Register_Virtual_Stack_MT.exec(exportForRigidAlignmentFolder, resultRigidAlignmentFolder, resultRigidAlignmentFolder, referenceName, params, use_shrinking_constraint)
	time.sleep(2)
	IJ.log('Warning: rigidAlignment closing all existing windows')
	WindowManager.closeAllWindows() # problematic because it also closes the log window
	
	# IJ.getImage().close()

	for l, layer in enumerate(layerset.getLayers()):
		imagePath = os.path.join(exportForRigidAlignmentFolder, 'exportForRigidAlignment_' + str(l).zfill(4) + '.tif')
		transformPath = os.path.join(resultRigidAlignmentFolder, 'exportForRigidAlignment_' + str(l).zfill(4) + '.xml')
		aff = getAffFromRVSTransformPath(transformPath)

		for patch in layer.getDisplayables(Patch):
			patch.setLocation(patch.getX()-roi.x, patch.getY()-roi.y) # compensate for the extracted bounding box
			# patch.setLocation(roi.x, roi.y) # compensate for the extracted bounding box
			currentAff = patch.getAffineTransform()
			currentAff.preConcatenate(aff)
			patch.setAffineTransform(currentAff)

	resizeDisplay(layerset)
	project.save()
	closeProject(project)
	IJ.log('All LM layers have been aligned in: ' + projectPath)

def getAffFromRVSTransformPath(path):
	theList = ArrayList(HashSet())
	read = CoordinateTransformXML.parse(path)

	if type(read) == CoordinateTransformList:
		read.getList(theList)
	else:
		theList.add(read)
	
	if theList.size() == 1: # because RVS either stores only one affine or one affine and one translation
		aff = theList.get(0).createAffine()
	else:
		aff =  theList.get(0).createAffine()
		aff1 = theList.get(1).createAffine()
		aff.preConcatenate(aff1) # option 2
		
	return aff

################
# Pipeline utils
################

def saveLog(path):
	logWindows = WindowManager.getWindow('Log')
	if logWindows: #if no display
		textPanel = logWindows.getTextPanel()
		theLogText =  textPanel.getText().encode('utf-8')
		with open(path,'a') as f:
			f.write('The log has been saved at this time: ' + time.strftime('%Y%m%d-%H%M%S') + '\n')
			f.write(theLogText)
		logWindows.close()
	return

# def readSessionMetadata(folder):
	# sessionMetadataPath = findFilesFromTags(folder,['session','metadata'])[0]
	# with open(sessionMetadataPath,'r') as f:
		# lines = f.readlines()
		# # IJ.log(str(lines[1].replace('\n','').split('\t')))
		# # IJ.log(str(len(lines[1].replace('\n','').split('\t'))))
		# width, height, nChannels, xGrid, yGrid, scaleX, scaleY = lines[1].replace('\n','').split('\t')
		# channels = []
		# # IJ.log(str(lines))
		# # IJ.log(nChannels)
		# for i in range(int(nChannels)):
			# # IJ.log(str(i))
			# l = lines[3+i]
			# # IJ.log(str(l))
			# l = l.replace('\n','')
			# # IJ.log(str(l))
			# l = l.split('\t')
			# # IJ.log(str(l))
			# channels.append(l[0])
	# return int(width), int(height), int(nChannels), int(xGrid), int(yGrid), float(scaleX), float(scaleY), channels

def readSessionMetadata(folder):
	LMMetadataPath = findFilesFromTags(folder,['LM_Metadata'])[0]
	LMMetadata = readParameters(LMMetadataPath)
	channels = map(str, LMMetadata['channels'])
	LMMetadata['channels'] = [str(channel.replace('"', '').replace("'",'')) for channel in channels]
	
	return [LMMetadata[parameter] for parameter in ['width', 'height', 'nChannels', 'xGrid', 'yGrid', 'scaleX' , 'scaleY', 'channels']]

def readMagCParameters(MagCFolder):
	MagCParametersPath = findFilesFromTags(MagCFolder, ['MagC_Parameters'])[0]
	return readParameters(MagCParametersPath)

def readParameters(path):
	key0 = ''
	d = {}
	with open(path,'r') as f:
		lines = f.readlines()
	for line in lines:
		if (len(line) > 12) and line[:12] == '##### Plugin':
			key0 = line.replace(' ','').replace('#','').replace('Plugin','').replace('\n','').replace('\r','')
			if key0 not in d:
				d[key0] = {}
		elif line[0] != '#' and '=' in line:
			line = line.split('=')
			cleanedLine = line[1].replace(' ','').replace('\n','').replace('[', '').replace(']', '').replace('\r', '')
			key = line[0].replace(' ','')
			values = []
			for value in cleanedLine.split(','):
				try:
					if '.' in value:
						value = float(value)
					else:
						value = int(value)
				except Exception, e:
					pass
				values.append(value)

			if len(values) == 1: # to return a single element if there is a single value
				values = values[0]

			if key0 == '': # then this is not the real MagC parameter file with the plugin information. It is a 'normal' parameter file.
				d[key] = values
			else:
				d[key0][key] = values
	return d
	
def startPlugin(namePlugin): # the argument received is always a folder
	IJ.log('Plugin ' + namePlugin + ' started at ' + str(time.strftime('%Y%m%d-%H%M%S')))
	externalArguments = Macro.getOptions().replace('"','').replace(' ','')
	externalArguments = os.path.normpath(externalArguments)
	return externalArguments

def terminatePlugin(namePlugin, MagCFolder, signalingMessage = 'kill me'):
	signalingPath = os.path.join(MagCFolder, 'signalingFile_' + namePlugin + '.txt')
	signalingPath = cleanLinuxPath(signalingPath)
        IJ.log('signalingPath ' + signalingPath)
	logFolder = mkdir_p(os.path.join(MagCFolder, 'MagC_Logs'))
	logPath = os.path.join(logFolder, 'log_' + namePlugin + '.txt')
	IJ.log('Plugin ' + namePlugin + ' terminated at '  + str(time.strftime('%Y%m%d-%H%M%S')))
	saveLog(logPath)
	with open(signalingPath, 'w') as f:
	    f.write(signalingMessage)
	IJ.log('Written: ' + str(signalingPath))
        # IJ.run('Quit')

#ToDo: shouldRunAgain should receive the path of the counter file, so that it can delete the file when it is done
def shouldRunAgain(namePlugin, l, nLayers, MagCFolder, project, increment = 1):
	logFolder = mkdir_p(os.path.join(MagCFolder, 'MagC_Logs'))
	logPath = os.path.join(logFolder, 'log_' + namePlugin + '.txt')
	if l + increment < nLayers:
		IJ.log('Plugin ' + namePlugin + ' still running at '  + str(time.strftime('%Y%m%d-%H%M%S')) + '. ' + str(min(l + increment, nLayers)) + '/' + str(nLayers) + ' done.' )
		time.sleep(3)
		closeProject(project)
		time.sleep(1)
		saveLog(logPath)
		terminatePlugin(namePlugin, MagCFolder, signalingMessage = 'kill me and rerun me')
		# sys.exit(2)
	else:
		IJ.log(namePlugin + ' done. ' + str(namePlugin ) + ' -- ' +  str(l) + ' -- ' + str(nLayers) + ' -- ' + str(increment))
		saveLog(logPath)
		time.sleep(2)
		closeProject(project)
		terminatePlugin(namePlugin, MagCFolder)
		# sys.exit(0)

def incrementCounter(path, increment = 1):
	l=''
	if not os.path.isfile(path):
		l = 0
	else:
		with open(path, 'r') as f:
			l = int(f.readline())
	with open(path, 'w') as f:
		f.write(str(l+increment))
	return l

def getLMEMFactor(MagCFolder):
	LMEMFactorPath = os.path.join(MagCFolder, 'LMEMFactor.txt')
	if not os.path.isfile(LMEMFactorPath):
		# LMEMFactor = getNumber('What is the scaling factor to go from LM to EM ? \n Open brightfield LM exported_scale_1 and an EM exportForAlignment and compare the scales \n (taking into account the factor 20)', default = 7, decimals = 1)
		warningUserInputText = '''**********************************************************************
		**********************************************************************
		****************** WARNING ****************** USER INPUT NEEDED ****************** :
		What is the scaling factor to go from LM to EM ? \n Open brightfield LM exported_scale_1 and an EM exportForAlignment and compare the scales \n (taking into account the factor 20) \n
		Save a file named LMEMFactor.txt containing the factor, e.g., 7.4, in the root MagC folder then rerun the pipeline (probably starting from exportEMForRegistration)
		**********************************************************************
		**********************************************************************
		**********************************************************************
		'''
		print warningUserInputText
		IJ.log(warningUserInputText)
		8/0
	else:
		with open(LMEMFactorPath, 'r') as f:
			line = f.readlines()[0]
			IJ.log(str(line))
			print line
			LMEMFactor = float(line)
	return LMEMFactor

	
def getEMLMScaleFactor(MagCFolder):
	try:
		IJ.log('Reading the EM pixel size')
		EMMetadataPath = findFilesFromTags(MagCFolder,['EM', 'Metadata'])[0]
		EMPixelSize = readParameters(EMMetadataPath)['pixelSize'] # in meters
		IJ.log('The EM pixel size is ' + str(EMPixelSize) + ' m')
		
		IJ.log('Reading the LM pixel size')
		LMMetadataPath = findFilesFromTags(MagCFolder,['LM_Metadata'])[0]
		width, height, nChannels, xGrid, yGrid, scaleX, scaleY, channels = readSessionMetadata(MagCFolder)
		LMPixelSize = scaleX * 1e-6 # in meters
		EMLMScaleFactor = int(round(LMPixelSize / float(EMPixelSize)))
		IJ.log('The EMLMScaleFactor is ' + str(EMLMScaleFactor))
	except Exception, e:
		EMLMScaleFactor = 20
		IJ.log('Warning: the real EMLM scale factor could not be read. Outputing the default value instead: ' + str(EMLMScaleFactor))
	return EMLMScaleFactor

def startThreads(function, fractionCores = 1, wait = 0, arguments = None, nThreads = None):
	threads = []
	if nThreads == None:
		threadRange = range(max(int(Runtime.getRuntime().availableProcessors() * fractionCores), 1))
	else:
		threadRange = range(nThreads)
	IJ.log('ThreadRange = ' + str(threadRange))
	for p in threadRange:
		if arguments == None:
			thread = threading.Thread(target = function)
		else:
			#IJ.log('These are the arguments ' + str(arguments) + 'III type ' + str(type(arguments)))
			thread = threading.Thread(group = None, target = function, args = arguments)
		threads.append(thread)
		thread.start()
		IJ.log('Thread ' + str(p) + ' started')
		time.sleep(wait)
	for idThread, thread in enumerate(threads):
		thread.join()
		IJ.log('Thread ' + str(idThread) + 'joined')

def readOrder(path): # read order from Concorde solution (custom format from script) or from manual solution (1 number per line)
	with open(path, 'r') as f:
		lines = f.readlines()
	order = []
	if len(lines[0].split(' ')) == 2: # concorde TSP format
		lines = lines[1:]
		for line in lines:
			order.append(int(line.split(' ')[0]))
		# remove the dummy city 0 and apply a -1 offset
		order.remove(0)
		for id, o in enumerate(order):
			order[id] = o-1
			
		# save a human-readable file, will be used by the pipeline
		saveFolder = os.path.join(os.path.dirname(os.path.normpath(path)), '')
		orderPath = os.path.join(saveFolder, 'sectionOrder.txt')
		if not os.path.isfile(orderPath):
			with open(orderPath, 'w') as f:
				for index in order:
					f.write(str(index) + '\n')
		else:
			IJ.log('That is weird that I was asked to open the TSP solution file while a human-readable section order already exists')
	else: # simple format, one index per line
		for line in lines:
			order.append(int(line.replace('\n', '')))
	IJ.log('Order read: ' + str(order))
	return order
		
# def reorderProject(projectPath, reorderedProjectPath, order):
	# folder = os.path.dirname(os.path.normpath(projectPath))

	# pReordered, loaderReordered, layersetReordered, nLayers = getProjectUtils( initTrakem(folder, len(order)) )
	# pReordered.saveAs(reorderedProjectPath, True)

	# IJ.log('reorderedProjectPath ' + reorderedProjectPath)

	# project, loader, layerset, nLayers = openTrakemProject(projectPath)

	# for l,layer in enumerate(project.getRootLayerSet().getLayers()):
		# IJ.log('Inserting layer ' + str(l) + '...')
		# reorderedLayer = layersetReordered.getLayers().get(order.index(l))
		# for patch in layer.getDisplayables():
			# patchPath = loader.getAbsolutePath(patch)
			# patchTransform = patch.getAffineTransform()
			
			# newPatch = Patch.createPatch(pReordered, patchPath)
			# reorderedLayer.add(newPatch)
			# newPatch.setAffineTransform(patchTransform)		
	# closeProject(project)
	# resizeDisplay(layersetReordered)
	# pReordered.save()
	# closeProject(pReordered)
	# IJ.log('Project reordering done')

def reorderProject(projectPath, reorderedProjectPath, order):
	if os.path.isfile(reorderedProjectPath):
		os.remove(reorderedProjectPath)
	shutil.copyfile(projectPath, reorderedProjectPath)

	project, loader, layerset, nLayers = openTrakemProject(reorderedProjectPath)
	project.saveAs(reorderedProjectPath, True)

	for l,layer in enumerate(project.getRootLayerSet().getLayers()):
		layer.setZ(order.index(l))
	project.getLayerTree().updateList(layerset)

	project.save()
	closeProject(project)
	IJ.log('Project reordering done')
	
#########
# Garbage ?
#########

def readWriteCurrentIndex(outputFolder,text):
	layerFile = outputFolder + os.sep + 'currentNumber_' + text + '.txt'
	if 	os.path.isfile(layerFile):
		f = open(layerFile,'r')
		layerCurrent = int(f.readline())
		f.close()
	else:
		layerCurrent = 1
	f = open(layerFile,'w')
	f.write(str(layerCurrent + 1))
	f.close()
	IJ.log('Current index called from ' + layerFile + ' : ' + str(layerCurrent))
	return layerCurrent

def from32To8Bit(imPath):
	im = IJ.openImage(imPath)
	im.getProcessor().setMinAndMax(0,255)
	im = ImagePlus(im.getTitle(),im.getProcessor().convertToByteProcessor())
	IJ.save(im,imPath)
