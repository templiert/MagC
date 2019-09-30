from __future__ import with_statement
from ij import IJ
from ij import Macro
import os, time
import sys
sys.path.append(IJ.getDirectory('plugins'))
import fijiCommon as fc 
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch
from java.util.concurrent.atomic import AtomicInteger
from ij.measure import Measurements

def contrastImage():
	while atomicI.get() < nPaths:
		k = atomicI.getAndIncrement()
		if k < nPaths:
			im = IJ.openImage(toContrastPaths[k][0])
			im = fc.normLocalContrast(im, normLocalContrastSize, normLocalContrastSize, 3, True, True)
			IJ.save(im, toContrastPaths[k][1])
			im.close()

namePlugin = 'assembly_LM'
MagCFolder = fc.startPlugin(namePlugin)

ControlWindow.setGUIEnabled(False)

# get some parameters
MagCParameters = fc.readMagCParameters(MagCFolder)
normLocalContrastSize = MagCParameters[namePlugin]['normLocalContrastSize'] # size of the neighborhood for local contrast for the brightfield channel
overlap = MagCParameters[namePlugin]['overlap'] # overlap between tiles, typically 0.1
refChannelIdentifier = str(MagCParameters[namePlugin]['refChannelIdentifier'])

normLocalContrastSizeFluo = MagCParameters[namePlugin]['normLocalContrastSizeFluo'] # for contrasting the fluo channels
minMaxFluo = MagCParameters[namePlugin]['minMaxFluo'] # for thresholding the fluo channels
flipHorizontally = MagCParameters[namePlugin]['flipHorizontally'] # flip horizontally the LM tiles

# initialize folders
LMDataFolder = os.path.join(MagCFolder, 'LMDataReordered')
LMFolder = fc.mkdir_p(os.path.join(MagCFolder, 'MagC_LM'))

# read metadata
LMMetadataPath = os.path.join(LMDataFolder, 'LM_Metadata.txt')
width, height, nChannels, xGrid, yGrid, scaleX, scaleY, channels = fc.readSessionMetadata(MagCFolder)

# get reference channel name
IJ.log('The reference channel identifier is: ' + refChannelIdentifier)
refChannel = filter(lambda x: refChannelIdentifier in x, channels)[0]
IJ.log('channels ' + str(channels))
IJ.log('refChannel: ' + refChannel)


# Preprocess the reference brightfield channel: 8-bit with mean sensible range pulled from all images
# find thresholding range
IJ.log('Reading all images of the channel to find a good intensity range: ' + str(refChannel) )
mins, maxs, imPaths = [], [], []
for (dirpath, dirnames, filenames) in os.walk(LMDataFolder):
	for filename in filenames:
		if (os.path.splitext(filename)[1] == '.tif') and (refChannel in filename): 
			IJ.log('Reading to determine 8-biting thresholding range: ' + filename)
			imPath = os.path.join(dirpath, filename)
			imPaths.append(imPath)
			im = IJ.openImage(imPath)
			stats = im.getStatistics(Measurements.MIN_MAX)
			mins.append(stats.min)
			maxs.append(stats.max)
meanMin = sum(mins)/float(len(mins))
meanMax = sum(maxs)/float(len(maxs))
IJ.log('The channel min/max is ' + str(meanMin) + ' - ' + str(meanMax))
# apply thresholding
for imPath in imPaths:
	im = IJ.openImage(imPath)
	im = fc.minMax(im, meanMin, meanMax)
	IJ.run(im, '8-bit', '')
	if flipHorizontally:
		IJ.run(im, 'Flip Horizontally', '') # {Leica DMI, NikonTiEclipse} to Merlin
	IJ.save(im, imPath)
	IJ.log('Image ' + imPath + ' thresholded 8-bited')

# process all other channels that are not refChannel: normLocalContrast, threshold, 8-bit
for (dirpath, dirnames, filenames) in os.walk(LMDataFolder):
	for filename in filenames:
		if (os.path.splitext(filename)[1] == '.tif') and not(refChannel in filename): 
			imPath = os.path.join(dirpath, filename)
			im = IJ.openImage(imPath)
			im = fc.normLocalContrast(im, normLocalContrastSizeFluo, normLocalContrastSizeFluo, 3, True, True)
			im = fc.minMax(im, minMaxFluo[0], minMaxFluo[1])
			IJ.run(im, '8-bit', '')
			if flipHorizontally:
				IJ.run(im, 'Flip Horizontally', '') # Leica DMI to Merlin
				IJ.log('flipHorizontally')
			IJ.save(im, imPath)
			IJ.log('Image ' + imPath + ' processed')
			
# add a contrasted reference channel (e.g., contrast the brightfield channel)
IJ.log('Adding a contrasted channel')
contrastedChannel = 'contrasted' + refChannel
channels.append(contrastedChannel)
toContrastPaths = []

for (dirpath, dirnames, filenames) in os.walk(LMDataFolder):
	for filename in filenames:
		IJ.log('ToContrast: ' + str(filename))
		if (os.path.splitext(filename)[1] == '.tif') and (refChannel in filename):
			imagePath = os.path.join(dirpath, filename)
			contrastedPath = os.path.join(dirpath, filename.replace(refChannel, contrastedChannel) )
			toContrastPaths.append([imagePath, contrastedPath])
IJ.log('toContrastPaths : ' + str(toContrastPaths))
nPaths = len(toContrastPaths)
atomicI = AtomicInteger(0)
fc.startThreads(contrastImage)

# Update metadata with the new contrasted channel
f = open(LMMetadataPath, 'r')
lines = f.readlines()			
for idLine, line in enumerate(lines):
	if 'nChannels' in line:
		lines[idLine] = 'nChannels = ' + str(nChannels + 1)
	if 'channels' in line:
		lines[idLine] = 'channels = [' + ','.join( map(lambda x: "'" + x + "'", channels) ) + ']'
f.close()
f = open(LMMetadataPath, 'w')
for line in lines:
	f.write(line + '\n')
f.close()
IJ.log('addContrastedChannel done')

# Create LM project with the contrastedChannel
nLayers = len(next(os.walk(LMDataFolder))[1])
IJ.log('nLayers is ' + str(nLayers))
IJ.log('Creating trakem project "LMProject" ')

project, loader, layerset, nLayers = fc.getProjectUtils(fc.initTrakem(LMFolder, nLayers))
projectPath = os.path.join(os.path.normpath(LMFolder) , 'LMProject.xml')
project.saveAs(projectPath, True)

# determining tiles locations taking into account the overlap
paths, locations, layers = [], [], []
widthEffective = int((1-overlap) * width)
heightEffective = int((1-overlap) * height)

for channel in [contrastedChannel]:
	IJ.log('Assembling LM sections from all layers')
	for l in range(nLayers):
		IJ.log('Each section consists of ' + str(xGrid) + ' x ' + str(yGrid) + ' patches')
		for y in range(yGrid):
			for x in range(xGrid):
				sectionFolder = os.path.join(LMDataFolder, 'section_' + str(l).zfill(4))
				patchName = 'section_' + str(l).zfill(4) + '_channel_' + channel + '_tileId_' + str(x).zfill(2) + '-' + str(y).zfill(2) + '-tissue.tif'
				patchPath = os.path.join(sectionFolder, patchName)
				paths.append(patchPath)
				locations.append([x*widthEffective, y*heightEffective])
				layers.append(l)
# import all tiles
importFile = fc.createImportFile(LMFolder, paths, locations, layers = layers)
task = loader.importImages(layerset.getLayers().get(0), importFile, '\t', 1, 1, False, 1, 0)
task.join()

# resize display and save
fc.resizeDisplay(layerset)
time.sleep(5)
project.save()
time.sleep(1)
fc.closeProject(project)
IJ.log('Assembling the LM project done and saved into ' + projectPath)

fc.terminatePlugin(namePlugin, MagCFolder)
