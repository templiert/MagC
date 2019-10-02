#this script puts the acquired EM tiles into Trakem at the right positions
from __future__ import with_statement
import os, re, errno, string, shutil, time

from java.awt.event import MouseAdapter, KeyEvent, KeyAdapter
from jarray import zeros, array
from java.util import HashSet, ArrayList
from java.awt.geom import AffineTransform
from java.awt import Color

from ij import IJ, Macro
from ij.io import Opener, FileSaver
from fiji.tool import AbstractTool
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch
from ini.trakem2.utils import Utils
from ini.trakem2.display import Display, Patch
from mpicbg.trakem2.align import Align, AlignTask

import sys
sys.path.append(IJ.getDirectory('plugins'))
import fijiCommon as fc 

from java.lang import Thread
from java.util.concurrent.atomic import AtomicInteger
from java.lang import Runtime

def parallelStitch(atom, foldersToStitch, allPatchCoordinates):
	while atom.get() < len(foldersToStitch):
		k = atom.getAndIncrement()
		if (k < len(foldersToStitch)):
			sectionFolder = foldersToStitch[k]

			tileConfigurationPath = os.path.join(sectionFolder, 'TileConfiguration_' + str(k).zfill(4) + '.registered.txt')

			stitchCommand = 'type=[Filename defined position] order=[Defined by filename         ] grid_size_x=' + str(numTilesX) + ' grid_size_y=' + str(numTilesY) + ' tile_overlap=' + str(100 * (tileOverlapX + tileOverlapY)/2.) + ' first_file_index_x=0 first_file_index_y=0 directory=' + sectionFolder + ' file_names=Tile_{xx}-{yy}_resized_' + factorString + '.tif output_textfile_name=TileConfiguration_' + str(k).zfill(4) +'.txt fusion_method=[Do not fuse images (only write TileConfiguration)] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap subpixel_accuracy computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory=' + sectionFolder
			IJ.log(stitchCommand)
			IJ.run('Grid/Collection stitching', stitchCommand)

			f = open(tileConfigurationPath, 'r')
			lines = f.readlines()[4:] # trimm the heading
			f.close()

			for line in lines:
				# paths
				path = os.path.join(sectionFolder, line.replace('\n', '').split(';')[0])
				#locations
				x = float(line.replace('\n', '').split(';')[2].split(',')[0].split('(')[1])
				y = float(line.replace('\n', '').split(';')[2].split(',')[1].split(')')[0])
				
				allPatchCoordinates.append([path, [x,y], k]) 

namePlugin = 'assembly_lowEM'
MagCFolder = fc.startPlugin(namePlugin)

ControlWindow.setGUIEnabled(False)
MagC_EM_Folder = fc.mkdir_p(os.path.join(MagCFolder, 'MagC_EM',''))
MagCParameters = fc.readMagCParameters(MagCFolder)

downsamplingFactor = MagCParameters['downsample_EM']['downsamplingFactor']
factorString = str(int(1000000*downsamplingFactor)).zfill(8)

# read some metadata
EMMetadataPath = fc.findFilesFromTags(MagCFolder,['EM', 'Metadata'])[0]
EMMetadata = fc.readParameters(EMMetadataPath)
numTilesX = EMMetadata['numTilesX']
numTilesY = EMMetadata['numTilesY']
xPatchEffectiveSize = EMMetadata['xPatchEffectiveSize']
yPatchEffectiveSize = EMMetadata['yPatchEffectiveSize']
tileOverlapX = EMMetadata['tileOverlapX']
tileOverlapY = EMMetadata['tileOverlapY']
nbLayers = EMMetadata['nSections']
IJ.log('There are ' + str(nbLayers) + ' EM layers')

##########################################
# Stitching of the low-res EM project
##########################################
allPatchCoordinates = []

downsampledFolder = os.path.join(MagC_EM_Folder, 'MagC_EM_' + factorString)
IJ.log('downsampledFolder ' + downsampledFolder)

foldersToStitch = [os.path.join(downsampledFolder, folderName) for folderName in os.walk(downsampledFolder).next()[1]]

# stitching should be done in parallel but the stitching plugin does not seem to run in parallel, so fractionCores=0 -> only one core used ...
atom = AtomicInteger(0)
fc.startThreads(parallelStitch, fractionCores = 0, wait = 0, arguments = (atom, foldersToStitch, allPatchCoordinates))

paths = [coordinates[0] for coordinates in allPatchCoordinates]
locations = [coordinates[1] for coordinates in allPatchCoordinates]
layers = [coordinates[2] for coordinates in allPatchCoordinates]

# create the low-res trakem project with the computed stitching coordinates
projectName = 'EMProject_' + factorString + '.xml'
projectPath = fc.cleanLinuxPath(os.path.join(MagC_EM_Folder , projectName))
IJ.log('Creating the Trakem project ' + projectName)

project, loader, layerset, nbLayers = fc.getProjectUtils(fc.initTrakem(fc.cleanLinuxPath(MagC_EM_Folder), nbLayers))
project.saveAs(projectPath, True)
time.sleep(1) # probably useless
loader.setMipMapsRegeneration(False)

importFilePath = fc.createImportFile(MagC_EM_Folder, paths, locations, layers = layers, name = namePlugin + factorString)

IJ.log('I am going to insert many files at factor ' + str(downsamplingFactor))
task = loader.importImages(layerset.getLayers().get(0), importFilePath, '\t', 1, 1, False, 1, 0)
task.join()

time.sleep(2)
fc.resizeDisplay(layerset)
time.sleep(2)
project.save()
time.sleep(2)

# save all transforms into one file
transformsPath = os.path.join(MagC_EM_Folder , namePlugin + '_Transforms.txt')
fc.writeAllAffineTransforms(project, transformsPath)

fc.closeProject(project)
IJ.log('Assembling the low EM project done and saved into ' + projectPath)
fc.terminatePlugin(namePlugin, MagCFolder)