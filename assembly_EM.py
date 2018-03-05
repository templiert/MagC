#this script puts the acquired EM tiles into Trakem at the right positions
from __future__ import with_statement
import os, re, errno, string, shutil, time

from java.awt.event import MouseAdapter, KeyEvent, KeyAdapter
from jarray import zeros, array
from java.util import HashSet, ArrayList
from java.awt.geom import AffineTransform
from java.awt import Color

import ij
from ij import IJ, Macro
from fiji.tool import AbstractTool
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch, Display
from ini.trakem2.utils import Utils
from mpicbg.trakem2.align import Align, AlignTask

import fijiCommon as fc

namePlugin = 'assembly_EM'
MagCFolder = fc.startPlugin(namePlugin)
MagCParameters = fc.readMagCParameters(MagCFolder)

ControlWindow.setGUIEnabled(False)
MagC_EM_Folder = os.path.join(MagCFolder, 'MagC_EM','')

# reading transforms from the low-resolution montage
IJ.log('reading the affine transform parameters from the affine montage')
transformsPath = os.path.join(MagC_EM_Folder, 'assembly_lowEM_Transforms.txt')
with (open(transformsPath, 'r')) as f:
	transforms = f.readlines()
nLayers = max([int(transforms[i]) for i in range(1, len(transforms),8) ]) + 1 # why not, could simply read the EM_Metadata parameters ...

# getting downsamplingFactor
downsamplingFactor = MagCParameters['downsample_EM']['downsamplingFactor']
factorString = str(int(1000000*downsamplingFactor)).zfill(8)

# create the normal resolution project
IJ.log('Creating project with the full size EM images')
projectPath = fc.cleanLinuxPath(os.path.join(MagC_EM_Folder,'EMProject.xml'))
project, loader, layerset, _ = fc.getProjectUtils(fc.initTrakem(fc.cleanLinuxPath(os.path.dirname(projectPath)), 1))
project.saveAs(projectPath, True)

# Assembling all images in the project with the transforms computed on the low res and adjusted with the scale factor
IJ.log('Assembling all images in the project with the transforms computed on the low res and adjusted with the scale factor')
paths = []
locations = []
layers = []
for i in range(0, len(transforms), 8):
	alignedPatchPath = transforms[i]
	alignedPatchName = os.path.basename(alignedPatchPath)
	toAlignPatchName = alignedPatchName.replace('_' + factorString, '').replace('_resized', '')
	toAlignPatchPath = os.path.join(MagCFolder, 'EMData', os.path.basename(os.path.dirname(alignedPatchPath)), toAlignPatchName)
	toAlignPatchPath = fc.cleanLinuxPath(toAlignPatchPath[:-1])  # mysterious trailing character ...
	IJ.log('toAlignPatchPath ' + toAlignPatchPath)
	l = int(transforms[i+1])
	paths.append(toAlignPatchPath)
	locations.append([0,0])	
	layers.append(l)

importFilePath = fc.createImportFile(MagC_EM_Folder, paths, locations, layers = layers, name = namePlugin)

# insert the tiles in the project
IJ.log('I am going to insert many files at factor ' + str(downsamplingFactor) + ' ...')
task = loader.importImages(layerset.getLayers().get(0), importFilePath, '\t', 1, 1, False, 1, 0)
task.join()

# apply the transforms
for i in range(0, len(transforms), 8):
	alignedPatchPath = transforms[i]
	alignedPatchName = os.path.basename(alignedPatchPath)
	toAlignPatchName = alignedPatchName.replace('_' + factorString, '').replace('_resized', '')
	toAlignPatchPath = os.path.join(MagCFolder, 'EMData', os.path.basename(os.path.dirname(alignedPatchPath)), toAlignPatchName)
	toAlignPatchPath = toAlignPatchPath[:-1]  # why is there a trailing something !?
	if toAlignPatchPath[:2] == os.sep + os.sep:
		toAlignPatchPath = toAlignPatchPath[1:]
	IJ.log('toAlignPatchPath ' + toAlignPatchPath)
	l = int(transforms[i+1])
	aff = AffineTransform([float(transforms[i+2]), float(transforms[i+3]), float(transforms[i+4]), float(transforms[i+5]), float(transforms[i+6])*float(downsamplingFactor), float(transforms[i+7])*float(downsamplingFactor) ])
	layer = layerset.getLayers().get(l)
	patches = layer.getDisplayables(Patch)
	thePatch = filter(lambda x: os.path.normpath(loader.getAbsolutePath(x)) == os.path.normpath(toAlignPatchPath), patches)[0]
	thePatch.setAffineTransform(aff)
	thePatch.updateBucket()

time.sleep(2)
fc.resizeDisplay(layerset)
time.sleep(2)
project.save()
fc.closeProject(project)
fc.terminatePlugin(namePlugin, MagCFolder)
