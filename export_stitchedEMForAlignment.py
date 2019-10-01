from __future__ import with_statement

import ij
from ij import IJ, Macro
import os, time

import sys
sys.path.append(IJ.getDirectory('plugins'))
import fijiCommon as fc 

from mpicbg.trakem2.align import RegularizedAffineLayerAlignment

from java.lang import Runtime, Thread 
from java.util.concurrent.atomic import AtomicInteger
from java.util import HashSet
from java.awt import Rectangle
from java.awt.geom import AffineTransform 

from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch
from ini.trakem2.imaging import Blending

def exportLayer():
	while atom.get() < min(nLayers, currentWrittenLayer + nLayersAtATime + 1) :
		k = atom.getAndIncrement()
		if k < min(nLayers, currentWrittenLayer + nLayersAtATime):
			IJ.log('Start exporting layer ' + str(k) + ' currentWrittenLayer - ' + str(currentWrittenLayer))
			fc.exportFlat(project, exportFolder, 1/float(downsamplingFactor), baseName = 'stitchedDownsampledEM', bitDepth = 8, layers = [k])

namePlugin = 'export_stitchedEMForAlignment'
MagCFolder = fc.startPlugin(namePlugin)
ControlWindow.setGUIEnabled(False)

MagCParams = fc.readMagCParameters(MagCFolder)
nLayersAtATime = min (MagCParams[namePlugin]['nLayersAtATime'], Runtime.getRuntime().availableProcessors())
nThreads = MagCParams[namePlugin]['nThreads']

# getting downsamplingFactor
downsamplingFactor = MagCParams['downsample_EM']['downsamplingFactor']

projectPath = fc.cleanLinuxPath(fc.findFilesFromTags(MagCFolder,['EM', 'Project'])[0])
exportFolder = fc.mkdir_p(os.path.join(os.path.dirname(projectPath), namePlugin))
project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)

fc.resizeDisplay(layerset) # has been done in previous script but in case ...

currentLayerPath = os.path.join(os.path.dirname(projectPath), 'currentLayer_' + namePlugin + '.txt')
currentWrittenLayer = fc.incrementCounter(currentLayerPath, increment = nLayersAtATime)

atom = AtomicInteger(currentWrittenLayer)
fc.startThreads(exportLayer, wait = 0, nThreads = nThreads)

# project.save() # why do I save the project here ? To save mipmaps for subsequent faster processing ? Probably not needed ...
time.sleep(3)

fc.shouldRunAgain(namePlugin, currentWrittenLayer, nLayers, MagCFolder, project, increment = nLayersAtATime)