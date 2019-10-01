from __future__ import with_statement
from ij import IJ
from ij import Macro
import os
import time
import sys
sys.path.append(IJ.getDirectory('plugins'))
import fijiCommon as fc 
from mpicbg.trakem2.align import RegularizedAffineLayerAlignment
from java.awt.geom import AffineTransform 
from java.awt import Rectangle
from java.util import HashSet
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch
from ini.trakem2.imaging import Blending

from java.lang import Runtime, Thread 
from java.util.concurrent.atomic import AtomicInteger

def exportLayer():
	while atom.get() < min(nLayers, currentWrittenLayer + nLayersAtATime + 1) :
		k = atom.getAndIncrement()
		if k < min(nLayers, currentWrittenLayer + nLayersAtATime):
			IJ.log('Start exporting layer ' + str(k) + ' currentWrittenLayer - ' + str(currentWrittenLayer))
			fc.exportFlat(project, exportFolder, 1/float(LMEMFactor), baseName = 'alignedDownsampledEM', bitDepth = 8, layers = [k])

namePlugin = 'export_alignedEMForRegistration'
MagCFolder = fc.startPlugin(namePlugin)
ControlWindow.setGUIEnabled(False)

MagCParams = fc.readMagCParameters(MagCFolder)
nLayersAtATime = MagCParams[namePlugin]['nLayersAtATime']
nThreads = MagCParams[namePlugin]['nThreads']

LMEMFactor = fc.getLMEMFactor(MagCFolder)
IJ.log('Exporting with LMEMFactor = ' + str(LMEMFactor))

projectPath = fc.findFilesFromTags(MagCFolder,['EM', 'Project'])[0]
exportFolder = fc.mkdir_p(os.path.join(os.path.dirname(projectPath), namePlugin))
project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)
temporaryFolder = fc.mkdir_p(os.path.join(os.path.dirname(projectPath), 'temporary_LMEMRegistration')) # to save contrasted images

# currentLayerPath stores in a file the current layer being processed by the script which is run several times
currentLayerPath = os.path.join(os.path.dirname(projectPath), 'currentLayer_' + namePlugin + '.txt')
currentWrittenLayer = fc.incrementCounter(currentLayerPath, increment = nLayersAtATime)

atom = AtomicInteger(currentWrittenLayer)
fc.startThreads(exportLayer, wait = 0, nThreads = nThreads)

# project.save() # why do I save the project here ?
time.sleep(3)

fc.shouldRunAgain(namePlugin, currentWrittenLayer, nLayers, MagCFolder, project, increment = nLayersAtATime)