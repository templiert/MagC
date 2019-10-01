from __future__ import with_statement
import os, sys, time

import ij
from ij import IJ, Macro

import sys
sys.path.append(IJ.getDirectory('plugins'))
import fijiCommon as fc 

from mpicbg.trakem2.align import Align, AlignTask, ElasticMontage
from mpicbg.imagefeatures import FloatArray2DSIFT

from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch

from java.lang import Runtime, Thread 
from java.util.concurrent.atomic import AtomicInteger

def elasticMontage():
	IJ.log('Thread called **************************')
	while l.get() < min(nLayers, currentWrittenLayer + nLayersAtATime + 1) :
		k = l.getAndIncrement()
		if k < min(nLayers, currentWrittenLayer + nLayersAtATime):
			IJ.log('Start montaging elastically layer ' + str(k))
			if layerset.getLayers().get(k).getNDisplayables() > 1: # some EM projects have a single large tile
				AlignTask().montageLayers(params, layerset.getLayers(k, k))

namePlugin = 'montage_ElasticEM'
MagCFolder = fc.startPlugin(namePlugin)
ControlWindow.setGUIEnabled(False)

MagCParams = fc.readMagCParameters(MagCFolder)
nLayersAtATime = MagCParams[namePlugin]['nLayersAtATime']
nThreads = MagCParams[namePlugin]['nThreads']

projectPath = fc.cleanLinuxPath(fc.findFilesFromTags(MagCFolder,['EM', 'Project'])[0])

project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)

IJ.log('Sleeping in case the opening of the large project takes some time ...')
time.sleep(20)

# parameters for elastic montage
params = ElasticMontage.Param().clone()
params.bmScale = 0.5
params.bmSearchRadius = 50
params.bmBlockRadius = 50

params.bmMinR = 0.1
params.bmMaxCurvatureR = 100
params.bmRodR = 1

params.bmUseLocalSmoothnessFilter = True
params.bmLocalModelIndex = 3
params.bmLocalRegionSigma = 100
params.bmMaxLocalEpsilon = 12
params.bmMaxLocalTrust = 3

params.isAligned = False
# params.isAligned = True # better to keep it to False
params.tilesAreInPlace = True

params.springLengthSpringMesh = 100
params.stiffnessSpringMesh = 0.1
params.maxStretchSpringMesh = 2000
params.maxIterationsSpringMesh = 1000
params.maxPlateauwidthSpringMesh = 200
params.useLegacyOptimizer = True

# params.dampSpringMesh 
# params.maxNumThreads
# params.visualize

currentLayerPath = os.path.join(os.path.dirname(projectPath), 'currentLayer_' + namePlugin + '.txt')
currentWrittenLayer = fc.incrementCounter(currentLayerPath, increment = nLayersAtATime)
l = AtomicInteger(currentWrittenLayer)

# fc.startThreads(elasticMontage(), wait = 1, nThreads = nThreads) /!\ it does not work I do not understand why. Probably a java6 issue because it works in other scripts in java8 ...

threads = []
for p in range(nThreads):
	thread = Thread(elasticMontage)
	threads.append(thread)
	thread.start()
	time.sleep(0.5)
	
for thread in threads:
	thread.join()


IJ.log( namePlugin + ' layer ' + str(currentWrittenLayer))
fc.resizeDisplay(layerset)
project.save()

IJ.log('Sleeping in case the saving of the large project takes some time ...')
time.sleep(20)

# save all transforms
transformsPath = os.path.join(os.path.dirname(projectPath) , namePlugin + '_Transforms.txt')
if l.get() > nLayers-1:
	fc.writeAllAffineTransforms(project,transformsPath)

fc.shouldRunAgain(namePlugin, currentWrittenLayer, nLayers, MagCFolder, project, increment = nLayersAtATime)