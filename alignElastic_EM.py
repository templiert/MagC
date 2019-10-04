#align rigid least square affine for a TrakEM project
from __future__ import with_statement
import ij
from ij import IJ
from ij import Macro
import os, time
# import subprocess
import sys
sys.path.append(IJ.getDirectory('plugins'))
import fijiCommon as fc 

from mpicbg.trakem2.align import ElasticLayerAlignment
from java.awt.geom import AffineTransform
from java.awt import Rectangle
from java.util import HashSet
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch

from ini.trakem2.utils import Filter

namePlugin = 'alignElastic_EM'
MagCFolder = fc.startPlugin(namePlugin)
ControlWindow.setGUIEnabled(False)

MagCParameters = fc.readMagCParameters(MagCFolder)

preProjectPath = fc.cleanLinuxPath(os.path.join(MagCFolder, 'MagC_EM', 'ToBeElasticAlignedEMProject.xml'))
projectPath = fc.cleanLinuxPath(os.path.join(MagCFolder, 'MagC_EM', 'ElasticAlignedEMProject.xml'))
saveExistingProjectInCasePath = fc.cleanLinuxPath(os.path.join(MagCFolder, 'MagC_EM', 'ElasticAlignedEMProject_ReplacedByLastRun.xml'))

IJ.log('preProjectPath :' + str(preProjectPath))

# check whether the pipeline had been interrupted
if os.path.isfile(preProjectPath):
	IJ.log('preProjectPath exists: ' + str(preProjectPath))
	if os.path.isfile(projectPath):
		IJ.log('projectPath exists: ' + str(projectPath))
		if os.path.isfile(saveExistingProjectInCasePath):
			os.remove(saveExistingProjectInCasePath)
		shutil.copyfile(projectPath, saveExistingProjectInCasePath)
		os.remove(projectPath)
	os.rename(preProjectPath, projectPath) # to have consistency in file naming when the pipeline is interrupted
nLayers = len(os.listdir(os.path.join(MagCFolder, 'MagC_EM', 'export_stitchedEMForAlignment')))

layerOverlap = MagCParameters[namePlugin]['layerOverlap']
nLayersAtATime = MagCParameters[namePlugin]['nLayersAtATime']

currentLayerPath = os.path.join(os.path.dirname(projectPath), 'currentLayer_' + namePlugin + '.txt')
currentLayer = fc.incrementCounter(currentLayerPath, increment = (nLayersAtATime - layerOverlap)) # increment = 1


# # # # TO DELETE UNLESS PARALLEL ELASTIC NEEDED AT SOME POINT
# # # # pluginsFolder = IJ.getDirectory('plugins')
# # # # l = 0
# # # # fijiPath = os.path.join(IJ.getDirectory('imagej'), 'ImageJ-win64.exe')
# # # # pluginPath = os.path.join(pluginsFolder, 'alignElastic_EMBash.bsh')
# # # # IJ.log('WARNING should first allow multiple fiji instances')
# # # # while l < nLayers:
	# # # # command = fijiPath + \
	# # # # " -Dl1=" + str(l) + \
	# # # # " -Dl2=" + str(min(l + nLayersAtATime, nLayers - 1)) + \
	# # # # " -Dprojectpath=" + projectPath + \
	# # # # " -- " + pluginPath
	# # # # IJ.log(str(command))
	# # # # result = subprocess.call(command, shell=True)
	# # # # l = l + (nLayersAtATime - layerOverlap)
# # # # fc.terminatePlugin(namePlugin, MagCFolder)

############################
# Transformation parameters
############################
p = {}

# Block Matching
p[27] = ['layerScale', 0.2]
p[40] = ['searchRadius', 100]
p[25] = ['blockRadius', 500]
p[38] = ['resolutionSpringMesh', 32]

# Correlation filters
p[37] = ['minR', 0.1]
p[30] = ['maxCurvatureR', 100]
p[39] = ['rodR', 1]

# Local smoothness filter
# p[42] = ['useLocalSmoothnessFilter', False]
p[42] = ['useLocalSmoothnessFilter', True]
p[28] = ['localModelIndex', 3]
p[29] = ['localRegionSigma', 1000]
p[32] = ['maxLocalEpsilon', 1000]
p[33] = ['maxLocalTrust', 100]

# Miscellaneous
p[12] = ['isAligned', True]
# p[16] = ['maxNumNeighbors', 2] # should be changed for larger stacks
p[16] = ['maxNumNeighbors', 1] # should be changed for larger stacks

# Approximate optimizer
p[9] = ['desiredModelIndex', 3] # unsure
p[10] = ['expectedModelIndex', 3] # unsure

p[14] = ['maxIterationsOptimize', 1000]
p[18] = ['maxPlateauwidthOptimize', 200]

# Spring mesh
p[41] = ['stiffnessSpringMesh', 0.1]
p[36] = ['maxStretchSpringMesh', 2000]
p[31] = ['maxIterationsSpringMesh', 1000]
p[34] = ['maxPlateauwidthSpringMesh', 200]
p[35] = ['useLegacyOptimizer', True]


p[0] = ['SIFTfdBins',8]
p[1] = ['SIFTfdSize', 8]
p[2] = ['SIFTinitialSigma', 1.6]
p[3] = ['SIFTmaxOctaveSize', 1024]
p[4] = ['SIFTminOctaveSize', 64]
p[5] = ['SIFTsteps', 1]
p[6] = ['clearCache', True]
p[7] = ['maxNumThreadsSift', 56]
p[8] = ['rod', 0.92]
p[11] = ['identityTolerance', 1]
p[13] = ['maxEpsilon', 30]
p[15] = ['maxNumFailures', 3]
p[17] = ['maxNumThreads', 56] # should be changed for workstations (or automatically infered ?)
p[19] = ['minInlierRatio', 0]
p[20] = ['minNumInliers', 12]
p[21] = ['multipleHypotheses', False]
p[22] = ['widestSetOnly', False]
p[23] = ['rejectIdentity', False]
p[24] = ['visualize', False]

p[26] = ['dampSpringMesh', 0.9] #/!\ attention, value copied from internet, not tested

p[43] = ['useTps', False]

params = ElasticLayerAlignment.Param( *[ a[1] for a in [p[i] for i in range(len(p))] ] )

IJ.log('4. Opening the real scale EM project for elastic alignment')
project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)

IJ.log('4. Starting alignment')
IJ.log('The layer range is: ' + str(currentLayer) + '-' +  str(min(currentLayer + nLayersAtATime - 1, nLayers -1)) )
IJ.log('The fixed layers are: ' + str(0) + '-' + str(currentLayer - layerOverlap + 1))

layerRange = layerset.getLayers(currentLayer, min(currentLayer + nLayersAtATime - 1, nLayers -1)) # nLayers -1 because starts at 0
fixedLayers = HashSet(layerset.getLayers(0, currentLayer - layerOverlap + 1))
emptyLayers = HashSet()
propagateTransformBefore = False
propagateTransformAfter = False
thefilter = None

ElasticLayerAlignment().exec(params, project, layerRange, fixedLayers, emptyLayers, layerset.get2DBounds(), propagateTransformBefore, propagateTransformAfter, thefilter)
# ElasticLayerAlignment().exec(paramElastic, project, layerRange, fixedLayers, emptyLayers, layerset.get2DBounds(), propagateTransformBefore, propagateTransformAfter, thefilter)
IJ.log('Elastic layer alignment is done')
fc.resizeDisplay(layerset)
time.sleep(2)
project.save()
time.sleep(2)

fc.closeProject(project)

fc.shouldRunAgain(namePlugin, currentLayer, nLayers, MagCFolder, '', increment = (nLayersAtATime - layerOverlap))
