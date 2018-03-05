from ij import IJ
from ij import Macro
import os
import time
import fijiCommon as fc
from mpicbg.trakem2.align import Align, AlignTask
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch
from ini.trakem2.imaging import StitchingTEM
from ini.trakem2.imaging.StitchingTEM import PhaseCorrelationParam
from java.util.concurrent.atomic import AtomicInteger
from mpicbg.trakem2.align import Align, AlignTask
from ini.trakem2.imaging import Blending

params = Align.ParamOptimize().clone()
params.correspondenceWeight = 1
params.desiredModelIndex = 0
params.expectedModelIndex = 0
params.maxEpsilon = 1
params.minInlierRatio = 0.05
params.minNumInliers = 7
params.regularize = False
params.regularizerModelIndex = 0
params.lambda = 0.01
params.maxIterations = 2000
params.maxPlateauwidth = 200
params.meanFactor = 3
params.filterOutliers = False
params.sift.fdBins = 8
params.sift.fdSize = 8
params.sift.initialSigma = 1.6
params.sift.maxOctaveSize = 1024
params.sift.minOctaveSize = 60
params.sift.steps = 6
tilesAreInPlaceIn = True
largestGraphOnlyIn = False
hideDisconnectedTilesIn = False
deleteDisconnectedTilesIn = False

def stitchLayers():
	while atomicI.get() < nLayers:
		l = atomicI.getAndIncrement()
		if l < nLayers:
			IJ.log('Stitching layer ' + str(l))
			AlignTask().montageLayers(params, layerset.getLayers(l, l), tilesAreInPlaceIn, largestGraphOnlyIn, hideDisconnectedTilesIn, deleteDisconnectedTilesIn)
			IJ.log('Blending layer ' + str(l))
			Blending.blendLayerWise(layerset.getLayers(l, l), True, None)
			if l%10 == 0: # save project every 5 layers
				project.save()

namePlugin = 'montage_LM'
MagCFolder = fc.startPlugin(namePlugin)
ControlWindow.setGUIEnabled(False)

# get mosaic size
MagCParameters = fc.readMagCParameters(MagCFolder)
mosaic = MagCParameters[namePlugin]['mosaic'] # e.g. [2,2]

if mosaic !=[1,1]:
	projectPath = fc.findFilesFromTags(MagCFolder,['LMProject'])[0]
	projectName = os.path.basename(projectPath)

	project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)

	nLayers = len(layerset.getLayers())

	atomicI = AtomicInteger(0)			
	fc.startThreads(stitchLayers, fractionCores = 0.1)
	fc.resizeDisplay(layerset)
	project.save()
	fc.closeProject(project)
fc.terminatePlugin(namePlugin, MagCFolder)