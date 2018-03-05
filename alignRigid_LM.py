from ij import IJ
from ij import Macro
import os
import time
import fijiCommon as fc
from mpicbg.trakem2.align import RegularizedAffineLayerAlignment
from java.awt.geom import AffineTransform
from java.awt import Rectangle
from java.util import HashSet
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch

from register_virtual_stack import Register_Virtual_Stack_MT

namePlugin = 'alignRigid_LM'
MagCFolder = fc.startPlugin(namePlugin)
ControlWindow.setGUIEnabled(False)

# get mosaic size
MagCParameters = fc.readMagCParameters(MagCFolder)
executeAlignment = MagCParameters[namePlugin]['executeAlignment']
boxFactor = MagCParameters[namePlugin]['boxFactor'] # e.g. 0.5, use only the center part of the layer to compute alignment: 0.5 divides the x and y dimensions by 2

projectPath = fc.findFilesFromTags(MagCFolder,['LMProject'])[0]

# alignment parameters
regParams = Register_Virtual_Stack_MT.Param()
regParams.minInlierRatio = 0
regParams.registrationModelIndex = 1
regParams.featuresModelIndex = 1

regParams.sift.fdBins = 8
regParams.sift.fdSize = 4
regParams.sift.initialSigma = 1.6
regParams.sift.maxOctaveSize = 1024
regParams.sift.minOctaveSize = 64
regParams.sift.steps = 6

regParams.interpolate = True
regParams.maxEpsilon = 25
regParams.minInlierRatio = 0
regParams.rod = 0.92

# perform alignment
if executeAlignment:
	fc.rigidAlignment(projectPath, regParams, name = namePlugin, boxFactor = boxFactor)

# open the project and save all transforms of all tiles in all sections
IJ.log('Saving the coordinates transforms of each patch of each layer')
project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)
transformsPath = os.path.join(os.path.dirname(projectPath), namePlugin + 'transforms.txt')
fc.writeAllAffineTransforms(project,transformsPath)
fc.closeProject(project)

fc.terminatePlugin(namePlugin, MagCFolder)
