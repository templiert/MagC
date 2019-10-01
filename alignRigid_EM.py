#align rigid least square affine for a TrakEM project
from __future__ import with_statement
from ij import IJ, Macro, WindowManager
import os, time
import sys
sys.path.append(IJ.getDirectory('plugins'))
import fijiCommon as fc 
# from mpicbg.trakem2.align import RegularizedAffineLayerAlignment

from register_virtual_stack import Transform_Virtual_Stack_MT
from register_virtual_stack import Register_Virtual_Stack_MT

from java.awt.geom import AffineTransform
from java.awt import Rectangle
from java.util import HashSet, ArrayList
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch
from java.util.concurrent.atomic import AtomicInteger

namePlugin = 'alignRigid_EM'
MagCFolder = fc.startPlugin(namePlugin)
ControlWindow.setGUIEnabled(False)
MagC_EM_Folder = os.path.join(MagCFolder, 'MagC_EM','')
MagCParameters = fc.readMagCParameters(MagCFolder)


inputFolder = fc.findFoldersFromTags(MagCFolder, ['export_stitchedEMForAlignment'])[0]
imagePaths = filter(lambda x: os.path.splitext(x)[1] == '.tif', fc.naturalSort([os.path.join(inputFolder, x) for x in os.listdir(inputFolder)]))

regParams = Register_Virtual_Stack_MT.Param()
regParams.minInlierRatio = 0
regParams.registrationModelIndex = 3 # 1-Rigid, 2-Similarity, 3-Affine
regParams.featuresModelIndex = 3

exportForRigidAlignmentFolder = inputFolder
resultRigidAlignmentFolder = fc.mkdir_p(os.path.join(MagC_EM_Folder, 'resultRigidAlignment'))

################################################
# rigid alignment outside trakem2 with register virtual stack plugin because bug in trakem2
transformsPath = os.path.join(MagC_EM_Folder, 'rigidAlignmentTransforms_' + namePlugin + '.txt')
referenceName = fc.naturalSort(os.listdir(exportForRigidAlignmentFolder))[0]
use_shrinking_constraint = 0
IJ.log('Rigid alignment with register virtual stack')
Register_Virtual_Stack_MT.exec(exportForRigidAlignmentFolder, resultRigidAlignmentFolder, resultRigidAlignmentFolder, referenceName, regParams, use_shrinking_constraint)
time.sleep(2)
# IJ.getImage().close()
WindowManager.closeAllWindows()
IJ.log('Rigid Alignment done')
################################################

###########################################
IJ.log('Aligning the lowEM with the new rigid transforms')
projectPath = fc.cleanLinuxPath(fc.findFilesFromTags(MagCFolder,['EMProject_'])[0]) # this is the low res EM
project, loader, layerset, nLayers = fc.openTrakemProject(projectPath) # the low res EM
for l, layer in enumerate(layerset.getLayers()):
	transformPath = os.path.join(resultRigidAlignmentFolder, 'stitchedDownsampledEM_' + str(l).zfill(4) + '.xml')
	theList = ArrayList(HashSet())
	aff = fc.getAffFromRVSTransformPath(transformPath)

	for patch in layer.getDisplayables(Patch):
		patch.setLocation(patch.getX(), patch.getY()) # compensate for the extracted bounding box
		currentAff = patch.getAffineTransform()
		currentAff.preConcatenate(aff)
		patch.setAffineTransform(currentAff)

fc.resizeDisplay(layerset)
project.save()
fc.closeProject(project)
IJ.log('All LM layers have been aligned in: ' + projectPath)

time.sleep(1)

# High EM
################################################
IJ.log('Aligning the highEM with the new rigid transforms')
downsamplingFactor = MagCParameters['downsample_EM']['downsamplingFactor']
IJ.log('The LM/EM pixel scale factor is ' + str(downsamplingFactor))

projectPath = fc.cleanLinuxPath(os.path.join(MagCFolder, 'MagC_EM', 'EMProject.xml')) # the high res EM project

afterProjectPath = fc.cleanLinuxPath(projectPath.replace('EMProject.', 'ToBeElasticAlignedEMProject.'))
IJ.log('afterProjectPath ' + str(afterProjectPath))

project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)

for l, layer in enumerate(layerset.getLayers()):
	transformPath = os.path.join(resultRigidAlignmentFolder, 'stitchedDownsampledEM_' + str(l).zfill(4) + '.xml')
	aff = fc.getAffFromRVSTransformPath(transformPath)
	# aff.scale(float(downsamplingFactor), float(downsamplingFactor)) # cannot be used because it is not a simple scaling
	aff = AffineTransform(aff.getScaleX(), aff.getShearY(), aff.getShearX(), aff.getScaleY(), aff.getTranslateX() * float(downsamplingFactor), aff.getTranslateY() * float(downsamplingFactor))

	for patch in layer.getDisplayables(Patch):
		currentAff = patch.getAffineTransform()
		currentAff.preConcatenate(aff)
		patch.setAffineTransform(currentAff)
IJ.log('The real EM project is now rigidly aligned. Saving and closing the project.')
fc.resizeDisplay(layerset)
project.save()
project.saveAs(afterProjectPath, True)
fc.closeProject(project)
time.sleep(2)

fc.terminatePlugin(namePlugin, MagCFolder)
