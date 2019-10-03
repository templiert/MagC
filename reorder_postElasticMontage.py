# section reorder after elastic montage
from __future__ import with_statement
from ij import IJ
import os, time, shutil
import sys
sys.path.append(IJ.getDirectory('plugins'))
import fijiCommon as fc 
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch
from distutils.dir_util import copy_tree

namePlugin = 'reorder_postElasticMontage'
MagCFolder = fc.startPlugin(namePlugin)
ControlWindow.setGUIEnabled(False)

MagCParameters = fc.readMagCParameters(MagCFolder)
# read the order. If there is no order file, use default order 0,1,2,...
try: # look in priority for sectionOrder, which means that the solution from concorde has already been processed
	orderPath = os.path.join(MagCFolder, filter(lambda x: 'sectionOrder' in x, os.listdir(MagCFolder))[0])
except Exception, e:
	try:
		orderPath = os.path.join(MagCFolder, filter(lambda x: 'solution' in x, os.listdir(MagCFolder))[0])
	except Exception, e:
		orderPath = ''
	
if os.path.isfile(orderPath):
	newOrder = fc.readOrder(orderPath)
else:
	newOrder = range(10000)
	
# Reorder both the low EM and the EM projects
## low EM

# downsamplingFactor
downsamplingFactor = MagCParameters['downsample_EM']['downsamplingFactor']
factorString = str(int(1000000*downsamplingFactor)).zfill(8)

MagCEMFolder = os.path.dirname(fc.cleanLinuxPath(fc.findFilesFromTags(MagCFolder,['montage_ElasticEM_Transforms.txt'])[0]))

projectPath = fc.cleanLinuxPath(os.path.join(MagCEMFolder, 'EMProject_' + factorString + '.xml')) # this is the low res EM project
unorderedProjectPath = fc.cleanLinuxPath(os.path.join(MagCEMFolder, 'LowEMProjectUnordered.xml'))

# if a reordering had already been made, reinitialize the unordered project
if os.path.isfile(unorderedProjectPath):
	if os.path.isfile(projectPath):
		os.remove(projectPath)
	shutil.copyfile(unorderedProjectPath, projectPath)
	project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)
	project.saveAs(projectPath, True)
	fc.closeProject(project)
	os.remove(unorderedProjectPath)

project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)
project.saveAs(unorderedProjectPath, True)
fc.closeProject(project)
os.remove(projectPath)

# reorder low-res project
fc.reorderProject(unorderedProjectPath, projectPath, newOrder)

## high EM
projectPath = fc.cleanLinuxPath(os.path.join(MagCFolder, 'MagC_EM', 'EMProject.xml')) # the high res EM project
unorderedProjectPath = os.path.join(os.path.dirname(projectPath), 'HighEMProjectUnordered.xml')

# if a reordering had already been made, reinitialize the unordered project
if os.path.isfile(unorderedProjectPath): 
	if os.path.isfile(projectPath):
		os.remove(projectPath)
	shutil.copyfile(unorderedProjectPath, projectPath)
	project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)
	project.saveAs(projectPath, True)
	fc.closeProject(project)
	os.remove(unorderedProjectPath)

project, loader, layerset, nLayers = fc.openTrakemProject(projectPath)
project.saveAs(unorderedProjectPath, True)
fc.closeProject(project)
os.remove(projectPath)

# reorder high-res project
fc.reorderProject(unorderedProjectPath, projectPath, newOrder)

# Reorder the exported files: a few checks to know whether a reordering had already taken place or not.
exportFolder = fc.findFoldersFromTags(MagCFolder, ['export_stitchedEMForAlignment'])[0]
unorderedExportFolder = os.path.join(os.path.dirname(os.path.normpath(exportFolder)), 'unorderedExport_stitchedEMForAlignment')
print 'exportFolder', exportFolder
print 'unorderedExportFolder', unorderedExportFolder

if os.path.exists(unorderedExportFolder):
	IJ.log('### Unordered folder exists')
	shutil.rmtree(exportFolder)
	time.sleep(3)
	copy_tree(unorderedExportFolder, exportFolder)
else:
	IJ.log('### Unordered folder does not exist: create it')
	copy_tree(exportFolder, unorderedExportFolder)
# at that stage the two folders are identical and unordered
for imName in os.listdir(exportFolder):
	imPath = os.path.join(exportFolder, imName)
	imIndex = int(imName.split('.')[0][-4:])
	newName = imName.replace(str(imIndex).zfill(4), str(newOrder.index(imIndex)).zfill(5))
	newPath = os.path.join(exportFolder, newName)
	os.rename(imPath, newPath)
for imName in os.listdir(exportFolder):
	imPath = os.path.join(exportFolder, imName)
	imIndex = int(imName.split('.')[0][-5:])
	newName = imName.replace(str(imIndex).zfill(5), str(imIndex).zfill(4))
	newPath = os.path.join(exportFolder, newName)
	os.rename(imPath, newPath)

fc.terminatePlugin(namePlugin, MagCFolder)
