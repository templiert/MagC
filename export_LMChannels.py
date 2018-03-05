from __future__ import with_statement
from ij import IJ
from ij import Macro
import os
import time
import fijiCommon as fc
from mpicbg.trakem2.align import RegularizedAffineLayerAlignment
from java.awt.geom import AffineTransform 
from java.awt import Rectangle
from java.util import HashSet
from ini.trakem2.display import Patch
from ini.trakem2.imaging import Blending
from ini.trakem2 import Project, ControlWindow

namePlugin = 'export_LMChannels'
MagCFolder = fc.startPlugin(namePlugin)
ControlWindow.setGUIEnabled(False)

transformsPath = fc.findFilesFromTags(MagCFolder, ['LM', 'transforms'])[0]
with (open(transformsPath, 'r')) as f:
	transforms = f.readlines()
nLayers = max([int(transforms[i]) for i in range(1, len(transforms),8) ]) + 1
IJ.log('nLayers = ' + str(nLayers))

width, height, nChannels, xGrid, yGrid, scaleX, scaleY, channels = fc.readSessionMetadata(MagCFolder)

MagCParameters = fc.readMagCParameters(MagCFolder)
scaleFactors = MagCParameters[namePlugin]['scaleFactors'] # scale factor for export, typically 1 and 0.1

IJ.log('Iterating over the LM channels')
for channel in channels:
	IJ.log('Processing channel ' + str(channel))
	IJ.log('Creating a TrakEM project')
	
	# create trakem project for the channel
	trakemFolder = os.path.join(os.path.dirname(transformsPath), '')
	project = fc.initTrakem(trakemFolder,nLayers)
	loader = project.getLoader()
	loader.setMipMapsRegeneration(False) # disable mipmaps
	layerset = project.getRootLayerSet()
	
	# insert the tiles according to the transforms computed on the reference brightfield channel
	IJ.log('Inserting all patches')
	for i in range(0, len(transforms), 8):
		alignedPatchPath = transforms[i]
		l = int(transforms[i+1])
		alignedPatchName = os.path.basename(alignedPatchPath)

		toAlignPatchPath = fc.cleanLinuxPath(os.path.join(os.path.dirname(alignedPatchPath), alignedPatchName.replace(channels[-1], channel)))
		toAlignPatchPath = toAlignPatchPath[:-1]  # remove a mysterious trailing character ...
		IJ.log('In channel ' + str(channel) + ', inserting this image: ' + str(toAlignPatchPath))
		aff = AffineTransform([float(transforms[a]) for a in range(i+2, i+8)])
		patch = Patch.createPatch(project, toAlignPatchPath)
		layer = layerset.getLayers().get(l)
		layer.add(patch)		
		patch.setAffineTransform(aff)
		patch.updateBucket()
	
	time.sleep(1)
	IJ.log('Readjusting display')	
	fc.resizeDisplay(layerset)
	IJ.log('Blending all layers')	
	Blending.blendLayerWise(layerset.getLayers(), True, None)
	
	IJ.log('Exporting')
	for scaleFactor in scaleFactors:
		theBaseName = 'exported_downscaled_' + str(int(1/float(scaleFactor))) + '_' + channel
		outputFolder = fc.mkdir_p( os.path.join(os.path.dirname(transformsPath), theBaseName))
		fc.exportFlat(project,outputFolder,scaleFactor, baseName = theBaseName, bitDepth = 8)
		
	fc.closeProject(project)

fc.terminatePlugin(namePlugin, MagCFolder)