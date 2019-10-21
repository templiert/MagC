from __future__ import with_statement
    
from mpicbg.imglib.algorithm.correlation import CrossCorrelation
from mpicbg.imglib.image import ImagePlusAdapter    

from ij import IJ, Macro, ImagePlus
from ij.process import ByteProcessor
import os, shutil, time
from java.awt import Rectangle, Color
from java.util import HashSet, ArrayList

from java.awt.geom import AffineTransform
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch

import sys
sys.path.append(IJ.getDirectory('plugins'))
import fijiCommon as fc 

from register_virtual_stack import Transform_Virtual_Stack_MT

from ini.trakem2.io import CoordinateTransformXML

def readIds(p):
    with open(p, 'r') as f:
        lines = f.readlines()
    return [
        int(l.replace('\n', ''))
        for l in lines
        ]


namePlugin = 'assembly_LMProjects'
MagCFolder = fc.startPlugin(namePlugin)
# MagCFolder = r'E:\Users\Thomas\MixturesClean\MinimalPipelineTest5Sections_OneMoreTest04_07_17'
ControlWindow.setGUIEnabled(False)

EMFolder = os.path.join(MagCFolder, 'MagC_EM','')
LMFolder = os.path.join(MagCFolder, 'MagC_LM', '')

width, height, nChannels, xGrid, yGrid, scaleX, scaleY, channels = fc.readSessionMetadata(MagCFolder)
# channels = ['Brightfield', 'GFP', 'DsRed', 'contrastedBrightfield']

projectPath = fc.findFilesFromTags(MagCFolder,['EM', 'Project'])[0] # should I make 2 projects ? One for rigid, one for warped ?

exportedEMFolder = fc.findFoldersFromTags(MagCFolder, ['export_alignedEMForRegistration'])[0]
nLayers = len(os.listdir(exportedEMFolder))

registrationFolder = os.path.join(os.path.dirname(projectPath), 'LMEMRegistration')

BIB = False
ordered_surface_ids_path = [os.path.join(MagCFolder, n)
    for n in os.listdir(MagCFolder)
    if 'ordered_surface_ids' in n]
if len(ordered_surface_ids_path) != 0:
    ordered_surface_ids_path = ordered_surface_ids_path[0]
    ordered_surface_ids = readIds(ordered_surface_ids_path)
    BIB = True

for idChannel, channel in enumerate(channels):
    affineCroppedFolder = os.path.join(LMFolder, 'affineCropped_' + channel)
    
    # the dimensions of the first affineCropped determine the size of the layerset of the trakem project (and for the export)
    firstImagePath = os.path.join(affineCroppedFolder, os.walk(affineCroppedFolder).next()[2][0])
    im0 = IJ.openImage(firstImagePath)
    width0 = im0.getWidth()
    height0 = im0.getHeight()
    im0.close()
    
    roiExport = Rectangle(0, 0, width0, height0)
    
    projectPath = os.path.join(EMFolder, 'LMProject_' + channel + '.xml')
    p, loader, layerset, nLayers = fc.getProjectUtils(fc.initTrakem(LMFolder, nLayers))
    p.saveAs(projectPath, True)
    layerset.setDimensions(0, 0, width0, height0)
    
    # for l, layer in enumerate(layerset.getLayers()):
    for l, z in enumerate(ordered_surface_ids):
        layer = layerset.getLayers().get(z)
    
        layerFolder = os.path.join(registrationFolder, 'layer_' + str(l).zfill(4))
        registeredFolder = os.path.join(layerFolder, 'registered')
        MLSTPath = os.path.join(registeredFolder, 'MLST.xml')
        if os.path.isfile(MLSTPath):
            MLSTransform = CoordinateTransformXML().parse(MLSTPath)
            affineCroppedImPath = os.path.join(affineCroppedFolder, 'affineCropped_' + channel + '_' + str(l).zfill(4) + '.tif')            
            patch = Patch.createPatch(p, affineCroppedImPath)
            layer.add(patch)
            patch.setCoordinateTransform(MLSTransform) # does the order matter ? apparently yes, but I have to be sure that it is not an offset problem
            IJ.log('Setting the mlsTransform in layer ' + str(l) + ' ' + str(MLSTransform))
            patch.updateBucket()
            
            IJ.log('idChannel ************* ' + str(idChannel))
            if idChannel < len(channels)-2: # if it is a fluochannel
                MLSTransformedFolder = fc.mkdir_p(os.path.join(LMFolder, 'MLS_Transformed_' + str(channel), ''))
                imp = loader.getFlatImage(layer, roiExport, 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layer.getAll(Patch), True, Color.black, None)
                impPath = os.path.join(MLSTransformedFolder, 'MLSTransformed_' + channel + '_' + str(l).zfill(4) + '.tif')
                IJ.save(imp, impPath)
            
            
    IJ.log('Project ' + channel + ' assembled')
    
    # # Warning ! Should I resize or not ? does this not create an offset ?
    # fc.resizeDisplay(layerset)
    
    p.save()
    fc.closeProject(p)
    
IJ.log('Done')
fc.terminatePlugin(namePlugin, MagCFolder)
