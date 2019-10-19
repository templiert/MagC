from __future__ import with_statement
# MLSTWithoutAffine
    # wrong, unscaled, contractViolated bug
# StandardMLST
    # now completely fails on the first 20 slices ... a bit surprising ...
# MLSTAffine
    # ok
# AffineFromMLSTXMLRead
    # is simply identity

# how to exclude non-matching registration ?
    # size of the transformed LM
        # would sometimes fail 
    # compute correlation of the cropped registered pair
        # does not work
        # I also do not know trivially where to crop
    # rerun a SIFT in parallel and get the displacement
    # check the amount of shear in the computed transform

    
from mpicbg.imglib.algorithm.correlation import CrossCorrelation
from mpicbg.imglib.image import ImagePlusAdapter    

from ij import IJ, Macro, ImagePlus
from ij.process import ByteProcessor
import os, shutil, time, pickle
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

from distutils.dir_util import copy_tree

namePlugin = 'export_TransformedCroppedLM'
MagCFolder = fc.startPlugin(namePlugin)
# MagCFolder = r'E:\Users\Thomas\MixturesClean\MinimalPipelineTest5Sections_OneMoreTest04_07_17'
ControlWindow.setGUIEnabled(False)

width, height, nChannels, xGrid, yGrid, scaleX, scaleY, channels = fc.readSessionMetadata(MagCFolder)
# channels = ['Brightfield', 'GFP', 'DsRed', 'contrastedBrightfield']

LMFolder = os.path.join(MagCFolder, 'MagC_LM')

projectPath = fc.findFilesFromTags(MagCFolder,['EM', 'Project'])[0] # should I make 2 projects ? One for rigid, one for warped ?

exportedEMFolder = fc.findFoldersFromTags(MagCFolder, ['export_alignedEMForRegistration'])[0]
nLayers = len(os.listdir(exportedEMFolder))

registrationFolder = os.path.join(os.path.dirname(projectPath), 'LMEMRegistration')

f = open(os.path.join(registrationFolder, 'lowResEMBounds'), 'r')
widthEM, heightEM = pickle.load(f)
f.close()

pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(registrationFolder, 1))
layersetZ.setDimensions(0, 0, widthEM * 5, heightEM * 5)
layerZ = layersetZ.getLayers().get(0)

# create the folders
for channel in channels:
    affineCroppedFolder = fc.mkdir_p(os.path.join(LMFolder, 'affineCropped_' + channel))

for l in range(nLayers):
    layerFolder = os.path.join(registrationFolder, 'layer_' + str(l).zfill(4))
    registeredFolder = os.path.join(layerFolder, 'registered')
    affTransformPath = os.path.join(registeredFolder, 'affineSerialized')
    if os.path.isfile(affTransformPath):
        affTransform = loaderZ.deserialize(affTransformPath)
        
        for channel in channels:
            affineCroppedFolder = os.path.join(LMFolder, 'affineCropped_' + channel)

            LMMosaicsPath = fc.cleanLinuxPath(
                os.path.join(
                    LMFolder,
                    'exported_downscaled_1_' + channel,
                    'exported_downscaled_1_' + channel 
                    + '_' + str(l).zfill(4) + '.tif'))
            
            patch = Patch.createPatch(pZ, LMMosaicsPath)
            layerZ.add(patch)
            IJ.log('Setting the affineTransform ' + str(affTransform))
            patch.setAffineTransform(affTransform)
            patch.updateBucket()

            bb = Rectangle(0, 0, widthEM, heightEM)
            affineCroppedIm = loaderZ.getFlatImage(
                layerZ,
                bb,
                1,
                0x7fffffff,
                ImagePlus.GRAY8,
                Patch,
                layerZ.getAll(Patch),
                True,
                Color.black,
                None)
            affineCroppedIm = fc.normLocalContrast(
                affineCroppedIm,
                50,
                50,
                3,
                True,
                True)
            IJ.run(affineCroppedIm, 'Median...', 'radius=2')
            affineCroppedImPath = os.path.join(
                affineCroppedFolder,
                'affineCropped_' + channel 
                + '_' + str(l).zfill(4) + '.tif')
            IJ.save(affineCroppedIm, affineCroppedImPath)
            affineCroppedIm.close()

            layerZ.remove(patch)
            layerZ.recreateBuckets()
            IJ.log('Has been written: ' + str(affineCroppedImPath))
fc.closeProject(pZ) # close dummy trakem

# # # # # # create the median folders
# # # # # for channel in channels[:-2]: # the fluorescent channels, excluding the brightfield and contrastedBrightfield channels 
    # # # # # affineCroppedFolder = os.path.join(LMFolder, 'affineCropped_' + channel)
    # # # # # finalLMFolder = fc.mkdir_p(os.path.join(LMFolder, 'finalLM_' + channel))
    # # # # # imPaths = [os.path.join(affineCroppedFolder, imName) for imName in fc.naturalSort(os.listdir(affineCroppedFolder))]
    # # # # # imStack = fc.stackFromPaths(imPaths)
    # # # # # IJ.run(imStack, 'Median 3D...', 'x=2 y=2 z=2')
    # # # # # stack = imStack.getImageStack()
    
    # # # # # for imId, imPath in enumerate(imPaths):
        # # # # # layerId = int(os.path.splitext((os.path.basename(imPath)))[0].split('_')[-1])
        
        # # # # # tileIndex = imStack.getStackIndex(0, 0, imId + 1) # to access the slice in the stack
        # # # # # finalIm = ImagePlus('finalLM_' + channel + '_' + str(layerId).zfill(4), stack.getProcessor(tileIndex).convertToByteProcessor())
        # # # # # finalImPath = os.path.join(finalLMFolder, 'finalLM_' + channel + '_' + str(layerId).zfill(4) + '.tif')
        # # # # # IJ.save(finalIm, finalImPath)

# # # # # # copy the brightfield and contrasted brightfield channels
# # # # # for channel in channels[-2:]: 
    # # # # # affineCroppedFolder = os.path.join(LMFolder, 'affineCropped_' + channel)
    # # # # # finalLMFolder = fc.mkdir_p(os.path.join(LMFolder, 'finalLM_' + channel))
    # # # # # copy_tree(affineCroppedFolder, finalLMFolder)
    # # # # # for imName in os.listdir(finalLMFolder):
        # # # # # imPath = os.path.join(finalLMFolder, imName)
        # # # # # newImName = imName.replace('affineCropped_', 'finalLM_')
        # # # # # newImPath = os.path.join(finalLMFolder, newImName)
        # # # # # os.rename(imPath, newImPath)

fc.terminatePlugin(namePlugin, MagCFolder)
