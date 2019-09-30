from distutils.dir_util import copy_tree		
import os, shutil, sys

import xml.etree.ElementTree as ET
from xml.dom import minidom

import ij
from ij import IJ

sys.path.append(IJ.getDirectory('plugins'))
import fijiCommon as fc 

namePlugin = 'preprocess_ForPipeline'
MagCFolder = fc.startPlugin(namePlugin)

try: # look in priority for sectionOrder, which means that it has already been processed
	orderPath = os.path.join(MagCFolder, filter(lambda x: 'sectionOrder' in x, os.listdir(MagCFolder))[0])
except Exception, e:
	orderPath = os.path.join(MagCFolder, filter(lambda x: 'solution' in x, os.listdir(MagCFolder))[0])

sectionOrder = fc.readOrder(orderPath)
IJ.log('sectionOrder: ' + str(sectionOrder))

MagCParams = fc.readMagCParameters(MagCFolder)

executeLM = MagCParams[namePlugin]['executeLM']
executeEM = MagCParams[namePlugin]['executeEM']

##########################
##########  LM  ##########
##########################
if executeLM:
	sourceLMDataFolder = os.path.join(MagCFolder, 'LMData')
	targetLMDataFolder = os.path.join(MagCFolder, 'LMDataReordered')

	try:
		os.makedirs(targetLMDataFolder)
	except Exception, e:
		print 'Folder not created', targetLMDataFolder
		pass

	# copy the metadata into working folder. Rename the original file so that it is not used later (only the file in the working folder will be found if correct naming is used)
	for fileName in os.listdir(sourceLMDataFolder):
		if ('ata.txt' in fileName) and ('LM' in fileName):
			sourceLMMetadataPath = os.path.join(sourceLMDataFolder, fileName)
			shutil.copyfile(sourceLMMetadataPath, os.path.join(targetLMDataFolder, 'LM_Metadata.txt'))
			os.rename(sourceLMMetadataPath, os.path.join(sourceLMDataFolder, 'LM_Meta_Data.txt')) # the second '_' is important so that this file is not used later
	
	# copy in the correct oder the LMData		
	for sourceId, targetId in enumerate(sectionOrder):
		IJ.log('LM prepipeline: processing section ' + str(sourceId))
		sourceFolder = os.path.join(sourceLMDataFolder, 'section_' + str(targetId).zfill(4))
		targetFolder = os.path.join(targetLMDataFolder, 'section_' + str(sourceId).zfill(4))
		try:
			os.makedirs(targetFolder)
		except Exception, e:
			print 'Folder not created', targetFolder
			pass
		
		for sourceImageName in os.listdir(sourceFolder):
			targetImageName = sourceImageName.replace('section_' + str(targetId).zfill(4) , 'section_' + str(sourceId).zfill(4))
			sourceImagePath = os.path.join(sourceFolder, sourceImageName)
			targetImagePath = os.path.join(targetFolder, targetImageName)
			IJ.log('Copying \n' + sourceImagePath + '\nto\n' + targetImagePath)
			shutil.copyfile(sourceImagePath, targetImagePath)
	
##############################
##########  EM  ##########
##############################
if executeEM:
	sourceEMDataFolder = os.path.join(MagCFolder, 'EMDataRaw')
	targetEMDataFolder = os.path.join(MagCFolder, 'EMData')
	try:
		os.makedirs(targetEMDataFolder)
	except Exception, e:
		print 'Folder not created', targetEMDataFolder
		pass

	# Copy image files
	for sourceId, targetId in enumerate(sectionOrder):
		IJ.log('EM prepipeline: processing section ' + str(sourceId))
		# sourceSliceFolder = os.path.join(sourceEMDataFolder, 'section_' + str(targetId).zfill(4)) # with reordering
		sourceSliceFolder = os.path.join(sourceEMDataFolder, 'section_' + str(sourceId).zfill(4)) # without reordering
		targetSliceFolder = os.path.join(targetEMDataFolder, 'section_' + str(sourceId).zfill(4))
		try:
			os.makedirs(targetSliceFolder)
		except Exception, e:
			pass
		for fileName in os.listdir(sourceSliceFolder):
			shutil.copy(os.path.join(sourceSliceFolder, fileName), targetSliceFolder)
		# copy_tree(sourceSliceFolder, targetSliceFolder) # copytree is very slow ...

	# Copy metadata
	metadataName = filter(lambda x: 'EM_Metadata' in x, os.listdir(sourceEMDataFolder))[0]
	print 'metadataName', metadataName
	sourceMetaData = os.path.join(sourceEMDataFolder, metadataName)
	targetMetaData = os.path.join(targetEMDataFolder, 'EM_Metadata.txt')		
	print 'sourceMetaData', sourceMetaData
	print 'targetMetaData', targetMetaData
	shutil.copyfile(sourceMetaData, targetMetaData)
	
fc.terminatePlugin(namePlugin, MagCFolder)	