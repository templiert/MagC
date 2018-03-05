#this script puts the acquired EM tiles into Trakem at the right positions
from __future__ import with_statement
import os, re, errno, string, shutil, time
import xml.etree.ElementTree as ET
from os import path  

from java.awt.event import MouseAdapter, KeyEvent, KeyAdapter
from jarray import zeros, array
from java.util import HashSet, ArrayList
from java.awt.geom import AffineTransform
from java.awt import Color

from ij import IJ, Macro
from fiji.tool import AbstractTool
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch
from ini.trakem2.utils import Utils
from ini.trakem2.display import Display, Patch
from mpicbg.trakem2.align import Align, AlignTask

import fijiCommon as fc

namePlugin = 'init_EM'
MagCFolder = fc.startPlugin(namePlugin)
ControlWindow.setGUIEnabled(False)

EMDataFolder = os.path.join(MagCFolder, 'EMData')
MagCEMFolder = fc.makeNeighborFolder(EMDataFolder, 'MagC_EM')

imageFolders = [os.path.join(EMDataFolder, sectionFolderName) for sectionFolderName in fc.naturalSort(os.walk(EMDataFolder).next()[1])]

nSections = len(imageFolders)
IJ.log('There are ' + str(nSections) + ' EM layers')

# reading EM metadata
EMMetadataPath = os.path.join(MagCEMFolder, 'EM_Metadata.txt')
try: # old Atlas format
	mosaicMetadata = os.path.join(imageFolders[0] , filter(lambda x: 'Mosaic' in x, os.listdir(imageFolders[0]))[0])
	root = ET.parse(mosaicMetadata).getroot()

	pixelSize = float(root.find('PixelSize').text)
	tileinfo = root.find('TileInfo')
	tileWidth = int(tileinfo.find('TileWidth').text)
	tileHeight = int(tileinfo.find('TileHeight').text)
	tileOverlapX = float(tileinfo.find('TileOverlapXum').text)
	tileOverlapY = float(tileinfo.find('TileOverlapYum').text)
	numTilesX = int(tileinfo.find('NumTilesX').text)
	numTilesY = int(tileinfo.find('NumTilesY').text)
	xPatchEffectiveSize = tileWidth - float(tileOverlapX * 1000 / float(pixelSize))
	yPatchEffectiveSize = tileHeight - float(tileOverlapX * 1000 / float(pixelSize))

	# writing EM metadata
	IJ.log('Writing the EM Metadata file')
	parameterNames = ['pixelSize', 'tileWidth', 'tileHeight', 'tileOverlapX', 'tileOverlapY', 'numTilesX', 'numTilesY', 'xPatchEffectiveSize', 'yPatchEffectiveSize', 'nSections']
	with open(EMMetadataPath, 'w') as f:
		f.write('# EM Metadata' + '\n')
		for parameterName in parameterNames:
			IJ.log('parameterName = ' + str(parameterName))
			parameterEntry = parameterName + ' = ' + str(vars()[parameterName]) + '\n'
			IJ.log(parameterEntry)
			f.write(parameterEntry)
except Exception, e: # standard metadata format from the EM_Imaging.py script
	shutil.copyfile(os.path.join(EMDataFolder, 'EM_Metadata.txt'), EMMetadataPath)
			
fc.terminatePlugin(namePlugin, MagCFolder)