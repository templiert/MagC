#This concatenates the render output to get all slices
#convention : [root directory]/[tile width]x[tile height]/[level]/[z]/[row]/[col].[format]

# Inspired from:
# Copyright (c) 2016, 2017, Forschungszentrum Juelich GmbH
# Author: Yann Leprince <y.leprince@fz-juelich.de>
#
# This software is made available under the MIT licence, see LICENCE.txt.

from functools import partial
from multiprocessing import Pool
from PIL import Image
import skimage.io
import argparse
import os
import sys
import copy
import json
import gzip
import pickle
import time
import numpy as np


#Get the list of immediate subdirectories
def getSubFoldersNames(folder):
	return sorted([name for name in os.listdir(folder)
			if os.path.isdir(os.path.join(folder, name))])

#Get all the row-col pairs
def getRowColPairs(folder):
	pairs = set()
	sliceFoldersNames = getSubFoldersNames(folder)
	for sliceFolderName in sliceFoldersNames:
		sliceFolder = os.path.join(folder, sliceFolderName)
		rowFolderNames = getSubFoldersNames(sliceFolder)
		for rowFolderName in rowFolderNames:
			row = int(rowFolderName)
			rowFolder = os.path.join(sliceFolder, rowFolderName)
			for mipmapName in os.listdir(rowFolder):
				col = int(os.path.splitext(mipmapName)[0])
				pairs.add((row, col))
	print('There are ', len(pairs), 'pairs')
	return pairs

def getMaxRowMaxCol(folder):
	maxRow, maxCol = 0, 0
	sliceFoldersNames = getSubFoldersNames(folder)
	for sliceFolderName in sliceFoldersNames:
		sliceFolder = os.path.join(folder, sliceFolderName)
		rowFolderNames = getSubFoldersNames(sliceFolder)
		maxRow = max([*map(int, rowFolderNames)] + [maxRow])
		for rowFolderName in rowFolderNames:
			rowFolder = os.path.join(sliceFolder, rowFolderName)
			maxCol = max([*map(lambda x: int(os.path.splitext(x)[0]), os.listdir(rowFolder))] + [maxCol])
	return maxRow, maxCol

def getMinRowsMinCols(folder): # to get the offset
	allMins = []
	resolutionNames = getSubFoldersNames(folder)
	#print('resolutionNames', resolutionNames)
	for resolutionName in resolutionNames:
		resolutionFolder = os.path.join(folder, resolutionName)
		minRow, minCol = 999999999, 999999999
		sliceFoldersNames = getSubFoldersNames(resolutionFolder)
		#print('sliceFoldersNames', sliceFoldersNames)
		for sliceFolderName in sliceFoldersNames:
			sliceFolder = os.path.join(resolutionFolder, sliceFolderName)
			#print('sliceFolder', sliceFolder)
			rowFolderNames = getSubFoldersNames(sliceFolder)
			minRow = min([*map(int, rowFolderNames)] + [minRow])
			for rowFolderName in rowFolderNames:
				rowFolder = os.path.join(sliceFolder, rowFolderName)
				minCol = min([*map(lambda x: int(os.path.splitext(x)[0]), os.listdir(rowFolder))] + [minCol])
		allMins.append([minRow, minCol])
	print('allMins', allMins)
	return allMins

#Convert slices to raw chunks
def writer(rowColPair, level, mipmapLevelFolder, sliceMax, precomputedFolder, mipmapSize, info, blackImage, visualizationMode, offset):
	if visualizationMode == 'local':
		RAW_CHUNK_PATTERN = '{key}/{0}-{1}/{2}-{3}/{4}-{5}.gz' # for local viewing with the HBP docker
	elif visualizationMode == 'online':
		RAW_CHUNK_PATTERN = '{key}/{0}-{1}_{2}-{3}_{4}-{5}' # for online viewing

	row, col = rowColPair

	mipmapLocationRow = row * mipmapSize
	mipmapLocationCol = col * mipmapSize

	chunkSize = info['scales'][level]['chunk_sizes'][0]
	size = info['scales'][level]['size']

	dataType = np.dtype(info['data_type']).newbyteorder('<')

	for sliceStart in range(0, sliceMax+1, chunkSize[2]): # or sliceMax + 1
	# for sliceStart in range(0, 3, chunkSize[2]): # or sliceMax + 1
		sliceEnd = min(sliceStart + chunkSize[2], size[2])

		# load z-stack of mipmaps
		mipmaps = []
		for sliceId in range(sliceStart, sliceEnd):
			mipmapPath = os.path.join(mipmapLevelFolder, str(sliceId), str(row), str(col) + '.png')
			if os.path.isfile(mipmapPath):
				mipmap = skimage.io.imread(mipmapPath)
			else:
				mipmap = blackImage
			# mipmap.T[-10:-5] = 255 # for debugging of the pixel shift
			# theMax = max(theMax, max(mipmap)) # to know whether the complete block is black, in which case I should probably write nothing. But is this case already handled by render ? Render does not render black mipmaps no ?
			mipmaps.append(mipmap)

		block = skimage.io.concatenate_images(mipmaps)

		if np.amax(block) > 0:

			# loop through all the chunks of this mipmap at this z chunk depth
			nChunkRow = mipmapSize//chunkSize[1]
			nChunkCol = mipmapSize//chunkSize[0]

			for chunkRow in range(nChunkRow):
				rowSlicing = np.s_[chunkRow * chunkSize[1] : (chunkRow + 1) * chunkSize[1]]
				chunkLocationRow = mipmapLocationRow + chunkSize[1] * chunkRow

				for chunkCol in range(nChunkCol):
					chunkLocationCol = mipmapLocationCol + chunkSize[0] * chunkCol

					colSlicing = np.s_[chunkCol * chunkSize[0] : (chunkCol + 1) * chunkSize[0]]

					chunk = block[np.s_[:], rowSlicing, colSlicing]
					if np.amax(chunk) > 0:
						x_coords = chunkLocationCol, chunkLocationCol + chunkSize[0]
						y_coords = chunkLocationRow, chunkLocationRow + chunkSize[1]
						z_coords = sliceStart, sliceEnd

						#chunk_name = RAW_CHUNK_PATTERN.format(x_coords[0]-offset[0], x_coords[1]-offset[0], y_coords[0]-offset[1], y_coords[1]-offset[1], z_coords[0], z_coords[1], key = info['scales'][level]['key'])
						chunk_name = RAW_CHUNK_PATTERN.format(x_coords[0], x_coords[1], y_coords[0], y_coords[1], z_coords[0], z_coords[1], key = info['scales'][level]['key'])
						chunkPath = os.path.join(precomputedFolder, chunk_name)

						os.makedirs(os.path.dirname(chunkPath), exist_ok=True)
						if not os.path.isfile(chunkPath):
							chunk = np.asfortranarray(chunk) # could be done like this also

							# for id, ch in enumerate(chunk):
							# 	if id<3:
							# 		skimage.io.imshow(ch)
							# 		skimage.io.show()
							# 8/0

							chunkByte = chunk.astype(dataType).tobytes()
							# 8/0

							if visualizationMode == 'local':
								with gzip.open(chunkPath, 'wb') as f:
									f.write(chunkByte)
							elif visualizationMode == 'online':
								with open(chunkPath, 'wb') as f:
									f.write(chunkByte)

#Main function
def mipmapToPrecomputed(mipmapFolder, precomputedFolder, mipmapSize, infoPath, nThreads, threadId, visualizationMode):
	with open(infoPath) as f:
		info = json.load(f)
	blackImage = np.zeros((mipmapSize, mipmapSize)).astype('uint8')
	# p = Pool(processes = 1)

	allMins = getMinRowsMinCols(mipmapFolder)
	
	for idLevel, level in enumerate(getSubFoldersNames(mipmapFolder)): # loop through resolution levels
		if idLevel > -1: # for debug
			mipmapLevelFolder = os.path.join(mipmapFolder, level)
			level = int(level)

			offset = [allMins[idLevel][1] * mipmapSize, allMins[idLevel][0] * mipmapSize]
			
			sliceFolderIds = sorted(map(int, getSubFoldersNames(mipmapLevelFolder)))
			# sliceLimits = (sliceFolderNumbers[0], sliceFolderNumbers[-1] + 1) # take simply 0 and the max. There is a problem when not taking 0, I do not understand why
			# sliceLimits = (0, sliceFolderNumbers[-1] + 1) # take simply 0 and the max. There is a problem when not taking 0, I do not understand why
			sliceMax = max(sliceFolderIds)

			rowColPairs = getRowColPairs(mipmapLevelFolder)
			# maxRow, maxCol = getMaxRowMaxCol(mipmapLevelFolder) # for all sections
			# rowColPairs = list(np.ndindex(maxRow + 1, maxCol + 1)) # all (row,col) pairs to process for current level
			print('There are ', len(rowColPairs), 'pairs at level', level)

			rowColPairs = list(rowColPairs)
			rowColPairs = rowColPairs[threadId::nThreads] # only subset of tasks done by this thread

			for rowColPair in rowColPairs:
				writer(rowColPair, level, mipmapLevelFolder, sliceMax, precomputedFolder, mipmapSize, info, blackImage, visualizationMode, offset)


def parse_command_line(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('mipmapFolder', help = 'Folder of the render mipmap files')
	parser.add_argument('precomputedFolder', help = 'Where do you want to save the precomputed raw data')
	parser.add_argument('mipmapSize', help = 'mipmapSize')
	parser.add_argument('infoPath', help = 'Where is the info file')
	parser.add_argument('nThreads', help = 'nThreads : tells this thread what to process (only tasks with id%nThreads == i)')
	parser.add_argument('threadId', help = 'threadId : tells this thread what to process (only tasks with id%nThreads == i)')
	parser.add_argument('visualizationMode', help = '"local" for local viewing, "online" for online viewing')
	args = parser.parse_args()
	return args

def main(argv):
	args = parse_command_line(argv)
	return mipmapToPrecomputed(args.mipmapFolder, args.precomputedFolder, int(args.mipmapSize), args.infoPath, int(args.nThreads), int(args.threadId), args.visualizationMode) or 0

if __name__ == "__main__":
	sys.exit(main(sys.argv))
