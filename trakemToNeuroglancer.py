# inputFolder
	# EM.xml
	# EMData
	# LM_Channel1.xml
	# finalLM_Channel1
	# LM_Channel2.xml
	# finalLM_Channel2

import os, shutil, subprocess, re, json, pickle, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import xml.etree.cElementTree as ET
import numpy as np

def getSubFoldersNames(folder):
	return sorted([name for name in os.listdir(folder)
			if os.path.isdir(os.path.join(folder, name))])

def getMaxRowMaxCol(folder): # to get the dimensions of the render project by looking at the mipmap folders
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

def renderCatmaidBoxesCall(l):
	p = subprocess.Popen([os.path.join(renderScriptsFolder, 'render_catmaid_boxes.sh'), '--baseDataUrl', url,
	 '--owner', owner, '--project', projectName, '--stack', stackName, '--numberOfRenderGroups', '1',
	  '--renderGroup', '1', '--rootDirectory', mipmapFolder, '--maxLevel', str(nResolutionLevels-1),
	  '--height', str(mipmapSize), '--width', str(mipmapSize), str(l)], cwd = renderScriptsFolder) # can add '--forceGeneration'
	p.wait()

def renderImportJson(path):
	p = subprocess.Popen([os.path.join(renderScriptsFolder, 'import_json.sh'), '--baseDataUrl', url, '--owner', owner, '--project', projectName, '--stack', stackName, path], cwd = renderFolder)
	p.wait()

### Dataset parameters ### (# weird offset when LMResolutionLevels = 3, just take 2 instead)
# EMPixelSize, LMEMFactor, datasetName, LMResolutionLevels, EMResolutionLevels, nMipmapThreads = 8, 10, 'B6', 3, 7, 9
EMPixelSize, LMEMFactor, datasetName, LMResolutionLevels, EMResolutionLevels, nMipmapThreads = 8, 13, 'C1', 4, 7, 9


### What to run in the script ###
XML_to_JSON, JSON_to_Render, Render_to_Mipmaps, Mipmaps_to_Precomputed = 1,1,1,1
doEM = 1
doLM = 1
### ###

visualizationMode = 'online' # 'online' for gs, or 'local' with the HBP docker
chunkSize = [64, 64, 64]

mipmapSize = 2048 # size of the Render mipmap files
# nRenderGroups = 12 # about 16GB of ram needed per renderGroup
# nMipmapThreads = 9
nThreadsMipmapToPrecomputed = 4
rootFolder = os.path.join(r'/home/tt/research/data/trakemToNeuroglancerProjects', '')

datasetFolder = os.path.join(rootFolder, datasetName)
inputFolder = os.path.join(datasetFolder, 'input')

### Manual configuration once ###
reposFolder = os.path.join(r'/home/tt/research/repos', '')
myScriptsFolder = os.path.join(reposFolder, 'puzzletomography', 'renderNG', 'myScriptsOptimized', '')
owner = 'Thomas'
projectName = 'MagC'
url = 'http://localhost:8080/render-ws/v1' # MOST PROBABLY SHOULD NOT BE MODIFIED
### Folders and paths initializations
renderFolder = os.path.join(reposFolder, 'render', '')
renderScriptsFolder = os.path.join(renderFolder, 'render-ws-java-client', 'src', 'main', 'scripts')
neuroglancerFolder = os.path.join(reposFolder, 'neuroglancer', '')
trakemToJsonPath = os.path.join(renderScriptsFolder, 'trakemToJson.sh')
shutil.copyfile(os.path.join(myScriptsFolder, 'trakemToJson.sh'), trakemToJsonPath)

outputFolder = os.path.join(datasetFolder, 'outputFolder')
os.makedirs(outputFolder, exist_ok=True)

try:
	nSections = len(list(os.walk(os.path.join(inputFolder, 'EMData')))[0][1]) # EM gives the number of layers, not the LM where some layers can be empty
except Exception as e:
	nSections = int(raw_input('How many sections are there ?'))

for trakemProjectFileName in filter(lambda x: os.path.splitext(x)[1] == '.xml', os.listdir(inputFolder)):
	print('trakemProjectFileName',trakemProjectFileName)
	trakemProjectPath = os.path.join(inputFolder, trakemProjectFileName)
	if ('LM' in trakemProjectFileName) and doLM:
		pixelSize = EMPixelSize * LMEMFactor
		channelName = os.path.splitext(trakemProjectFileName)[0].split('_')[1]# LMProject_546.xml
		nResolutionLevels = LMResolutionLevels
		if 'LMProject' in trakemProjectFileName:
			dataFolder = os.path.join(inputFolder, 'affineCropped_' + channelName)
			stackName = datasetName + '_LM_' + channelName
		elif 'Segmented' in trakemProjectFileName:
			dataFolder = os.path.join(inputFolder, 'segmentedTracks_' + channelName)
			stackName = datasetName + '_SegmentedLM_' + channelName
		else:
			print('Error in reading an LM project - exit')
			sys.exit()
		# find nonEmptyLayers based on the .xml
		nonEmptyLayers = []
		with open(trakemProjectPath, 'r') as f:
			for line in f:
				if 'file_path="' in line:
					nonEmptyLayers.append(int(float(line.split('_')[-1].replace('.tif"\n','')))) # file_path="finalLM_546/finalLM__546_0001.tif"
		nonEmptyLayers = sorted(list(set(nonEmptyLayers))) # remove duplicates

	elif ('EM' in trakemProjectFileName) and doEM:
		stackName = datasetName + '_EM'
		dataFolder = os.path.join(inputFolder, 'EMData')
		nResolutionLevels = EMResolutionLevels
		pixelSize = EMPixelSize
		nonEmptyLayers = range(nSections)
		channelName = ''
	else:
		print('Either nothing to do (check doLM and doEM), or error because the trakem xml file should contain "LM" or "EM"')
		exit()

	print('\n *** \nProcessing stack', stackName, '\n', 'with pixelSize', pixelSize, '\n', 'channelName', channelName, '\n', 'nNonEmptyLayers', len(nonEmptyLayers), '\n***\n')

	precomputedFolderProject = os.path.join(outputFolder, 'precomputed', stackName)
	os.makedirs(precomputedFolderProject, exist_ok=True)

	trakemDimensionsPath = os.path.join(outputFolder, stackName + '_Dimensions')

	jsonPath = os.path.join(outputFolder, stackName + '.json')

	renderProjectFolder = os.path.join(outputFolder, 'renderProject_' + stackName, '') # new folder that will contain the whole render project
	os.makedirs(renderProjectFolder, exist_ok=True) # xxx should be created ?
	mipmapFolder = os.path.join(renderProjectFolder, 'mipmaps', '')


	# ###################
	# ### XML to JSON ###
	# ###################
	if XML_to_JSON:
		# If LMProject: (the LMSegmented project is not faulty, correct only the projects with the MLSTransforms)
			# add a <ict_transform_list> </ict_transform_list> around the MLST
		trakemProjectCorrectedPath = os.path.join(outputFolder, stackName + '_Corrected.xml')
		with open(trakemProjectPath, 'r') as f, open(trakemProjectCorrectedPath, 'w') as g:
			for line in f:
				if 'LMProject' in trakemProjectFileName:
					# adding the missing <ict_transform_list> </ict_transform_list> around the MLST
					if 'ict_transform class=' in line:
						g.write('\n<ict_transform_list>\n')
					elif '</t2_patch>' in line:
						g.write('\n</ict_transform_list>\n')
				# writing the corrected xml file
				g.write(line)

		p = subprocess.Popen(['chmod +x ' + trakemToJsonPath], shell = True)
		p.wait()

		p = subprocess.Popen([trakemToJsonPath, trakemProjectCorrectedPath, dataFolder, jsonPath], cwd = renderScriptsFolder)
		p.wait()

		# Correct the json:
		# - correct the relative image paths with the new data location
		# - remove initial comma when first layer empty
		# - if LM: reset the patch transform to identity (weirdly added by the converter)
		jsonCorrectedPath = os.path.join(outputFolder, stackName + '_Corrected.json')

		with open(jsonPath, 'r') as f, open(jsonCorrectedPath, 'w') as g:
			for idLine, line in enumerate(f):
				# trakem2.converter adds an unnecessary comma when the first trakem layer is empty
				if (idLine == 1) and (',' in line):
					line = ''
				# correct data location
				elif 'imageUrl' in line:
					if 'EM' in trakemProjectFileName:
						splitted = line.split('EMData')
						line = splitted[0] + 'EMData' + splitted[2]
					if 'LM' in trakemProjectFileName:
						splitted = line.split('file:')
						pathParts = list(Path(splitted[1].replace('"\n', '')).parts)
						del pathParts[-2]
						newPath = os.path.join(*pathParts)
						line = splitted[0] + 'file:' + newPath + '"\n'
						# "imageUrl" : "file:/home/thomas/research/trakemToNeuroglancerProjects/firstMinimalTest/input/finalLM_brightfield/finalLM_brightfield/finalLM__brightfield_0001.tif"
				# correct the wrongly added transform by the converter
				elif ('LM' in trakemProjectFileName) and ('"dataString" : "1.0' in line): # revert to identity the affine transform that has been wrongly added by the trakem2.Converter
					splitted = line.split('"dataString" : "1.0')
					line = splitted[0] + '"dataString" : "1.0 0.0 0.0 1.0 0.0 0.0"\n'
				# writing the corrected json
				g.write(line)

		os.remove(jsonPath)
		os.rename(jsonCorrectedPath, jsonPath)

	# # ######################
	# # ### JSON to Render ###
	# # ######################
	if JSON_to_Render:
		p = subprocess.Popen(['sudo service mongod start'], shell = True)
		p.wait()

		p = subprocess.Popen([os.path.join(renderFolder, 'deploy', 'jetty_base', 'jetty_wrapper.sh')  + ' start'], shell = True)
		p.wait()

		# split the json into smaller ones (the 2012 tiles of A7 trigger a failure)
		tilesPerJson = 100
		splitJsonPaths = []

		with open(jsonPath, 'r') as f:
			mainJson = json.load(f)
			nTiles = len(mainJson)
			splitJsons = [mainJson[i:min(i + tilesPerJson, nTiles)] for i in range(0, nTiles, tilesPerJson)]
			for id, splitJson in enumerate(splitJsons):
				splitJsonPath = os.path.join(outputFolder, stackName + '_' + str(id).zfill(4) + '_splitJson.json')
				splitJsonPaths.append(splitJsonPath)
				if not os.path.isfile(splitJsonPath):
					with open(splitJsonPath, 'w') as g:
						json.dump(splitJson, g, indent=4)

		p = subprocess.Popen([os.path.join(renderScriptsFolder, 'manage_stacks.sh'), '--baseDataUrl', url,
		 '--owner', owner, '--project', projectName, '--stack', stackName,
		  '--action', 'CREATE', '--cycleNumber', str(1), '--cycleStepNumber', str(1)], cwd = renderFolder)
		p.wait()

		with ThreadPoolExecutor(max_workers=6) as executor: # import the jsons into the project
			executor.map(renderImportJson, splitJsonPaths)

		p = subprocess.Popen([os.path.join(renderScriptsFolder, 'manage_stacks.sh'), '--baseDataUrl', url,
		 '--owner', owner, '--project', projectName, '--stack', stackName, '--action', 'SET_STATE', '--stackState', 'COMPLETE'], cwd = renderFolder)
		p.wait()



	# #########################
	# ### Render to MipMaps ###
	# #########################
	# echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
	# sudo sysctl -p
	# inotifywait -m -r -e create /home/tt/research/data/trakemToNeuroglancerProjects/B6/outputFolder

	if Render_to_Mipmaps:
		with ThreadPoolExecutor(max_workers=nMipmapThreads) as executor:
			executor.map(renderCatmaidBoxesCall, nonEmptyLayers)
			# executor.map(renderCatmaidBoxesCall, range(163,175))

	# sudo swapoff -a && sudo swapon -a

	##############################
	### MipMaps to Precomputed ###
	##############################
	if Mipmaps_to_Precomputed:
		mipmapFolderDirect = os.path.join(mipmapFolder, projectName, stackName, str(mipmapSize) + 'x' + str(mipmapSize), '')

		# create the info file
		infoPath = os.path.join(precomputedFolderProject, 'info')
		shutil.copyfile(os.path.join(myScriptsFolder, 'infoTemplate'), infoPath)
		with open(infoPath, 'r') as f:
			info = json.load(f)

		del info['scales'][nResolutionLevels:] # ok with n=4, to be checked for different resolution level numbers

		# get the dimensions of the render universe
		level0MipmapFolder = os.path.join(mipmapFolderDirect, '0')
		maxRow, maxCol = np.array(getMaxRowMaxCol(level0MipmapFolder)) * mipmapSize
		print(maxRow, maxCol)

		for idScale, scale in enumerate(info['scales']):
			resolution = pixelSize * 2**idScale # pixel size increases with powers of 2
			scale['resolution'] = [resolution, resolution, 50]
			scale['chunk_sizes'] = [[chunkSize[0], chunkSize[1], min(nSections, chunkSize[2])]] # nSections-1 because issue with section number 0 ?
			scale['key'] = str(resolution) + 'nm'
			scale['encoding'] = 'raw'
			# adding a *1.5 because I do not understand why otherwise the volume gets truncated at low resolution ...
			scale['size'] = [ int((maxCol // 2**idScale)*1.5) , int((maxRow // 2**idScale)*1.5), nSections] # integers specifying the x, y, and z dimensions of the volume in voxels
			info['scales'][idScale] = scale

		with open(infoPath, 'w') as f:
			json.dump(info, f)

		start = time.perf_counter()
		processes = []
		for threadId in range(nThreadsMipmapToPrecomputed):
			p = subprocess.Popen(['python3', os.path.join(myScriptsFolder, 'mipmapToPrecomputed.py'), mipmapFolderDirect, precomputedFolderProject, str(mipmapSize), infoPath, str(nThreadsMipmapToPrecomputed), str(threadId), visualizationMode]) # nThreads, threadId tells the thread what to process (only tasks with threadId%nThreads = i)
			processes.append(p)
		[p.wait() for p in processes]
		print('Mipmap to precomputed took: ', time.perf_counter() - start, ' seconds')

#####################
### Visualization ###
#####################
'''
Upload to GCS

# install gsutil
curl https://sdk.cloud.google.com | bash
Restart your shell:
exec -l $SHELL
Run gcloud init to initialize the gcloud environment:
gcloud init

gcloud auth login
gcloud config set project affable-ring-187517 # (the project id can be found in the online gs browser)

add 'allUsers' as a member in the IAM setting of GS

cd outputFolder
gsutil -m cp -r -Z precomputed/  gs://xxx
'''