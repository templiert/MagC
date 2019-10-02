# orchestrator script that launches all scripts for MagC
import tkinter
from tkinter import filedialog
import os, sys, time
import shutil
import subprocess
import signal
from subprocess import call, Popen
import argparse
import platform

def getDirectory(text):
	root = tkinter.Tk()
	root.withdraw()
	path = filedialog.askdirectory(title = text) + os.sep
	path = os.path.join(path, '')
	return path

def askFile(*args):
	root = tkinter.Tk()
	root.withdraw()
	if len(args) == 1:
		path =  filedialog.askopenfilename(title = args[0])
	else:
		path = filedialog.askopenfilename(title = args[0], initialdir = args[1])
	return path

def whereAmI():
	path = os.path.dirname(os.path.realpath(__file__))
	return os.path.join(path, '')

def whereIs(item, itemType, displayText, MagCScriptsFolder, isNew):
	storedItemPath = os.path.join(MagCScriptsFolder , 'whereIs' + item + '.txt')
	try:
		if isNew:
			raise IOError
		with open(storedItemPath , 'r') as f:
			itemPath = f.readline()
			itemPath = itemPath.replace('\n', '').replace(' ', '')
			itemPath = os.path.join(itemPath)
		if itemType == 'file' and not os.path.isfile(itemPath):
			raise IOError
	except IOError:
		print('I do not know where ', item, ' is')
		if itemType == 'file':
			try:
				itemPath = askFile(displayText)
			except Exception as e:
				print('Please create yourself the files whereIsFiji6.txt, whereIsFiji8.txt, whereIsMagCFolder.txt in the MagC folder. Each file contains the folder or fiji executable location in one line')
				sys.exit()
		elif itemType == 'folder':
			itemPath = getDirectory(displayText)
		with open(storedItemPath, 'w') as f:
			f.write(itemPath)
	return itemPath

def init():

	parser = argparse.ArgumentParser()
	parser.add_argument('-p', default='', help = '"new" (to trigger a dialog to enter the path of the main MagC folder) OR The path to the parent folder that contains all the MagC data')
	parser.add_argument('-f', default='', help = '"new" (to trigger a dialog to enter the path to the Fiji executable) OR The path to the Fiji executable')
	args = parser.parse_args()

	MagCScriptsFolder = whereAmI()
	# get the fiji Path
	if args.p == '' or args.p == 'new':
		fiji8Path = whereIs('Fiji8', 'file', 'Please select the *** JAVA 8 *** Fiji', MagCScriptsFolder, args.p == 'new')
	else:
		fiji8Path = os.path.normPath(args.p) # broken because of 2 fiji

	if args.p == '' or args.p == 'new':
		fiji6Path = whereIs('Fiji6', 'file', 'Please select the *** JAVA 6 *** Fiji', MagCScriptsFolder, args.p == 'new')
	else:
		fiji6Path = os.path.normPath(args.p) # broken because of 2 fiji

	# plugins folder based on fiji path
	fijiPluginsFolders = [os.path.join (os.path.split(fijiPath)[0], 'plugins','') for fijiPath in [fiji8Path, fiji6Path]]

	# copy all the scripts into the plugins folders of Fiji
	for fijiPluginsFolder in fijiPluginsFolders:
		for root, dirs, files in os.walk(MagCScriptsFolder):
			for file in filter(lambda x: x.endswith('.py'), files):
					shutil.copy(os.path.join(root, file), fijiPluginsFolder)

	# get the MagCFolder path
	if args.f == '' or args.f == 'new':
		MagCFolder = whereIs('MagCFolder', 'folder', 'Please select the MagC folder', MagCScriptsFolder, args.f == 'new')
	else:
		MagCFolder = os.path.join(os.path.normPath(args.f),'')

	# If the MagC_Parameters file is not there, then add the standard one from the repo
	MagCParamPath = findFilesFromTags(MagCFolder, ['MagC_Parameters'])
	if len(MagCParamPath) == 0: # MagC_Parameters is not in the data folder
		shutil.copy(os.path.join(MagCScriptsFolder, 'MagC_Parameters.txt'), MagCFolder)

	return MagCFolder, fiji8Path, fiji6Path

def findImageryFolder(MagCFolder, modality):
	ImageryPath = ''
	for (dirpath, dirnames, filenames) in os.walk(MagCFolder):
		for filename in filenames:
			if filename.endswith ( '.zva' * (modality == 'LM') + '.ve-asf' * (modality == 'EM')):
				ImageryPath = dirpath
	return cleanPathForFijiCall(ImageryPath)

def cleanPathForFijiCall(path):
# the path here is provided as an argument to a Fiji script. The path has to be handled differently if it is a folder or a file path so that Fiji understands it well.
	path = os.path.normpath(path)
	if not os.path.isfile(path):
		path = os.path.join(path, '')
	path = path.replace(os.sep, os.sep + os.sep)
	return path

def runFijiScript(plugin):
	fijiFlag = plugin[1]
	plugin = plugin[0]

	repeat = True
	signalingPath = os.path.join(MagCFolder, 'signalingFile_' + plugin.replace(' ', '_') + '.txt')
	print('signalingPath', signalingPath)
	plugin = "'" + plugin + "'"
	arguments = cleanPathForFijiCall(MagCFolder)
	while repeat:
		print('running plugin ', plugin, ' : ', str(time.strftime('%Y%m%d-%H%M%S')))

		# print(' with arguments ', arguments)
		# command = fijiPath + ' -eval ' + '"run(' + plugin + ",'" + arguments  + "'"

		if fijiFlag == 0:
			fijiPath = fiji8Path
		else:
			fijiPath = fiji6Path

		command = fijiPath + ' -eval ' + '"run(' + plugin + ",'" + arguments  + "'" + ')"'
		print('command', command)
		if platform.system() == 'Linux':
			p = subprocess.Popen(command, shell=True, preexec_fn = os.setsid) # do not use stdout = ... otherwise it hangs
		else:
			p = subprocess.Popen(command, shell=True) # do not use stdout = ... otherwise it hangs
			# result = subprocess.call(command, shell=True)

		# print('subprocess', p)

		waitingForPlugin = True
		while waitingForPlugin:
			# print('waitingForPlugin')
			if os.path.isfile(signalingPath):
				time.sleep(2)
				with open(signalingPath, 'r') as f:
					line = f.readlines()[0]
					if line == 'kill me':
						if platform.system() == 'Linux':
							#p.terminate()
							os.killpg(os.getpgid(p.pid), signal.SIGTERM)
						else: # what else ?
							subprocess.call(['taskkill', '/F', '/T', '/PID', str(p.pid)])
						print(plugin , ' has run successfully: ', str(time.strftime('%Y%m%d-%H%M%S')))
						repeat = False
					elif line == 'kill me and rerun me':
						if platform.system() == 'Linux':
							#p.terminate()
							os.killpg(os.getpgid(p.pid), signal.SIGTERM)
						else: # what else ?
							subprocess.call(['taskkill', '/F', '/T', '/PID', str(p.pid)])
						print(plugin , ' has run successfully and needs to be rerun ', str(time.strftime('%Y%m%d-%H%M%S')))
					else:
						print('********************* ERROR')
				print('signalingPath from MagC', signalingPath)
				os.remove(signalingPath)
				waitingForPlugin = False
			time.sleep(1)

		# # # if result == 0:
			# # # print 'result',result
			# # # print plugin , ' has run successfully: ', str(time.strftime('%Y%m%d-%H%M%S'))
			# # # repeat = False
		# # # elif result == 2:
			# # # print plugin , ' has run successfully and needs to be rerun ', str(time.strftime('%Y%m%d-%H%M%S'))
		# # # else:
			# # # print plugin, ' has failed'
			# # # sys.exit(1)

def findFilesFromTags(folder,tags):
	filePaths = []
	for (dirpath, dirnames, filenames) in os.walk(folder):
		for filename in filenames:
			if (all(map(lambda x:x in filename,tags)) == True):
				path = os.path.join(dirpath, filename)
				filePaths.append(path)
	return filePaths

#############################################################
# Script starts here
#############################################################

MagCFolder, fiji8Path, fiji6Path = init()

pipeline = [
#['preprocess ForPipeline', 0],

### LM ###

#['assembly LM', 0],
#['montage LM', 0],
#['alignRigid LM', 0],
#['export LMChannels', 0],

### EM ###

['init EM', 0],
['downsample EM', 0],
['assembly lowEM', 0],
['assembly EM', 0],
['montage ElasticEM', 1], # fails in java8
# ['export stitchedEMForAlignment', 0],
# ['reorder postElasticMontage', 0],
# ['alignRigid EM', 0],
# ['alignElastic EM', 0],
# ['export alignedEMForRegistration', 0],

### LM-EM registration###

# ['compute RegistrationMovingLeastSquares', 1], #fiji8 fails to save MLS transforms
# ['export TransformedCroppedLM', 0],
# ['assembly LMProjects', 1], #java8 fails to apply coordinateTransforms

]

for step in pipeline:
	runFijiScript(step)
