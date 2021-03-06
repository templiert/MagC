from __future__ import with_statement
import ij
from ij import IJ
import sys
sys.path.append(IJ.getDirectory('plugins'))
import fijiCommon as fc
import os, time, shutil, pickle
from ij.io import Opener, FileSaver
from java.lang import Thread, Runtime, Math
from java.util.concurrent.atomic import AtomicInteger
from ini.trakem2 import ControlWindow
from ij.gui import Roi

def sbemimagePathToName(p, snake=True):
    p = os.path.normpath(p)
    imName = os.path.basename(p)
    tileFolder = os.path.dirname(p)
    imTile = int(os.path.basename(tileFolder).split('t')[1])
    gridFolder = os.path.dirname(tileFolder)
    imGrid = int(os.path.basename(gridFolder).split('g')[1])
    x,y = nToXY(imTile, numTilesX, snake = snake)
    newName = 'Tile_' + str(x).zfill(2) + '-' + str(y).zfill(2) + '.tif'
    return newName

def nToXY(n, nX, snake=True):
    y = Math.floor(n/float(nX))
    if snake:
        if y%2 == 0:
            x = n - y*nX
        else:
            x = nX - (n - y*nX) - 1
    else:
        x = nX - (n - y*nX) - 1
    return int(x), int(y)
    
def resizeAndSave(filePaths, l):
    while l.get() < min(len(filePaths), currentWrittenLayer + nTilesAtATime + 1) :
        k = l.getAndIncrement()
        if k < min(len(filePaths), currentWrittenLayer + nTilesAtATime):

            filePath = filePaths[k]
            imageName = os.path.basename(filePath)

            if sbemimage:
                nSection = int(os.path.basename(os.path.dirname(os.path.dirname(filePath))).split('g')[1])
                imageFolderName = 'section_' + str(nSection).zfill(4)
                tileName = sbemimagePathToName(filePath, snake=False)
                resizedImageName = os.path.splitext(tileName)[0] + '_resized_' + factorString + os.path.splitext(tileName)[1]
            else:
                imageFolderName = os.path.basename(os.path.dirname(filePath))
                resizedImageName = os.path.splitext(imageName)[0] + '_resized_' + factorString + os.path.splitext(imageName)[1]

            resizedFilePath = fc.cleanLinuxPath(os.path.join(downSampledEMFolder, imageFolderName, resizedImageName))
            fc.mkdir_p(os.path.join(downSampledEMFolder, imageFolderName))
            im = Opener().openImage(filePath)
            im = IJ.openImage(filePath)
            # IJ.log('Am I going to process the image: im.height = ' + str(im.height) + ' - tileHeight = ' + str(tileHeight) + ' tile number ' + str(k))
            # crop a few lines at the top only if it has not already been done (sometimes the pipeline gets rerun)
            if int(cropTiles) != 0 and im.height == tileHeight:
                im = fc.crop(im,cropRoi)
            im = fc.normLocalContrast(im, normLocalContrastSize, normLocalContrastSize, 3, True, True)
            # IJ.run(im, 'Replace value', 'pattern=0 replacement=1') # only for final waferOverview
            IJ.save(im, filePath) # save the contrasted file
            
            im = fc.resize(im, scaleFactor)
            #FileSaver(im).saveAsTiff(filePath)
            IJ.save(im, resizedFilePath)
            IJ.log('Image resized to ' + resizedFilePath)
            im.close()

namePlugin = 'downsample_EM'
MagCFolder = fc.startPlugin(namePlugin)
ControlWindow.setGUIEnabled(False)
MagCParameters = fc.readMagCParameters(MagCFolder)

EMDataFolder = os.path.join(MagCFolder, 'EMData')
MagCEMFolder = os.path.join(MagCFolder, 'MagC_EM')

# read metadata
EMMetadataPath = fc.findFilesFromTags(MagCFolder,['EM_Metadata'])[0]
EMMetadata = fc.readParameters(EMMetadataPath)
tileWidth = int(EMMetadata['tileWidth'])
tileHeight = int(EMMetadata['tileHeight'])
numTilesX = EMMetadata['numTilesX']
numTilesY = EMMetadata['numTilesY']
IJ.log('TileWidth ' + str(tileWidth))
IJ.log('TileHeight ' + str(tileHeight))

cropTiles = MagCParameters[namePlugin]['cropTiles'] # 1 for yes, 0 for no
cropRoi = Roi(100, 20, tileWidth - 2*100, tileHeight-20) # remove first lines because the Zeiss API 

# read downsampling factor
downsamplingFactor = MagCParameters[namePlugin]['downsamplingFactor']
scaleFactor = 1./downsamplingFactor
factorString = str(int(1000000*downsamplingFactor)).zfill(8)
filePathsPath = os.path.join(MagCEMFolder, 'imagePathsForDownsampling' + factorString + '.txt')

nTilesAtATime = MagCParameters[namePlugin]['nTilesAtATime']

# create or read the file with the paths to process
if not os.path.isfile(filePathsPath):
    filePaths = []
    for (dirpath, dirnames, filenames) in os.walk(EMDataFolder):
        for filename in filenames:
            if filename.endswith('.tif'):
                imPath = fc.cleanLinuxPath(os.path.join(dirpath, filename))
                filePaths.append(imPath)
    filePaths = fc.naturalSort(filePaths)
    with open(filePathsPath,'w') as f:
        for path in filePaths:
            f.write(path + '\n')
    # pickle.dump(filePaths,f)
else:
    filePaths = []
    with open(filePathsPath,'r') as f:
        lines = f.readlines()
        for line in lines:
            filePaths.append(line.replace('\n', ''))
    # filePaths = pickle.load(f)

filePaths = fc.naturalSort(filePaths)
if os.path.basename(os.path.dirname(os.path.normpath(filePaths[0]))) == 't0000':
    sbemimage = True
else:
    sbemimage = False
IJ.log('sbemimage flag: ' + str(sbemimage))
#Create all the subfolders
downSampledEMFolder = fc.mkdir_p(os.path.join(MagCEMFolder, 'MagC_EM_' + factorString, ''))
# for sectionFolderName in os.walk(EMDataFolder).next()[1]:
    # fc.mkdir_p(os.path.join(downSampledEMFolder, sectionFolderName))

normLocalContrastSize = MagCParameters[namePlugin]['normLocalContrastSize']
# downsample in parallel
threads = []
currentLayerPath = os.path.join(MagCEMFolder, 'currentLayer_' + namePlugin + '.txt')
currentWrittenLayer = fc.incrementCounter(currentLayerPath, increment = nTilesAtATime)
IJ.log(namePlugin + ' layer ' + str(currentWrittenLayer))
atomicI = AtomicInteger(currentWrittenLayer)
fc.startThreads(resizeAndSave, fractionCores = 0.9, wait = 0, arguments = (filePaths, atomicI))

# terminate or rerun if more tiles to be processed  
time.sleep(1)
fc.shouldRunAgain(namePlugin, atomicI.get(), len(filePaths), MagCFolder, '')
