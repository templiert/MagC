from __future__ import with_statement
from ij import IJ
from ij import ImagePlus
from ij import WindowManager
from ij.process import ByteProcessor
import os, time, pickle, threading
import sys
sys.path.append(IJ.getDirectory('plugins'))
import fijiCommon as fc 
from java.awt import Rectangle, Color
from java.awt.geom import AffineTransform
from java.util import HashSet, ArrayList
from java.lang import Runtime
from java.util.concurrent.atomic import AtomicInteger
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch
from ini.trakem2.imaging import Blending
from mpicbg.trakem2.align import RegularizedAffineLayerAlignment
from bunwarpj.bUnwarpJ_ import computeTransformationBatch, elasticTransformImageMacro
from bunwarpj import MiscTools

from mpicbg.ij import SIFT
from mpicbg.imagefeatures import FloatArray2DSIFT

from register_virtual_stack import Transform_Virtual_Stack_MT
from register_virtual_stack import Register_Virtual_Stack_MT

from mpicbg.models import PointMatch
from mpicbg.ij import FeatureTransform
from mpicbg.models import RigidModel2D, AffineModel2D
from mpicbg.models import NotEnoughDataPointsException

from mpicbg.trakem2.transform import MovingLeastSquaresTransform
from mpicbg.trakem2.transform import CoordinateTransform
from mpicbg.trakem2.transform import CoordinateTransformList
from mpicbg.trakem2.transform import TranslationModel2D
from mpicbg.trakem2.transform import TransformMesh

import java
from Jama import Matrix
from Jama import SingularValueDecomposition
import jarray


def pythonToJamaMatrix(m):
	a = Matrix(jarray.array([[0]*len(m) for id in range(len(m))], java.lang.Class.forName("[D")))
	for x, col in enumerate(m):
		for y, val in enumerate(col):
			a.set(x, y, m[x][y])
	return a

def getSIFTMatchingParameters(steps, initialSigma, minOctaveSize, maxOctaveSize, fdBins, fdSize):
	p = FloatArray2DSIFT.Param().clone()
	p.steps = steps
	p.initialSigma = initialSigma
	p.minOctaveSize = minOctaveSize
	p.maxOctaveSize = maxOctaveSize
	p.fdBins = fdBins
	p.fdSize = fdSize
	return p

def getFeatures(imPath, p):
	features = HashSet()
	im = IJ.openImage(imPath)
	SIFT(FloatArray2DSIFT(p)).extractFeatures(im.getProcessor(), features)
	IJ.log(str(features.size()) + ' features extracted' )
	im.close()
	return features

def getMatchingResults(features1, features2):
	candidates = ArrayList()
	inliers = ArrayList()
	FeatureTransform.matchFeatures(features1, features2, candidates, 0.92)
	# FeatureTransform.matchFeatures(features1, features2, candidates, 0.95)
	model = AffineModel2D()
	try:
		modelFound = model.filterRansac(candidates, inliers, 1000, 10, 0, 7) # (candidates, inliers, iterations, maxDisplacement, ratioOfConservedFeatures, minNumberOfConservedFeatures)
	except NotEnoughDataPointsException, e:
		modelFound = False
		IJ.log('NotEnoughDataPointsException')
		return None
	if not modelFound:
		IJ.log('model not found ')
		return None
	else:
		IJ.log('model found')
		return [model, inliers]

def getScalingFactors(aff):
	m = pythonToJamaMatrix([[aff.getScaleX(), aff.getShearX()], [aff.getShearY(), aff.getScaleY()]])
	SVD = SingularValueDecomposition(m)
	S = SVD.getS().getArrayCopy()
	return S[0][0], S[1][1]			
	
def computeRegistration():
	while atomicI.get() < nSections:
		k = atomicI.getAndIncrement()
		if k < nSections:
			l = k
			IJ.log('Computing EM/LM registration for layer ' + str(l).zfill(4))

			layerFolder = fc.mkdir_p(os.path.join(registrationFolder, 'layer_' + str(l).zfill(4)))
			toRegisterFolder = fc.mkdir_p(os.path.join(layerFolder, 'toRegister'))
			registeredFolder = fc.mkdir_p(os.path.join(layerFolder, 'registered'))

			# Applying appropriate filters to make lowresEM and LM look similar for layer l
			imLM = IJ.openImage(imPaths['LM'][l])
			imLM = fc.localContrast(imLM)
			imLMPath = os.path.join(toRegisterFolder, 'imLM_' + str(l).zfill(4) + '.tif')
			IJ.save(imLM, imLMPath)

			imEM = IJ.openImage(imPaths['EM'][l])
			imEM = fc.localContrast(imEM)
			imEMPath = os.path.join(toRegisterFolder, 'imEM_' + str(l).zfill(4) + '.tif')
			IJ.save(imEM, imEMPath)

			# Compute first a simple affine registration on the non-cropped images
			IJ.log('Computing affine and moving least squares alignment for layer ' + str(l).zfill(4))
			firstStepRegistered = False
			
			# registration at first step with 1step/octave (less features)
			pLowRes = getSIFTMatchingParameters(nOctaves[0], 1.6, 16, 4000, 8, 4)

			featuresLM = getFeatures(imLMPath, pLowRes)
			featuresEM = getFeatures(imEMPath, pLowRes)

			matchingResults = getMatchingResults(featuresLM, featuresEM)
			if matchingResults is None:
				IJ.log('No registration matching at low resolution matching step 1 in layer ' + str(l).zfill(4))
			else:
				model, inliers = matchingResults
				distance = PointMatch.meanDistance(inliers) # mean displacement of the remaining matching features
				IJ.log('---Layer ' + str(l).zfill(4) + ' distance ' + str(distance) + ' px with ' + str(len(inliers)) + ' inliers')
				if distance > matchingThreshold[0]:
					IJ.log('Matching accuracy is lower than the threshold at the low resolution step 1 - ' + str(l).zfill(4) + ' - distance - ' + str(distance))
				else:					
					affTransform = model.createAffine()
					s1, s2 = getScalingFactors(affTransform)
					IJ.log('Layer ' + str(l).zfill(4) + ' scaling factors - step 1 - ' + str(s1) + ' - ' + str(s2) + '--' + str(s1*s2) + ' affDeterminant ' + str(affTransform.getDeterminant()) + ' nInliers ' + str(len(inliers)))
					if (abs(s1-1) < 0.2) and (abs(s2-1) < 0.2): # scaling in both directions should be close to 1
						IJ.log('First step ok - layer ' + str(l).zfill(4))
						firstStepRegistered = True
						loaderZ.serialize(affTransform, os.path.join(registeredFolder, 'affineSerialized'))

			if not firstStepRegistered:
				IJ.log('First step registration in layer ' + str(l).zfill(4) + ' with few features has failed. Trying with more features.')
				# registration at first step with 3steps/octave (more features)
				# pLowRes = getSIFTMatchingParameters(3, 1.6, 64, 4000, 8, 4)
				pLowRes = getSIFTMatchingParameters(nOctaves[0], 1.6, 16, 4000, 8, 4) # for BIB


				featuresLM = getFeatures(imLMPath, pLowRes)
				featuresEM = getFeatures(imEMPath, pLowRes)

				matchingResults = getMatchingResults(featuresLM, featuresEM)
				if matchingResults is None:
					IJ.log('No registration matching at low resolution matching step 1bis in layer ' + str(l).zfill(4))
				else:
					model, inliers = matchingResults
					distance = PointMatch.meanDistance(inliers) # mean displacement of the remaining matching features
					IJ.log('---Layer ' + str(l).zfill(4) + ' distance ' + str(distance) + ' px with ' + str(len(inliers)) + ' inliers')
					if distance > matchingThreshold[0]:
						IJ.log('Matching accuracy is lower than the threshold at the high resolution step 1bis - ' + str(l).zfill(4) + ' - distance - ' + str(distance))
					else:					
						affTransform = model.createAffine()
						s1, s2 = getScalingFactors(affTransform)
						IJ.log('Layer ' + str(l).zfill(4) + ' scaling factors - step 1bis - ' + str(s1) + ' - ' + str(s2) + '--' + str(s1*s2) + ' affDeterminant ' + str(affTransform.getDeterminant()) + ' nInliers ' + str(len(inliers)))
						if (abs(s1-1) < 0.2) and (abs(s2-1) < 0.2): # scaling in both directions should be close to 1
							IJ.log('First step 1bis ok - layer ' + str(l).zfill(4))
							firstStepRegistered = True
							loaderZ.serialize(affTransform, os.path.join(registeredFolder, 'affineSerialized'))
				
			if not firstStepRegistered:
				IJ.log('The two first step trials in layer ' + str(l).zfill(4) + ' have failed')
			else:
				# Affine transform and crop the LM, and compute a high res MLS matching
				with lock: # only one trakem working at a time
					# apply affTransform
					patch = Patch.createPatch(pZ, imLMPath)
					layerZ.add(patch)
					patch.setAffineTransform(affTransform)
					patch.updateBucket()
					
					# crop and export
					bb = Rectangle(0, 0, widthEM, heightEM)
					affineCroppedIm = loaderZ.getFlatImage(layerZ, bb, 1, 0x7fffffff, ImagePlus.GRAY8, Patch, layerZ.getAll(Patch), True, Color.black, None)
					affineCroppedImPath = os.path.join(toRegisterFolder, 'affineCroppedLM_' + str(l).zfill(4) + '.tif')
					IJ.save(affineCroppedIm, affineCroppedImPath)
					affineCroppedIm.close()
					
					layerZ.remove(patch)
					layerZ.recreateBuckets()						
				
				pHighRes = getSIFTMatchingParameters(nOctaves[1], 1.6, 64, 4096, 8, 4)
				featuresLM = getFeatures(affineCroppedImPath, pHighRes)
				featuresEM = getFeatures(imEMPath, pHighRes)

				# get the MLS
				matchingResults = getMatchingResults(featuresLM, featuresEM)
				if matchingResults is None:
					IJ.log('It cannot be, there should be a good match given that an affine was computed. Layer ' + str(l).zfill(4))
				else:
					model, inliers = matchingResults
					affTransform = model.createAffine()
					s1, s2 = getScalingFactors(affTransform)
					IJ.log('Second step determinant - layer ' + str(l).zfill(4) + ' - determinant - ' + str(affTransform.getDeterminant()) + ' nInliers ' + str(len(inliers)) + 'Scaling factors - step 2 - ' + str(s1) + ' - ' + str(s2))
					if (abs(s1-1) < 0.2) and (abs(s2-1) < 0.2) and len(inliers) > 50: # scaling in both directions should be close to 1
						distance = PointMatch.meanDistance(inliers) # mean displacement of the remaining matching features
						if distance > matchingThreshold[1]:
							IJ.log('Weird: matching accuracy is lower than the threshold at the high resolution step 2 - ' + str(l).zfill(4) + ' - distance - ' + str(distance))
						else:
							mlst = MovingLeastSquaresTransform()
							mlst.setModel(AffineModel2D)
							mlst.setAlpha(1)
							mlst.setMatches(inliers)

							xmlMlst = mlst.toXML('\t')
							with open(os.path.join(registeredFolder, 'MLST.xml'), 'w') as f:
								f.write(xmlMlst)

							loaderZ.serialize(mlst, os.path.join(registeredFolder, 'mlstSerialized'))
							
							registrationStats.append([l, distance, len(inliers)])

namePlugin = 'compute_RegistrationMovingLeastSquares'
MagCFolder = fc.startPlugin(namePlugin)
ControlWindow.setGUIEnabled(False)

MagCParams = fc.readMagCParameters(MagCFolder)
matchingThreshold = MagCParams[namePlugin]['matchingThreshold'] # rejection threshold for the mean displacement of the transforms for the low and high resolution steps, respectively 
nOctaves = MagCParams[namePlugin]['nOctaves']

width, height, nChannels, xGrid, yGrid, scaleX, scaleY, channels = fc.readSessionMetadata(MagCFolder)

EMMetadataPath = fc.findFilesFromTags(MagCFolder,['EM', 'Metadata'])[0]

nSections = fc.readParameters(EMMetadataPath)['nSections']
# nSections = 20

projectPath = fc.findFilesFromTags(MagCFolder,['EM', 'Project'])[0]
exportedEMFolder = fc.findFoldersFromTags(MagCFolder, ['export_alignedEMForRegistration'])[0]
exportedLMFolder = fc.findFoldersFromTags(MagCFolder,['exported_downscaled_1_' + channels[-1] ])[0] # finds the brightfield contrasted channel
temporaryFolder = fc.mkdir_p(os.path.join(os.path.dirname(projectPath), 'temporary_LMEMRegistration')) # to save contrasted images
registrationFolder = fc.mkdir_p(os.path.join(os.path.dirname(projectPath), 'LMEMRegistration')) # to save contrasted images


imPaths = {}
imPaths['EM'] = [os.path.join(exportedEMFolder, imageName) for imageName in fc.naturalSort(os.listdir(exportedEMFolder)) if os.path.splitext(imageName)[1] == '.tif']
imPaths['LM'] = [os.path.join(exportedLMFolder, imageName) for imageName in fc.naturalSort(os.listdir(exportedLMFolder)) if os.path.splitext(imageName)[1] == '.tif']

# surfaceIds = [0,16,32,48,65,81,97,113,129,145,162,179,195,211,227,243,260,276,293,310] # optimal 16-17
# imPaths['EM'] = [imPaths['EM'][i] for i in surfaceIds]

# get the dimensions of the EM layerset by looking at the dimensions of the first EM image, save for next script
imEM0 = IJ.openImage(imPaths['EM'][0])
widthEM = imEM0.width
heightEM = imEM0.height
imEM0.close()
f = open(os.path.join(registrationFolder, 'lowResEMBounds'), 'w')
pickle.dump([widthEM, heightEM], f)
f.close()

registrationStatsPath = os.path.join(registrationFolder, 'registrationStats')
registrationStats = []

# create dummy trkem for applying affine and cropping LM in the first registration step
pZ, loaderZ, layersetZ, nLayersZ = fc.getProjectUtils(fc.initTrakem(temporaryFolder, 1))
layersetZ.setDimensions(0, 0, widthEM * 5, heightEM * 5)
layerZ = layersetZ.getLayers().get(0)

lock = threading.Lock()

# Setting up the parallel threads and starting them
atomicI = AtomicInteger(0)
fc.startThreads(computeRegistration, fractionCores = 1, wait = 0.5)

fc.closeProject(pZ) # close dummy trakem

# save some stats on the registration
with open(registrationStatsPath, 'w') as f:
	pickle.dump(registrationStats, f)

fc.terminatePlugin(namePlugin, MagCFolder)
# fc.shouldRunAgain(namePlugin, currentLayer, nSections, MagCFolder, '', increment = nLayersAtATime)