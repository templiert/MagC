# Platforms
Windows  - for all steps except Linux for after LM-EM registration (for exports to render and Neuroglancer)  
Linux - should work too but currently untested (probably to adjust: call to concorde linkern for reordering, maybe add some fc.cleanLinuxPath to problematic paths with trakEM)

# Installation
Download Fiji - java8  
Download Fiji - java6 (needed because some components currently broken in the java 8 version, e.g., elastic montage and moving least squares transforms in trakEM2)  
Place the file fijiCommon.py in the 'plugins' folder of Fiji: it is a library of helpful functions.  
Python 2 for everything until final data export in linux  - Typically install with anaconda  
Python 3 for the data export in linux  
Git for windows recommended to make command line calls  
The software Concorde for solving traveling salesman problems. On Windows, download [linkern-cygwin](http://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm) and place both linkern.exe and cygwin1.dll in this locally cloned repository.


# Imaging

## LM Imaging
### Wafer overview for section segmentation

No scripts were used for the acquisition of low magnification (5x) brightfield imagery of wafers. Using the software of my microscope (ZEN) I acquired DAPI and brightfield mosaics of the wafer. Assemble the obtained data in a folder:

```
AllImages
│ProjectName_Fluo_b0s0c0x22486-1388y7488-1040m170.tif
│... (all tiles from the Fluo DAPI channel)
│
│ProjectName_BF_b0s0c0x22486-1388y7488-1040m170.tif
│...(all tiles from the Brightfield channel)
│
│Mosaic_Metadata.xml (the mosaic metadata file written by ZEN, rename it with this exact name)
```
This part of the name "b0s0c0x22486-1388y7488-1040m170" comes from ZEN and cannot be controlled. Any 'ProjectName' is ok, but  ```_Fluo_``` and ```_BF_``` must be present to indicate which channel is which.

If you are not using ZEN, then adjust the code to use your own mosaic format.

### Fluorescent imaging of beads for section order retrieval
After following the section segmentation step explained later, you will have the folder 'preImaging' containing 

- images showing the locations of the landmarks: these images help navigating when setting up the landmarks at the light and electron microscopes
- text files with 
	- the locations of the corners of the magnetic portion of the sections 
	- the locations of the corners of the tissue portion of the sections 
	- the locations of the landmarks
	- the locations of the corners of the tissue and of the magnetic portion of a reference section 
	- the locations of the corners of the tissue portions and of a region of interest relative to that tissue portion (optional)

Configure your microscope to be usable with Micromanager. In LM_Imaging.py, adjust the paths of the configuration at the beginning of the script. The current configuration is suited for the Nikon BC Nikon2 G-Floor of Peter's lab at ETHZ. 

To use with another microscope, you will probably need to adjust names of some components (e.g., 'Objective-Turret' might have another name on another microscope, etc.)

#### Setup
- Load the wafer and set the 20x objective. 
- In LM_Imaging.py, adjust experiment parameters such as, waferName, channels, mosaic size. These parameters are at the beginning of the script.

#### Calibrate landmarks
- Run LM_imaging.py with spyder. In the GUI, click on the button "loadSectionsAndLandmarksFromPipeline" and select the 'preImaging' folder containing the section coordinates.
- Click on "Live BF" to activate live brightfield imaging. Using the overview images from the 'preImaging' folder, locate the first landmark on wafer (red cross) by moving the stage. Once the landmark is centered on the central cross in the field of view, press 'Add lowres landmark'.  
- Navigate to the second landmark and press again 'Add lowres landmark'. This button press has triggered the movement of the stage to the 3rd landmark: adjust it to the center and press 'Add lowres landmark' again.
- The last button press has triggered again a stage movement. The stage is now centered on the 4th landmark: adjust and press again the 'Add lowres landmark'. All four landmarks are now calibrated (message updates are given in the console). If you had defined more than 4 landmarks in the wafer segmentation pipeline, then you would need to adjust more landarks similarly. Only the first two landmarks need to be adjusted manually.

Warning: when using with another microscope, make sure that the axes are not flipped and that there is no scaling different between the x and y axes (e.g., the confocal Visitron at ETHZ has a factor 3.x between x and y axes, and the y axis is flipped). Adjust the getXY and setXY functions accordingly (e.g., y = -y, x = x * 1/3.), etc.).

- To verify the position of the landmarks, now click "Add highres landmark": it will move to the first landmark. Adjust the landmark if needed, then press again 'Add highres landmark' and so on until the last (typically 4th) landmark is calibrated. The landmarks are now calibrated.

This "Add highres landmarks" procedure is actually useful when calibrating with 20x without oil and then calibrating with a higher-magnification oil objective (typically done for imaging immunostained sections at high resolution).

After successful calibration, a file "target_highres_landmarks.txt" has been added in the preImaging folder. Keep it for further processing in the section reordering part of the pipeline (this file helps orienting correctly the acquired images).

#### Calibrate hardware autofocus
To calibrate the hardware autofocus (HAF)
- start live imaging with a fluorescent channel with which beads are visible (e.g. "Live Green" for 488)
- locate a patch of fluorescent beads
- press the button "ToggleNikonHAF" (you hear a beep from the hardware autofocus)
- adjust the focus with the wheel of the HAF
- press again "ToggleNikonHAF".

If there is a focus offset between different channels, then adjust the offset values at the beginning of the script. These values are already calibrated for the ETHZ Nikon microscope.

#### Fluorescent bead acquisition
If you had stopped the GUI, you can restart it by rerunning the script and loading the wafer file that had been automatically saved when calibrating the landmarks (button "load wafer"). The wafer file is in the saveFolder defined in the script. Press the button "Acquire mag HAF" to start the automated acquisition of the bead imagery.

### Fluorescent imaging for immunostained tissue

#### Setup
Load the wafer (which has mounting medium and a coverslip), set the 20x objective, and ensure that the sample holder is well anchored to one corner of the sample holder slot (so that you can remove and replace the sample holder at the same position without too much offset).

#### Calibrate landmarks

The calibration procedure is the same as described earlier for the imaging of beads. After successful calibration of the "low resolution" landmarks (with the 20x, and with the coverslip **without** immersion oil)
- remove the sample holder
- add immersion oil on the coverslip above the area with sections (ensure that no oil is touching the wafer, which would be a bad contamination)
- set the 63x oil objective
- place back the holder at the same location (make sure it touches well one of the corners the same way as you inserted it before adding immersion oil).

Adjust manually the focus to make sure that the objective is well immersed in the oil, then press "Add highres landmark": it will move the stage to the first landmark. Adjust it and press the same button again, and so on until all landmarks are calibrated.

#### Calibrate hardware autofocus
Same procedure as described earlier for the beads.

#### Acquisition of fluorescently stained tissue
Press "Acquire tissue HAF" to start the automated acquisition.

The ROI in each section is defined in the file "source_ROI_description" in the "preImaging" folder (created in the wafer segmentation part of the pipeline). This ROI description can also be created manually, it contains the coordinates of the four corners of a section (the tissue portion, x,y) and the four corners of the ROI (a,b) (tab-delimited).

```
x1,y1	x2,y2	x3,y3	x4,y4  
a1,b1	a2,b2	a3,b3	a4,b4
```

If there is no "source_ROI_description" file, then the center of the tissue section is the center of the region acquired.

## EM Imaging
The script EM_imaging.py was used with a Zeiss Merlin that controlled the microscope through the Zeiss API.

#### Setup

Load the wafer and adjust imaging parameters (brightness, contrast). These imaging parameters will not be changed during automated imaging and can be changed during the acquisition if needed. Adjust parameters at the bottom of the EM_imaging.py script (mosaic grid, tile size, scan speed, wafer name).

#### Calibrate landmarks

Run EM_Imaging.py with Spyder. Click "loadSectionsAndLandmarksFromPipeline" in the GUI and select the "preImaging" folder containing section coordinates.

Locate the first landmark and center it in the field of view. Similarly as for the LM landmark calibration, repetitively press the same button and calibrate the other landmarks (the first two landmarks are calibrated manually, the following ones are precomputed and you simply need to adjust them).

#### Automated acquisition

Set the correct detector and start automated acquisition with "Acquire wafer". If you want to acquired only a subset of sections, press the "Acquire sub wafer" and then enter the indices of the sections in the spyder console then press enter.

The acquisition of a wafer can be interrupted and restarted. The wafer file keeps track of which sections were already acquired.

The acquisition of a specific ROI in the tissue section is determined the same way as for the LM described earlier, that is, by the text file with the coordinates of a reference tissue section and of a the relative ROI.


# Section segmentation

Organize the sections in a folder like described earlier in the LM 'Wafer overview for section segmentation' paragraph.

Adjust the root folder in the script of sectionSegmentation.py and run it with the Fiji script editor. Follow the instructions that will pop up during the processing.

The output of this script is the folder "preImaging" that contains 
- images showing the locations of the landmarks: these images help navigating when setting up the landmarks at the light and electron microscopes
- text files with 
	- the locations of the corners of the magnetic portion of the sections 
	- the locations of the corners of the tissue portion of the sections 
	- the locations of the landmarks
	- the locations of the corners of the tissue and of the magnetic portion of a reference section 
	- the locations of the corners of the tissue portions and of a region of interest relative to that tissue portion (optional).

# Section order retrieval with fluorescent beads

Organize the fluorescent imagery of the beads acquired with the pipeline with the following format:

```
rootFolder
│└───preImaging (comes from section segmentation part)
│└───section_0000
│   │section_0000_channel_488_tileId_00-00-mag.tif
│   │section_0000_channel_488_tileId_00-01-mag.tif
│   │...
│   │section_0000_channel_546_tileId_00-00-mag.tif
│   │...
│└───section_0001
│└───...
│└───section_n
```

In the file SOR.py (Section Order Retrieval), adjust the inputFolder path to your rootFolder.

Run the SOR.py script from the Fiji script editor. It will output the section order in the folder "calculations" with the name solution488-546.txt (using two fluorescent channels) or solution488.txt (using only one fluorescent channel). You can manually copy paste this file for the data assembly pipeline below.

The script also outputs many trakemProjects that show reordered bead imagery at different stages of the processing and with all fluorescent channels.

# CLEM Data assembly

## Initial folder setup with input data
If you have used a different imaging pipeline than the one described above you should arrange your data with the following format:

```
YourMagCProjectFolder
│MagCParameters.txt
│solutionxxx.txt
│sectionOrder.txt
│LMEMFactor.txt
└───EMDataRaw
│   │SomeName_EM_Metadata.txt  
│   └───section_0000
│   │   │Tile_0-0.tif (Tile_x-y.tif)
│   │   │Tile_0-1.tif
│   │   │Tile_1-0.tif
│   │   │Tile_1-1.tif
│   └───section_0001
│   └───...
│   └───section_n
│
└───LMData
│   │xxx_LM_Meta_Data.txt  
│   └───section_0000
│   │   │section_0000_channel_488_tileId_00-00-tissue.tif
│   │   │section_0000_channel_488_tileId_00-01-tissue.tif
│   │   │...
│   │   │section_0000_channel_546_tileId_00-01-tissue.tif
│   │   │...
│   │   │section_0000_channel_brightfield_tileId_00-01-tissue.tif
│   └───section_0001
│   └───...
│   └───section_n
```
Description of the files above:
- MagCParameters.txt - if you do not put it yourself from the template in the repository, the default one will be added with default parameters. There are surely parameters that you need to adjust.
- solutionxxx.txt (e.g. solution488-546.txt, solution488.txt) - the section reordering solution computed from Concorde from the reordering pipeline using fluorescent beads
- sectionOrder.txt - indices of the sections in the correct order, one number per line; If this file does not exist, then it will be automatically generated from solutionxxx.txt, or it will be generated at the beginning of the EM data assembly pipeline using EM imagery
- LMEMFactor.txt - the magnification factor between LM and EM imagery (float, the file contains this single number). Typically around 7-13 for 60x magnification LM and about 10 nm EM pixel size. Typically measure the distance between 2 easily identifiable points in LM and EM and calculate the distance ratio in piixels. 
- xxx_EM_Metadata.txt - created by EM_Imaging.py. If you do not use this script for EM imaging, look at the Example_EM_Metadata.txt to create this file yourself
- xxx_LM_Meta_Data.txt - created by LM_Imaging.py. If you do not use this script for LM imaging, look at the Example_LM_Meta_Data.txt to create this file yourself  

## Running the pipeline

The pipeline consists of Fiji scripts that are called one after the other externally from the orchestrator python script MagC.py. You can run it directly from where you have cloned the repository. Upon first run it will open a GUI to ask the user to input:
- the location of the Fiji-java8 executable
- the location of the Fiji-java6 executable
- the location of YourMagCProjectFolder  

It will create three corresponding text files in the repository that store the three locations. If you want to change these, edit these files or remove them to trigger the GUI (the GUI does not pop up when these files are already present).

## Scripts of the pipeline

If you want to run only a part of the pipeline, comment out the steps in MagC.py

Here is a brief description of what each script does in the pipeline.
### LM
- preprocess_ForPipeline - copy and reorder the LM sections. Copy EM sections.  
- assembly_LM - preprocess the LM channels (local contrast enhancement, thresholding, and 8-biting). Creates the contrastedBrightfield channel used for alignmnent, stiching, and CLEM registration. Assemble the tiles of the reference brightfield channel in a trakem project according to LM metadata.
- montage_LM - use on of the montage plugins to montage the LM tiles (phase correlation from Cardona, least squares from Saalfeld, or the main Fiji stitching plugin from Preibisch)
- alignRigid_LM - align (with rigid transforms) the 3D stack (using the brightfield imagery). This alignment is not crucial. If it is faulty, set doAlignment in the parameters to 0. The alignment will anyway be redone during the CLEM registration.
- export_LMChannels - export to disk assembled sections from all channels

### EM

- init_EM - read metadata and initialize folder
- EM_Reorderer - performs sections order retrieval using EM imagery. Pairwise similarities between sections are calculated at the center of each tile of the mosaic grid (e.g. 2x2), and then averaged.
- downsample_EM - downsample and preprocess all tiles with local contrast normalization
- assembly_lowEM - assemble the downsampled tiles into a trakem project according to metadata (to determine tile position) followed by montaging with translations (using Fiji's stitching plugin by Preibisch et al.)
- assembly_EM - assemble a trakem project with original resolution using the transforms computed previously on low resolution data
- montage_ElasticEM - montage all tiles with elastic transforms 
- export_stitchedEMForAlignment - downscale and export to file the stitched sections
- reorder_postElasticMontage - reorder projects and exported sections with the order provided in the sectionOrder.txt file (or solutionxxx.txt file if sectionOrder.txt not present)
- alignRigid_EM - rigidly align the low resolution EM stack and propagate the transforms to the high resolution project
- alignElastic_EM' - elastically align the EM stack at full resolution
- export_alignedEMForRegistration' - export all sections to file with the downscaling LMEMFactor so that the exported EM sections have roughly the same resolution as the LM imagery

### LM-EM registration

- compute_RegistrationMovingLeastSquares - compute the cross-modality moving least squares (MLS) LM-EM transforms  
- export_TransformedCroppedLM - export to file affine transformed and cropped LM channels : these images can be transformed with the computed MLS transforms and upscaled to fit in the EM imagery  
- assembly_LMProjects - create trakem projects containing the LM imagery transformed with the MLS transforms (not upscaled)

# Export of assembled data (linux only)
## Install
In a folder, e.g. 'repos', clone the following repositories:
- this repo
- [render](https://github.com/saalfeldlab/render) from Saalfeld's lab

Create a folder with the following data computed from the pipeline above:
```
projects
└───project_yourProjectName
	│ElasticaAlignedEMProject.xml (from the pipeline)
	│LMProject_488.xml (from the pipeline)
	│LMProject_546.xml (from the pipeline)
	│LMProject_brightfield.xml (from the pipeline)
	└───EMData (from the pipeline)
	└───affineCropped_488 (from the pipeline)
	└───affineCropped_546 (from the pipeline)
	└───affineCropped_brightfield (from the pipeline)
```
In trakemToNeuroglancer.py adjust the paths to the 'repo' folder and 'projects' folder.

## Run
Run trakemToNeuroglancer.py. It will  
- create separate render projects for the EM and for the LM channels
- render to file mipmaps from the EM and the LM channels
- create precomputed chunks for the EM and LM channels ready to be visualized with neuroglancer


# Section collection

The script Motor.py allows control of a 2-axis manipulator (Thorlabs) using the [PyAPT library](https://github.com/mcleung/PyAPT) from Michael Leung. Follow instructions on the github page of the repo for installation.

The script syringePump.py allows control of a syringe pump (KDScientific 200) for water infusion and withdrawal.
