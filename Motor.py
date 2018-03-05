# -*- coding: utf-8 -*-
"""
Example code showing how to control Thorlabs TDC Motors using PyAPT
V1.2
Michael Leung
mcleung@stanford.edu
"""
from PyAPT import APTMotor
import time

def left(d, v = 0.3):
	motVel = v #motor velocity, in mm/sec
	Motor2.mcRel(d, motVel) # Negative Right / Positive Left

def right(d, v = 0.3):
	motVel = v #motor velocity, in mm/sec
	Motor2.mcRel(-d, motVel) # Negative Right / Positive Left

def up(d, v = 0.3):
	motVel = v #motor velocity, in mm/sec
	Motor1.mcRel(d, motVel) # negative me / positive knife

def down(d, v = 0.3):
	motVel = v #motor velocity, in mm/sec
	Motor1.mcRel(-d, motVel) # negative me / positive knife

Motor1 = APTMotor(45844773, HWTYPE=42)
Motor2 = APTMotor(45844576, HWTYPE=42)

# print 'Motor1', Motor1.getPos()
# time.sleep(0.1)
# print 'Motor2', Motor2.getPos()

# offset1 = 0
# offset2 = 0

# print 'Motor1', Motor1.getStageAxisInformation()
# Motor1.setStageAxisInformation(offset1, offset1 + 145)
# print 'Motor1', Motor1.getStageAxisInformation()

# print 'Motor2', Motor2.getStageAxisInformation()
# Motor2.setStageAxisInformation(offset2, offset2 + 145)
# print 'Motor2', Motor2.getStageAxisInformation()

######################################################################

# down(5, v = 0.3)
# up(5, v = 0.3)
# left(5, v = 0.3)
right(40, v = 0.2)

######################################################################

Motor1.cleanUpAPT()
Motor2.cleanUpAPT()