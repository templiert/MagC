import serial
import time

def cSum(l):
	cS = o
	for i in range(len(l)):
		cS ^= l[i]
		print 'csum( ', str(l), ' ) is ', cS
		print formate(cS, '02x')
		return cS

s = serial.Serial('COM4')
print 's', s

# s.close()
# s.open()
# print 's opened', s
# isopen = s.isOpen()
# print 'isopen', isopen

### working
# s.write('ratew?' + '\r\n')
# s.write('volw?' + '\r\n')
# s.write('mode W' + '\r\n')
# s.write('mode?' + '\r\n')

mode = 'i'
# mode = 'W'

iVolume = 2
iRate = 1500

wVolume = 1
wRate = 150


### Withdrawal
if mode == 'W':
	s.write('mode W' + '\r\n')
	time.sleep(0.2)
	s.write('volw ' + str(wVolume) + ' ml' + '\r\n')
	time.sleep(0.2)
	s.write('ratew '+ str(wRate) + ' ml/h' + '\r\n')
	time.sleep(0.2)
elif mode == 'i':
	### Infusion
	s.write('mode i' + '\r\n')
	time.sleep(0.2)
	s.write('voli ' + str(iVolume) + ' ml' + '\r\n')
	time.sleep(0.2)
	s.write('ratei ' + str(iRate) + ' ml/h' + '\r\n')
	time.sleep(0.2)

s.write('run' + '\r\n')
# s.write('stop' + '\r\n')

time.sleep(0.1)
# read = s.read(6)
read = s.read(3)
print 'read', len(read), read

s.close()