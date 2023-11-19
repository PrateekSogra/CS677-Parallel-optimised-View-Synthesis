import numpy as np
from PIL import Image,ImageOps, ImageFile
from scipy.stats import entropy
from scipy.optimize import minimize
import sys
import skimage.measure
from paraview.simple import *
from mpi4py import MPI
import time
import random
import logging
import threading
import queue


ImageFile.LOAD_TRUNCATED_IMAGES = True


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#print(rank, size)



def generate_image(viewpoint):

	#### disable automatic camera reset on 'Show'
	paraview.simple._DisableFirstRenderCameraReset()

	# create a new 'NetCDF Reader'
	output_weather_simnc = NetCDFReader(registrationName='output_weather_sim.nc', FileName=['/users/misc/psogra20/Desktop/Project/output_weather_sim.nc'])
	output_weather_simnc.Dimensions = '(x, y, z)'

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')

	# show data in view
	output_weather_simncDisplay = Show(output_weather_simnc, renderView1, 'UniformGridRepresentation')

	# trace defaults for the display properties.
	output_weather_simncDisplay.Representation = 'Outline'
	output_weather_simncDisplay.ColorArrayName = [None, '']
	output_weather_simncDisplay.SelectTCoordArray = 'None'
	output_weather_simncDisplay.SelectNormalArray = 'None'
	output_weather_simncDisplay.SelectTangentArray = 'None'
	output_weather_simncDisplay.OSPRayScaleArray = 'field'
	output_weather_simncDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
	output_weather_simncDisplay.SelectOrientationVectors = 'None'
	output_weather_simncDisplay.ScaleFactor = 6.300000000000001
	output_weather_simncDisplay.SelectScaleArray = 'None'
	output_weather_simncDisplay.GlyphType = 'Arrow'
	output_weather_simncDisplay.GlyphTableIndexArray = 'None'
	output_weather_simncDisplay.GaussianRadius = 0.315
	output_weather_simncDisplay.SetScaleArray = ['POINTS', 'field']
	output_weather_simncDisplay.ScaleTransferFunction = 'PiecewiseFunction'
	output_weather_simncDisplay.OpacityArray = ['POINTS', 'field']
	output_weather_simncDisplay.OpacityTransferFunction = 'PiecewiseFunction'
	output_weather_simncDisplay.DataAxesGrid = 'GridAxesRepresentation'
	output_weather_simncDisplay.PolarAxes = 'PolarAxesRepresentation'
	output_weather_simncDisplay.ScalarOpacityUnitDistance = 1.8966608180553592
	output_weather_simncDisplay.OpacityArrayName = ['POINTS', 'field']
	output_weather_simncDisplay.ColorArray2Name = ['POINTS', 'field']
	output_weather_simncDisplay.SliceFunction = 'Plane'
	output_weather_simncDisplay.Slice = 31
	output_weather_simncDisplay.SelectInputVectors = [None, '']
	output_weather_simncDisplay.WriteLog = ''

	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	output_weather_simncDisplay.ScaleTransferFunction.Points = [-4.3200000000000007e-05, 0.0, 0.5, 0.0, 123039.0, 1.0, 0.5, 0.0]

	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	output_weather_simncDisplay.OpacityTransferFunction.Points = [-4.3200000000000007e-05, 0.0, 0.5, 0.0, 123039.0, 1.0, 0.5, 0.0]

	# init the 'Plane' selected for 'SliceFunction'
	output_weather_simncDisplay.SliceFunction.Origin = [15.5, 31.5, 31.5]

	# reset view to fit data
	renderView1.ResetCamera(True)

	# get the material library
	materialLibrary1 = GetMaterialLibrary()

	# update the view to ensure updated data information
	renderView1.Update()

	# set scalar coloring
	ColorBy(output_weather_simncDisplay, ('POINTS', 'field'))

	# rescale color and/or opacity maps used to include current data range
	output_weather_simncDisplay.RescaleTransferFunctionToDataRange(True, True)

	# change representation type
	output_weather_simncDisplay.SetRepresentationType('Volume')

	# rescale color and/or opacity maps used to include current data range
	output_weather_simncDisplay.RescaleTransferFunctionToDataRange(True, False)

	# get color transfer function/color map for 'field'
	fieldLUT = GetColorTransferFunction('field')

	# get opacity transfer function/opacity map for 'field'
	fieldPWF = GetOpacityTransferFunction('field')

	# get 2D transfer function for 'field'
	fieldTF2D = GetTransferFunction2D('field')

	# get layout
	layout1 = GetLayout()

	# layout/tab size in pixels
	layout1.SetSize(1538, 789)

	# current camera placement for renderView1
	renderView1.CameraPosition = [15.5, 31.5, 213.7402813226413]
	renderView1.CameraFocalPoint = [15.5, 31.5, 31.5]
	renderView1.CameraParallelScale = 47.167255591140766
	
	camera=GetActiveCamera()
	renderView1.ResetCamera()
	camera.Elevation(viewpoint[0]) 
	camera.Azimuth(viewpoint[1])
	renderView1.Update()

	# save screenshot
	SaveScreenshot('/users/misc/psogra20/Desktop/Project/Final_Code/6_' + str(rank) + '.png', renderView1, ImageResolution=[1538, 789])

#================================================================

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).



#objective function
def objfunc(viewpoint):
	#viewpoint = solution.get_x()
	#get image from viewpoint
	generate_image(viewpoint)
	print(viewpoint, flush = True)
	image_path = '6_' + str(rank) + '.png'


	image = Image.open(image_path)
	image = ImageOps.grayscale(image)
	# rgblist = Image.Image.split(image)
	# rbglist[0].convert()
	image_array = np.array(image)

	#red_channel = image_array[:, :, 0]
	#green_channel = image_array[:, :, 1]
	#blue_channel = image_array[:, :, 2]

	#entropy_red = entropy(red_channel.ravel())
	#entropy_green = entropy(green_channel.ravel())
	#entropy_blue = entropy(blue_channel.ravel())

	#overall_entropy = (entropy_red + entropy_green + entropy_blue) / 3.0

	overall_entropy = skimage.measure.shannon_entropy(image_array)
	print(f'Overall entropy of the image: {overall_entropy} ' + str(rank))
	return -1*(overall_entropy) # we minimise this


#optimiser
def optim(x,y):
	x0 = [x,y]
	bounds = [(max(-90,x0[0] -20), min(90, x0[0] + 20)), (max(-180,x0[1] -20), min(180, x0[1] + 20))]
	#dim = Dimension(2, [[x0[0] -20, x0[0] + 20], [x0[1] -20, x0[1] + 20]], [False]*2)
	#obj = Objective(objfunc, dim)
	result = minimize(objfunc, x0 = x0, bounds = bounds, method= 'Nelder-Mead', options={ 'maxfev' : 200})
	#result = Opt.min(obj, Parameter(budget=100*2))

	return result.x

#main


if rank != 0:

	q = queue.Queue()

	#take initial viewpoints
	views = comm.recv()
	
	threads = []
	for i in range(1):
		threads.append(threading.Thread(target=optim, args=views[i], name="thread" + str(i)))
		threads[i].start()
		
	print(str(rank) + ' initialised',flush = True)
	
	
	#creating new threads to replace completed ones
	while(1):		
		#name of dead thread
		thname = q.get()
		print("Rank " + str(rank) + ", " + str(thname) + ' completed!',flush = True)
		#ask for new viewpoint
		comm.send(rank, dest=0)
		
		#get new viewpoint
		newv = comm.recv()
		print(str(rank) + ' got new viewpoint',flush = True)
		
		#start new thread with new viewpoint and same name as dead thread
		newthread = threading.Thread(target=optim, args= newv, name=thname)
		newthread.start()
		print("Rank " + str(rank) + ", " + str(thname) + ' created again.',flush = True)




elif rank == 0:
	#get data from csv
	my_data = np.loadtxt('data.csv', delimiter=',')
	my_data = my_data[7:]
	n = len(my_data)
	my_data = my_data.tolist()
	#my_data = [[random.randint(0, 100), random.randint(0, 100)] for _ in range(100)]
	print(my_data,flush = True)
	#initialise all processes
	for rrank in range (1,size):
		comm.send(my_data[:4], dest= rrank)
		my_data = my_data[4:] #remove first 4 elements
	
	print(my_data,flush = True)
	#while list isnt empty, keep receieving and sending
	while (len(my_data) > 0):
		rrec = comm.recv()
		print('Receieve from rank ' + str(rrec),flush = True)
		#print('To send: ',my_data[0],flush = True)
		comm.send(my_data[0],dest= rrec)
		my_data = my_data[1:]
		
	comm.recv()
	print('all done')
	MPI.Finalize();
	sys.exit()

#x0 = [0,0]
#generate_image([0,0])
#x0[0] = float(sys.argv[1])
#x0[1] = float(sys.argv[2])
#final_view = optim(x0)

