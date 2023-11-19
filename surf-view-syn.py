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

	paraview.simple._DisableFirstRenderCameraReset()

	# create a new 'Legacy VTK Reader'
	example_large_it9vtk = LegacyVTKReader(registrationName='example_large_it=9.vtk', FileNames=['/users/misc/psogra20/Desktop/Project/example_large_it=9.vtk'])

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')

	# show data in view
	example_large_it9vtkDisplay = Show(example_large_it9vtk, renderView1, 'UnstructuredGridRepresentation')

	# get color transfer function/color map for 'Scalars10'
	scalars10LUT = GetColorTransferFunction('Scalars10')

	# get opacity transfer function/opacity map for 'Scalars10'
	scalars10PWF = GetOpacityTransferFunction('Scalars10')

	# trace defaults for the display properties.
	example_large_it9vtkDisplay.Representation = 'Surface'
	example_large_it9vtkDisplay.ColorArrayName = ['POINTS', 'Scalars10']
	example_large_it9vtkDisplay.LookupTable = scalars10LUT
	example_large_it9vtkDisplay.SelectTCoordArray = 'None'
	example_large_it9vtkDisplay.SelectNormalArray = 'None'
	example_large_it9vtkDisplay.SelectTangentArray = 'None'
	example_large_it9vtkDisplay.OSPRayScaleArray = 'Scalars10'
	example_large_it9vtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
	example_large_it9vtkDisplay.SelectOrientationVectors = 'None'
	example_large_it9vtkDisplay.ScaleFactor = 9.9
	example_large_it9vtkDisplay.SelectScaleArray = 'Scalars10'
	example_large_it9vtkDisplay.GlyphType = 'Arrow'
	example_large_it9vtkDisplay.GlyphTableIndexArray = 'Scalars10'
	example_large_it9vtkDisplay.GaussianRadius = 0.495
	example_large_it9vtkDisplay.SetScaleArray = ['POINTS', 'Scalars10']
	example_large_it9vtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
	example_large_it9vtkDisplay.OpacityArray = ['POINTS', 'Scalars10']
	example_large_it9vtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
	example_large_it9vtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
	example_large_it9vtkDisplay.PolarAxes = 'PolarAxesRepresentation'
	example_large_it9vtkDisplay.ScalarOpacityFunction = scalars10PWF
	example_large_it9vtkDisplay.ScalarOpacityUnitDistance = 77.2841842661301
	example_large_it9vtkDisplay.OpacityArrayName = ['POINTS', 'Scalars10']
	example_large_it9vtkDisplay.SelectInputVectors = [None, '']
	example_large_it9vtkDisplay.WriteLog = ''

	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	example_large_it9vtkDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 799999.0, 1.0, 0.5, 0.0]

	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	example_large_it9vtkDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 799999.0, 1.0, 0.5, 0.0]

	# reset view to fit data
	renderView1.ResetCamera(False)

	# get the material library
	materialLibrary1 = GetMaterialLibrary()

	# show color bar/color legend
	example_large_it9vtkDisplay.SetScalarBarVisibility(renderView1, True)

	# update the view to ensure updated data information
	renderView1.Update()

	# get 2D transfer function for 'Scalars10'
	scalars10TF2D = GetTransferFunction2D('Scalars10')

	# get layout
	layout1 = GetLayout()

	# layout/tab size in pixels
	layout1.SetSize(1538, 789)

	# current camera placement for renderView1
	renderView1.CameraPosition = [49.5, 49.5, 350.0597994267811]
	renderView1.CameraFocalPoint = [49.5, 49.5, 39.5]
	renderView1.CameraParallelScale = 80.37879073486985

	
	camera=GetActiveCamera()
	renderView1.ResetCamera()
	camera.Elevation(viewpoint[0]) 
	camera.Azimuth(viewpoint[1])
	renderView1.Update()
	# save screenshot
	SaveScreenshot('/users/misc/psogra20/Desktop/Project/new_' + str(rank) + '.png', renderView1, ImageResolution=[1538, 789])


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
	image_path = 'new_' + str(rank) + '.png'


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

