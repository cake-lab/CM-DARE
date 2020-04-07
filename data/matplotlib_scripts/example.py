#!env python

import matplotlib.pyplot as plt
import numpy as np

import common



# Common Function used for demonstration
def plotOnAx(ax, num_lines, colors, markers):

	# Plot lines with different colors, markers and labels

	X = np.arange(0, 10.1, 1) # to 10.1 so it includes 10
	for i in range(num_lines):
		Y = (1+i) * X
		ax.plot(X, Y, color=colors[i], marker=markers[i], label=("Line%s" % i))

	# Don't forget to make your plot look good and sensible!
	ax.set_xlim([0,10])
	ax.set_ylim([0,(10*num_lines)])

	# Set X ticks
	# Note: Not all ranges are good for all image sizes! Compare the PNG and PDF
	ax.set_xticks(np.arange(0, 10.1, 1))
	ax.set_yticks(np.arange(0, (10*num_lines)+0.1, num_lines))

	# Add a legend so it makes sense
	ax.legend(loc='best')

	# Add labels so we know what we're looking at
	ax.set_xlabel("Time (ms)")
	ax.set_ylabel("Total Cost ($)")



num_lines = 3

# Get Standard markers and BW Color palette
colors = common.getBWColors(num_lines)
markers = common.getMarkers()


# Load a common matplotlibrc file
# Note: if this file is located at ~/.matplotlib/matplotlibrc then this is unneeded
common.loadRCParamsFile('./matplotlibrc')


# Generate a test image in markdown style
common.setRCParams("markdown")
fig, ax = plt.subplots(nrows=1, ncols=1)

plotOnAx(ax, num_lines, colors, markers)

plt.savefig('test_image')



# Generate a test image in paper (single column width) style
common.setRCParams("0.5")
fig, ax = plt.subplots(nrows=1, ncols=1)

plotOnAx(ax, num_lines, colors, markers)

plt.savefig('test_image')

