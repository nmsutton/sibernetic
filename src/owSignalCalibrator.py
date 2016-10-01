'''
*******************************************************************************
 * The MIT License (MIT)
 *
 * Copyright (c) 2011, 2013 OpenWorm.
 * http://openworm.org
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License
 * which accompanies this distribution, and is available at
 * http://opensource.org/licenses/MIT
 *
 * Contributors:
 *     	OpenWorm - http://openworm.org/people.html
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************

Methods are included here to calibrate neural signals into input which
creates appropriate actions in sibernetic worm model movements.

Author: Nate Sutton
Copyright 2016
'''

import numpy as np
import scipy.integrate as de
import matplotlib.pyplot as plt

# Initalization parameters
V = -10.001 # inital voltage
W = 10; # target voltage
Z = 0.02; # tau, speed limiting factor
x = np.linspace(0,0.5,200) # time points

class single_neuron():
	def __init__(neuron, V, W, Z, x, lag_time):
		neuron.V = V		
		neuron.W = W
		neuron.Z = Z
		neuron.x = x
		neuron.refrac_delay = 120#0.1
		neuron.refrac_current = 0.0
		neuron.refrac_toggle = False
		neuron.refrac_trigger = 9.9
		neuron.reset_voltage = -10.0
		neuron.lag_time = lag_time
		neuron.lag_counter = 0.0

	def print_i(neuron):
		print(neuron.i)

	def iter_i(neuron):
		neuron.i += 1

	def g(neuron, V, x):
		'''
		Equations for neural spiking
		'''
		V = neuron.refrac_test(V)

		if (neuron.lag_finished()):
			V = (-V+neuron.W)/neuron.Z

		neuron.V = V

		return V

	def lag_finished(neuron):
		neuron.lag_counter += 1

		lag_over = False

		if neuron.lag_counter > neuron.lag_time:
			lag_over = True

		return lag_over

	def neuron_out(neuron):
		neuron.V = de.odeint(neuron.g, neuron.V, neuron.x)

		return neuron.V

	def refrac_test(neuron, V):
		W, Z, refrac_delay, refrac_current, \
		refrac_toggle, refrac_trigger, reset_voltage = \
		neuron.W, neuron.Z, neuron.refrac_delay, \
		neuron.refrac_current, neuron.refrac_toggle, \
		neuron.refrac_trigger, neuron.reset_voltage

		#refractory point
		if V >= refrac_trigger:
			V = reset_voltage

			refrac_toggle = True

		if refrac_toggle == True:
			W = 0
			refrac_current += 1

			if refrac_current >= refrac_delay: 
				W = 10 	
				refrac_current = 0.0
				refrac_toggle = False 		

		neuron.W, neuron.Z, neuron.refrac_delay, \
		neuron.refrac_current, neuron.refrac_toggle, \
		neuron.refrac_trigger, neuron.reset_voltage = \
		W, Z, refrac_delay, refrac_current, \
		refrac_toggle, refrac_trigger, reset_voltage

		return V		

lag_time = 0.0
sn = single_neuron(V, W, Z, x, lag_time)
no = sn.neuron_out()

lag_time = 23.0
sn_2 = single_neuron(V, W, Z, x, lag_time)
no_2 = sn_2.neuron_out()

plt.plot(x, no[:,0], x, no_2[:,0])
plt.show()
