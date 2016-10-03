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
from scipy.integrate import ode

# Initalization parameters
V = -25.001#-10.001 # inital voltage
W = 25.0;#10; # target voltage
Z = 0.02; # tau, speed limiting factor
x = np.linspace(0,0.5,200) # time points

class single_neuron():
	def __init__(neuron, V, W, Z, x, lag_time):
		neuron.V = V		
		neuron.W = W
		neuron.Z = Z
		neuron.x = x
		neuron.target_V = W
		neuron.reset_V = V + .001#-10.0		
		neuron.refrac_delay = 120#0.1
		neuron.refrac_current = 0.0
		neuron.refrac_toggle = False
		neuron.refrac_trigger = W - .1#9.9
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
		refrac_toggle, refrac_trigger, reset_V, target_V = \
		neuron.W, neuron.Z, neuron.refrac_delay, \
		neuron.refrac_current, neuron.refrac_toggle, \
		neuron.refrac_trigger, neuron.reset_V, neuron.target_V

		#refractory point
		if V >= refrac_trigger:
			V = reset_V

			refrac_toggle = True

		if refrac_toggle == True:
			W = neuron.reset_V
			refrac_current += 1

			if refrac_current >= refrac_delay: 
				W = neuron.target_V 	
				refrac_current = 0.0
				refrac_toggle = False 		

		neuron.W, neuron.Z, neuron.refrac_delay, \
		neuron.refrac_current, neuron.refrac_toggle, \
		neuron.refrac_trigger, neuron.reset_V, neuron.target_V = \
		W, Z, refrac_delay, refrac_current, \
		refrac_toggle, refrac_trigger, reset_V, target_V

		return V	

class movement_angle:
	def __init__(m, sections, neurons_V, x):
		m.a = 0.0 # angle
		m.da = 0.0 # angle derivative
		m.sections = sections
		m.neurons_V = neurons_V
		m.t = 0
		m.t_counter = 0
		m.old_x = 0
		m.x = x		
		m.x_rounded = [ '%.9f' % elem for elem in x.tolist() ]
		m.x_int = range(len(x))
		m.w_max_db = 9951
		m.w_max_vb = 6987
		m.D_V_db = 0.0826
		m.D_V_vb = 0.2888
		m.V_0_db = 25.0
		m.V_0_vb = 22.8

	def sigmoid_out(m, section, V):
		w_max = 0.0
		D_V = 0.0	
		V_0 = 0.0		

		if section == 'vb':
			w_max = m.w_max_vb
			V_0 = m.V_0_vb
			D_V = m.D_V_vb

		elif section == 'db':
			w_max = m.w_max_db
			V_0 = m.V_0_db
			D_V = m.D_V_db				
		
		return w_max/(1+np.exp(-(V-V_0)/D_V))

	def angle_deriv(m, t, y):
		sec_1 = m.sections[0]
		sec_2 = m.sections[1]

		V_1 = m.neurons_V[0][m.t][0]
		V_2 = m.neurons_V[1][m.t][0]

		y[0] = ((m.w_max_vb/(1+np.exp(-(V_1-m.V_0_vb)/m.D_V_vb))) - \
		(m.w_max_db/(1+np.exp(-(V_2-m.V_0_db)/m.D_V_db))))

		return y

	def angle_out(m):
		mi = ode(m.angle_deriv).set_integrator('vode', method='adams',
							order=10, atol=1e-6,
							with_jacobian=False)

		y = 20.0
		mi.set_initial_value(y, 0)
		T = 0.5
		dt = T/200
		u = [];	t = []
		while mi.successful() and mi.t <= T:
			#print(m.t)
			m.t = int(1+mi.t*(1/dt))
			mi.integrate(mi.t + dt)
			'''
			# apply rotation matrix
			rot = np.pi*.6
			x2 = mi.t
			y2 = mi.y[0]
			x_rot = np.cos(rot)*x2+np.sin(rot)*y2
			y_rot = -np.sin(rot)*x2+np.cos(rot)*y2
			u.append(y_rot);	t.append(x_rot)
			'''
			u.append(mi.y[0]);	t.append(mi.t)

		m.a = u

		return m.a	

lag_time = 0.0
sn = single_neuron(V, W, Z, x, lag_time)
neuron_1_V = sn.neuron_out()

lag_time = 23.0
sn_2 = single_neuron(V, W, Z, x, lag_time)
neuron_2_V = sn_2.neuron_out()

sections = 'vb', 'db'
neurons_V = neuron_1_V, neuron_2_V
find_angle = movement_angle(sections, neurons_V, x)
a_1 = find_angle.angle_out()
#print(a_1)

plt.subplot(2, 1, 1)
plt.plot(x, neuron_1_V[:,0], x, neuron_2_V[:,0])
plt.grid(True, color='0.9', linestyle='-', which='both', axis='both')
plt.title('Neuron Voltage')

plt.subplot(2, 1, 2)
plt.plot(x, a_1)
plt.grid(True, color='0.9', linestyle='-', which='both', axis='both')
plt.title('Bending Angle')
plt.show()