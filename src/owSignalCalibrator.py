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

class single_neuron():
	def __init__(neuron, V, W, Z, x, reset_V, lag_time):
		neuron.V = V		
		neuron.W = W
		neuron.Z = Z
		neuron.x = x
		neuron.target_V = W
		neuron.reset_V = reset_V		
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
	def __init__(m, sections, neurons_V, time_params, x):
		m.a = 0.0 # angle
		#m.da = 0.0 # angle derivative
		m.sections = sections
		m.neurons_V = neurons_V
		m.t = 0
		m.old_t = 0		
		#m.t_counter = 0
		m.du = []
		m.x = x		
		#m.x_rounded = [ '%.9f' % elem for elem in x.tolist() ]
		m.x_int = range(len(x))
		m.w_max_db = 9951
		m.w_max_vb = 6987
		m.D_V_db = 0.0826
		m.D_V_vb = 0.2888
		m.V_0_db = 25.0
		m.V_0_vb = 22.8
		m.start = time_params[0]
		m.end = time_params[1]
		m.steps = time_params[2]

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

		#y[0] = m.sigmoid_out(sec_1, V_1) - m.sigmoid_out(sec_2, V_2)

		return m.sigmoid_out(sec_1, V_1) - m.sigmoid_out(sec_2, V_2)#y

	def angle_out(m):
		mi = ode(m.angle_deriv).set_integrator('vode', method='adams',
							order=10, atol=1e-6,
							with_jacobian=False)

		y = 0.0
		mi.set_initial_value(y, 0)
		T = m.end#0.5
		dt = T/m.steps
		u = [];	t = []; du = []
		while mi.successful() and mi.t <= T:
			#print(m.t)
			m.old_t = m.t
			m.t = int(1+mi.t*(1/dt))
			if m.t < m.steps:
				m.du.append(m.angle_deriv(t, y))
			else:
				m.t = m.old_t

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

		return m.a, m.du	

# Initalization parameters
t_start = 0
t_end = 2.0#0.5#2.0
t_steps = int(t_end*400)#200#800
x = np.linspace(t_start,t_end,t_steps) # time points

V = -25.001#-10.001 # inital voltage
W = 22.9#25.0;#10; # target voltage
Z = 0.028; # tau, speed limiting factor
reset_V = -25 # refractory period reset voltage
lag_time = 0.0
sn = single_neuron(V, W, Z, x, reset_V, lag_time)
neuron_1_V = sn.neuron_out()

V = -25.001
W = 25.2
Z = 0.0355
reset_V = -20.0
lag_time = 23.0
sn_2 = single_neuron(V, W, Z, x, reset_V, lag_time)
neuron_2_V = sn_2.neuron_out()

sections = 'vb', 'db'
neurons_V = neuron_1_V, neuron_2_V
time_params = t_start, t_end, t_steps
find_angle = movement_angle(sections, neurons_V, time_params, x)
a_1, du = find_angle.angle_out()
du_len = len(du)
#print(a_1)
print(len(a_1))
print(du_len)

plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(x, a_1[0:t_steps])
plt.grid(True, color='0.9', linestyle='-', which='both', axis='both')
plt.title(r"Bending Angle ${\Theta}$(deg)")

plt.subplot(3, 1, 2)
plt.plot(np.linspace(t_start,t_end,du_len), du)
plt.grid(True, color='0.9', linestyle='-', which='both', axis='both')
plt.title(r"Angle Derivative d${\Theta}$/dt (deg/s)")

plt.subplot(3, 1, 3)
plt.plot(x, neuron_1_V[:,0], x, neuron_2_V[:,0])
plt.grid(True, color='0.9', linestyle='-', which='both', axis='both')
plt.title('Neuron Voltage (mV)')
plt.ylim(-50,100)
plt.show()

# output example angle recording
example_angle_out = []
recording_length = 21
for i in range(recording_length):
	example_angle_out.extend(65*[-20.0])
	up_trans = np.linspace(-20.0,20.0,12)
	example_angle_out.extend(up_trans.tolist())	
	example_angle_out.extend(70*[20.0])
	down_trans = np.linspace(20.0,-20.0,7)
	example_angle_out.extend(down_trans.tolist())

print("length")
print(len(example_angle_out))
#print(example_angle_out)

time_range = range(len(example_angle_out))
out_file = open('examp_angle.dat', 'w')
for i in time_range:
	#line = str(time_range[i])+"\t"+str(example_angle_out[i])+"\r\n"
	line = str(example_angle_out[i]) + " , "
	out_file.write(line)
print('example output written to file')

'''
plt.subplot(3, 1, 1)
plt.plot(time_range, example_angle_out)
plt.grid(True, color='0.9', linestyle='-', which='both', axis='both')
plt.title(r"Bending Angle ${\Theta}$(deg)")
plt.ylim(-22,22)
plt.show()
'''