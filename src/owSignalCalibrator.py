from pylab import *;

# Basic parameters.
W=10;
Z=0.02;
# Initial time.
t0 = 0;
# End time.
t1 = 0.5;
# Time vector with 200 points.
npts=200;
t=linspace(t0,t1,npts);

# General analytical solution.
V = -10.001*exp(-t/Z)+W;


import numpy as np
import scipy.integrate as de
import matplotlib.pyplot as plt

W=10;
Z=0.02;
V2 = -10.001
x = np.linspace(0,0.5,200) # time points

class single_neuron():
	def __init__(neuron, V, W, Z, lag_time):
		neuron.V = V		
		neuron.W = W
		neuron.Z = Z
		neuron.x = np.linspace(0,0.5,200)
		neuron.refrac_delay = 120#0.1
		neuron.refrac_current = 0.0
		neuron.refrac_toggle = 0
		neuron.refrac_trigger = 9.9
		neuron.reset_voltage = -10.0
		neuron.lag_time = lag_time
		neuron.lag_counter = 0.0

	def print_i(neuron):
		print(neuron.i)

	def iter_i(neuron):
		neuron.i += 1

	def g(neuron, V, x):
		V = neuron.refrac_test(V)
		#V = neuron.V
		#print(V)

		if (neuron.lag_finished()):
			V = (-V+neuron.W)/neuron.Z

		neuron.V = V

		return V

	def lag_finished(neuron):
		neuron.lag_counter += 1

		lag_over = False

		if neuron.lag_counter > neuron.lag_time:
			lag_over = True

		#print(neuron.lag_counter)
		#print(neuron.lag_time)

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
		#print(V)
		#refractory point
		if V >= refrac_trigger:
			V = reset_voltage

			refrac_toggle = 1

		if refrac_toggle == 1:
			W = 0
			refrac_current += 1#0.1/200.0
			#print(refrac_current)

			if refrac_current >= refrac_delay: 
				W = 10 	
				refrac_current = 0.0
				refrac_toggle = 0  		

		neuron.W, neuron.Z, neuron.refrac_delay, \
		neuron.refrac_current, neuron.refrac_toggle, \
		neuron.refrac_trigger, neuron.reset_voltage = \
		W, Z, refrac_delay, refrac_current, \
		refrac_toggle, refrac_trigger, reset_voltage

		return V		

'''
sn = single_neuron()
sn.print_i()
sn.iter_i()
sn.iter_i()
sn.print_i()
'''

# Initial conditions on y, y' at x=0
#init = V2#W, Z, V2, refrac_delay, refrac_current, refrac_toggle
# First integrate from 0 to 2

#sol=de.odeint(g, init, x)
#plt.plot(x, sol[:,0], t, V)
lag_time = 0.0
sn = single_neuron(V2, W, Z, lag_time)
no = sn.neuron_out()

lag_time = 23.0
sn_2 = single_neuron(V2, W, Z, lag_time)
no_2 = sn_2.neuron_out()

#print(no)
#print(len(no))
#print(len(x))
#plt.plot(x, no)
#plt.plot(x, no, t, V)
#plt.plot(x, no[:,0], t, V)
#plt.plot(x, no[:,0])
plt.plot(x, no[:,0], x, no_2[:,0])
plt.show()

'''
# Preallocation.
Vn = zeros(t.shape);
# Initial conditions.
Vn[0] = -0.001; 
# Now do the explicit Euler integration.
for ij in range(1,len(t)):
	# Delta t.
	dt = t[ij]-t[ij-1];
	# Slope.
	dV_dt = (-Vn[ij-1]+W)/Z;
	Vn[ij]=Vn[ij-1] + dV_dt*dt;

# First solution.
#plot(t,V,t,Vn);
plot(t,V);
show();
'''