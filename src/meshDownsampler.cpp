/*
 * main.cpp
 *
 *  Created on: Apr 6, 2016
 *      Author: nmsutton
 */

#include <stdio.h>
#include <iostream>

using namespace std;

struct original_mesh {
	double x[12]; // = {1,2,3};
	double y[12]; // = {4,5,6};
	double z[12]; // = {3,2,1};
};

struct downsampled_mesh {
	double x[4]; // = {1,1};
	double y[4]; // = {1,1};
	double z[4]; // = {1,1};
};

original_mesh orig_mesh;
downsampled_mesh downs_mesh;

const int DOWNS_MESH_VERTS = sizeof(downs_mesh.x)/sizeof(downs_mesh.x[0]);  // from http://stackoverflow.com/questions/2037736/finding-size-of-int-array

void downsample_mesh() {
	/*
	Use self organizing maps to cluster downsampled verticies

	from: https://en.wikipedia.org/wiki/Self-organizing_map

	s is the current iteration
	L is the iteration limit
	t is the index of the target input data vector in the input data set \mathbf{D}
	D(t) is a target input data vector
	v is the index of the node in the map
	W_v is the current weight vector of node v
	u is the index of the best matching unit (BMU) in the map
	Θ(u, v, s) is a restraint due to distance from BMU, usually called the neighborhood function, and
	α (s) is a learning restraint due to iteration progress.

	Wv(s + 1) = Wv(s) + Θ(u, v, s) α(s)(D(t) - Wv(s))
	 */

	double W[DOWNS_MESH_VERTS];
	int s = 0, t = 0, v = 0, u = 0, theta = 0, alpha = 0;
	int L = 10;

	for (int i = 0; i < DOWNS_MESH_VERTS; i++) {

	}
}

void create_mesh(int mesh_x, int mesh_y, int mesh_z, string type) {
	/*
	 * Fill in sample coordinates into meshs for testing
	 *
	 * x,y,z can be modified in loop to change resulting verticies
	 */
	int i = 0, x = 0, y = 0, z = 0;

	for (int x_i = 0; x_i < mesh_x; x_i++) {
		for (int y_i = 0; y_i < mesh_y; y_i++) {
			for (int z_i = 0; z_i < mesh_z; z_i++) {
				i = x_i+y_i+z_i;
				if (type == "orig") {
					x = x_i;
					y = y_i;
					z = z_i;
					orig_mesh.x[i]=x; orig_mesh.y[i]=y; orig_mesh.z[i]=z;
				}
				if (type == "downs") {
					x = 1;
					y = 1;
					z = 1;
					downs_mesh.x[i]=x; downs_mesh.y[i]=y; downs_mesh.z[i]=z;
				}
			}
		}
	}
}

int main() {
	/*
	 * Create downsampling
	 */
	create_mesh(3,2,2,"orig");
	create_mesh(2,2,2,"downs");

	downsample_mesh();

	cout<<"downsampled mesh coordinates:"<<endl;
	for (int i = 0; i < DOWNS_MESH_VERTS; i++) {
		cout<<downs_mesh.x[i]<<"\t"<<downs_mesh.y[i]<<"\t"<<downs_mesh.z[i]<<endl;
	}

	cout<<endl<<"finished"<<endl;
	return 0;
}
