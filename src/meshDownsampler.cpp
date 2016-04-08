/*
 * main.cpp
 *
 *  Created on: Apr 6, 2016
 *      Author: nmsutton
 */

#include <stdio.h>
#include <iostream>
#include <math.h>       /* sqrt */

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

const int ORIG_MESH_VERTS = sizeof(orig_mesh.x)/sizeof(orig_mesh.x[0]);  // from http://stackoverflow.com/questions/2037736/finding-size-of-int-array
const int DOWNS_MESH_VERTS = sizeof(downs_mesh.x)/sizeof(downs_mesh.x[0]);

double find_euclidean_dist(double x_1, double y_1, double z_1, double x_2, double y_2, double z_2) {
	/*
	 * https://en.wikipedia.org/wiki/Euclidean_distance
	 */
	double distance = sqrt(pow((x_2-x_1),2)+pow((y_2-y_1),2)+pow((z_2-z_1),2));
	return distance;
}

double weight_update_formula(double orig_coord, double downs_weight, double bmu_dist, double alpha) {
	/*
	 * SOM formula:
	 * Wv(s + 1) = Wv(s) + Θ(u, v, s) α(s)(D(t) - Wv(s))
	 */
	double weight = downs_weight + bmu_dist * alpha * (orig_coord - downs_weight);
	return weight;
}

void downsample_mesh() {
	/*
	Use self organizing maps to cluster downsampled verticies

	from: https://en.wikipedia.org/wiki/Self-organizing_map

	s = the current iteration, a.k.a. the 'step' of the computations
	L = the iteration limit
	t = the index of the target input data vector in the input data set D
	IV = number of input vector verticies
	T = number of verticies in input vector
	D(t) = a target input data vector
	v = the index of the node in the map (map being the vectors to map the input vector onto)
	W_v = the current weight vector of node v
	u = the index of the best matching unit (BMU) in the map for D(t) (the input vector)
	bmu_dist = Θ(u, v, s) = a restraint due to distance from BMU, usually called the neighborhood function
	the neighborhood function gives the distance between the neuron u and the neuron v in step s
	α(s) = a learning restraint due to iteration progress.

	SOM formula:
	Wv(s + 1) = Wv(s) + Θ(u, v, s) α(s)(D(t) - Wv(s))

	TODO: Stop running algorithm once it reaches a convergence level instead of just the maximum L

	W is initialized as 1s for now, if wanted later work could address "The weights of the neurons
	are initialized either to small random values or sampled evenly from the subspace spanned by
	the two largest principal component eigenvectors."

	first BMU is found for each vertex
	second bmu_dist is used to update SOM weights
	 */

	struct W {
		double x[DOWNS_MESH_VERTS];
		double y[DOWNS_MESH_VERTS];
		double z[DOWNS_MESH_VERTS];
	};
	int s = 0, t = 0, v = 0, u = 0;
	int L = 10;
	int IV = ORIG_MESH_VERTS;
	int T = DOWNS_MESH_VERTS;
	double alpha = 1.0;
	double new_dist = 0;
	double bmu_dist = 0;

	W W;

	for (int map_i = 0; map_i < T; map_i++) {W.x[map_i] = 1; W.y[map_i] = 1; W.z[map_i] = 1;}

	for (int L_i = 0; L_i < L; L_i++) {
		for (int in_i = 0; in_i < IV; IV++) {
			u = 0;
			new_dist = 0;
			bmu_dist = find_euclidean_dist(orig_mesh.x[0], orig_mesh.y[0], orig_mesh.z[0], downs_mesh.x[0], downs_mesh.y[0], downs_mesh.z[0]); // reinitialize

			for (int map_i = 0; map_i < T; map_i++) {
				new_dist = find_euclidean_dist(orig_mesh.x[in_i], orig_mesh.y[in_i], orig_mesh.z[in_i], downs_mesh.x[map_i], downs_mesh.y[map_i], downs_mesh.z[map_i]);
				if (new_dist < bmu_dist) {bmu_dist = new_dist; u = in_i;}

				W.x[map_i] = weight_update_formula(W.x[map_i], orig_mesh.x[in_i], bmu_dist, alpha);
				W.y[map_i] = weight_update_formula(W.y[map_i], orig_mesh.y[in_i], bmu_dist, alpha);
				W.z[map_i] = weight_update_formula(W.z[map_i], orig_mesh.z[in_i], bmu_dist, alpha);
			}
		}
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
