/*
 * owMuscleNeuro.h
 *
 *  Created on: May 27, 2016
 *      Author: nmsutton
 */

#include <sstream>
#include <string>
#include <cstring>		/* strcpy */
#include <fstream>
#include <iostream>
#include <vector>
#include <stdlib.h>     /* exit, EXIT_FAILURE */

using namespace std;

#ifndef OWMUSCLENEURO_H_
#define OWMUSCLENEURO_H_

class owMuscleNeuro
{
public:
	//void owMuscleNeuro();
	owMuscleNeuro();
	vector<vector<double> > owImportNeuro(string in_filename);
};

#endif /* OWMUSCLENEURO_H_ */
