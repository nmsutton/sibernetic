/*******************************************************************************
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
 *******************************************************************************/

#include "owMuscleNeuro.h"

owMuscleNeuro::owMuscleNeuro() {

}

double string_to_double( const std::string& s )
/*
 * http://stackoverflow.com/questions/392981/how-can-i-convert-string-to-double-in-c
 */
 {
   std::istringstream i(s);
   double x;
   if (!(i >> x))
     return 0;
   return x;
 }

vector<vector<double> > owMuscleNeuro::owImportNeuro(string in_filename) {

	vector<vector<double> > neuro_signals;
	vector<double> new_signals_entry;
	string signal;

	cout<<"input file: "<<in_filename<<endl;

	std::string x = "/CompNeuro/Software/openworm/CElegansNeuroML/CElegans/pythonScripts/c302/examples/c302_A_Pharyngeal.dat";
	char *y = new char[x.length() + 1]; // or
	// char y[100];//

	std::strcpy(y, x.c_str());
	delete[] y;

	ifstream inFile(y);
	if (!inFile) {
		cerr << "File "<<in_filename<<" not found." << endl;
		exit (EXIT_FAILURE);
	}

	string line;
	while (getline(inFile, line)) {
		if (line.empty()) continue;
		stringstream signals_line(line);

		new_signals_entry.clear();
		while (signals_line >> signal) {new_signals_entry.push_back((double) string_to_double(signal));}

		neuro_signals.push_back(new_signals_entry);
	}

	return neuro_signals;
}
