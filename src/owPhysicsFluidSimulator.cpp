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

#include <stdexcept>
#include <iostream>
#include <fstream>

#include "owSignalSimulator.h"
#include "owPhysicsFluidSimulator.h"
#include "owVtkExport.h"


/** Constructor method for owPhysicsFluidSimulator.
 *
 *  @param helper
 *  pointer to owHelper object with helper function.
 *  @param dev_type
 *  defines preferable device type for current configuration
 */
owPhysicsFluidSimulator::owPhysicsFluidSimulator(owHelper * helper,int argc, char ** argv)
{
	//int generateInitialConfiguration = 1;//1 to generate initial configuration, 0 - load from file

	try{
		iterationCount = 0;
		config = new owConfigProperty(argc, argv);
		// LOAD FROM FILE
		owHelper::preLoadConfiguration(config);
		config->initGridCells();
		position_cpp = new float[ 4 * config->getParticleCount() ];
		velocity_cpp = new float[ 4 * config->getParticleCount() ];
		muscle_activation_signal_cpp = new float [config->MUSCLE_COUNT];
		muscle_groups_oc = new float [320];
		muscle_pids_oc = new int [320];
		if(config->numOfElasticP != 0)
			elasticConnectionsData_cpp = new float[ 4 * config->numOfElasticP * MAX_NEIGHBOR_COUNT ];
		if(config->numOfMembranes<=0)
			membraneData_cpp = NULL;
		else
			membraneData_cpp = new int [config->numOfMembranes*3];
		if(config->numOfElasticP<=0)
			particleMembranesList_cpp = NULL;
		else
			particleMembranesList_cpp = new int [config->numOfElasticP * MAX_MEMBRANES_INCLUDING_SAME_PARTICLE];
		for(unsigned int i=0;i<config->MUSCLE_COUNT;++i)
		{
			muscle_activation_signal_cpp[i] = 0.f;
		}

		//The buffers listed below are only for usability and debug
		density_cpp = new float[ 1 * config->getParticleCount() ];
		particleIndex_cpp = new unsigned int[config->getParticleCount() * 2];

		// LOAD FROM FILE
		owHelper::loadConfiguration( position_cpp, velocity_cpp, elasticConnectionsData_cpp, membraneData_cpp, particleMembranesList_cpp, config );		//Load configuration from file to buffer
		this->helper = helper;
		if(config->numOfElasticP != 0){
			ocl_solver = new owOpenCLSolver(position_cpp, velocity_cpp, config, elasticConnectionsData_cpp, membraneData_cpp, particleMembranesList_cpp);	//Create new openCLsolver instance
		}else
			ocl_solver = new owOpenCLSolver(position_cpp,velocity_cpp, config);	//Create new openCLsolver instance
	}catch(std::runtime_error &ex){
		/* Clearing all allocated buffers and created object only not ocl_solver
		 * case it wont be created yet only if exception is throwing from its constructor
		 * but in this case ocl_solver wont be created
		 * */
		destroy();
		delete config;
		throw ex;
	}
}
/** Reset simulation
 *
 *  Restart simulation with new or current simulation configuration.
 *  It redefines all required data buffers and restart owOpenCLSolver
 *  by run owOpenCLSolver::reset(...).
 */
void owPhysicsFluidSimulator::reset(){
	// Free all buffers
	destroy();
	config->resetNeuronSimulation();
	iterationCount = 0;
	config->numOfBoundaryP = 0;
	config->numOfElasticP = 0;
	config->numOfLiquidP = 0;
	config->numOfMembranes = 0;
	// LOAD FROM FILE
	owHelper::preLoadConfiguration(config);
	config->initGridCells();
	position_cpp = new float[ 4 * config->getParticleCount() ];
	velocity_cpp = new float[ 4 * config->getParticleCount() ];
	muscle_activation_signal_cpp = new float [config->MUSCLE_COUNT];
	if(config->numOfElasticP != 0) elasticConnectionsData_cpp = new float[ 4 * config->numOfElasticP * MAX_NEIGHBOR_COUNT ];
	if(config->numOfMembranes<=0) membraneData_cpp = NULL; else membraneData_cpp = new int [ config->numOfMembranes * 3 ];
	if(config->numOfElasticP<=0)  particleMembranesList_cpp = NULL; else particleMembranesList_cpp = new int [config->numOfElasticP*MAX_MEMBRANES_INCLUDING_SAME_PARTICLE];
	for(unsigned int i=0;i<config->MUSCLE_COUNT;++i){
		muscle_activation_signal_cpp[i] = 0.f;
	}
	//The buffers listed below are only for usability and debug
	density_cpp = new float[ 1 * config->getParticleCount() ];
	particleIndex_cpp = new unsigned int[config->getParticleCount() * 2];
	// LOAD FROM FILE
	owHelper::loadConfiguration( position_cpp, velocity_cpp, elasticConnectionsData_cpp, membraneData_cpp, particleMembranesList_cpp, config );		//Load configuration from file to buffer
	if(config->numOfElasticP != 0){
		ocl_solver->reset(position_cpp, velocity_cpp, config, elasticConnectionsData_cpp, membraneData_cpp, particleMembranesList_cpp);	//Create new openCLsolver instance
	}else
		ocl_solver->reset(position_cpp,velocity_cpp, config);	//Create new openCLsolver instance
}
/** Run one simulation step
 *
 *  Run simulation step in pipeline manner.
 *  It starts with neighbor search algorithm than
 *  physic simulation algorithms: PCI SPH [1],
 *  elastic matter simulation, boundary handling [2],
 *  membranes handling and finally numerical integration.
 *  [1] http://www.ifi.uzh.ch/vmml/publications/pcisph/pcisph.pdf
 *  [2] M. Ihmsen, N. Akinci, M. Gissler, M. Teschner,
 *      Boundary Handling and Adaptive Time-stepping for PCISPH
 *      Proc. VRIPHYS, Copenhagen, Denmark, pp. 79-88, Nov 11-12, 2010
 *
 *  @param looad_to
 *  If it's true than Sibernetic works "load simulation data in file" mode.
 */
double owPhysicsFluidSimulator::simulationStep(const bool load_to)
{
	int iter = 0;//PCISPH prediction-correction iterations counter
                 //
	// now we will implement sensory system of the c. elegans worm, mechanosensory one
	// here we plan to implement the part of openworm sensory system, which is still
	// one of the grand challenges of this project

	//if(iterationCount==0) return 0.0;//uncomment this line to stop movement of the scene

	helper->refreshTime();
	std::cout << "\n[[ Step "<< iterationCount << " ]]\n";
	//SEARCH FOR NEIGHBOURS PART
	//ocl_solver->_runClearBuffers();								helper->watch_report("_runClearBuffers: \t%9.3f ms\n");
	ocl_solver->_runHashParticles(config);							helper->watch_report("_runHashParticles: \t%9.3f ms\n");
	ocl_solver->_runSort(config);									helper->watch_report("_runSort: \t\t%9.3f ms\n");
	ocl_solver->_runSortPostPass(config);							helper->watch_report("_runSortPostPass: \t%9.3f ms\n");
	ocl_solver->_runIndexx(config);									helper->watch_report("_runIndexx: \t\t%9.3f ms\n");
	ocl_solver->_runIndexPostPass(config);							helper->watch_report("_runIndexPostPass: \t%9.3f ms\n");
	ocl_solver->_runFindNeighbors(config);							helper->watch_report("_runFindNeighbors: \t%9.3f ms\n");
	//PCISPH PART
	if(config->getIntegrationMethod() == LEAPFROG){ // in this case we should remmember value of position on stem i - 1
		//Calc next time (t+dt) positions x(t+dt)
		ocl_solver->_run_pcisph_integrate(iterationCount,0/*=positions_mode*/, config);
	}
	ocl_solver->_run_pcisph_computeDensity(config);
	ocl_solver->_run_pcisph_computeForcesAndInitPressure(config);
	ocl_solver->_run_pcisph_computeElasticForces(config);
	do{
		//printf("\n^^^^ iter %d ^^^^\n",iter);
		ocl_solver->_run_pcisph_predictPositions(config);
		ocl_solver->_run_pcisph_predictDensity(config);
		ocl_solver->_run_pcisph_correctPressure(config);
		ocl_solver->_run_pcisph_computePressureForceAcceleration(config);
		iter++;
	}while( iter < maxIteration );

	//and finally calculate v(t+dt)
	if(config->getIntegrationMethod() == LEAPFROG){
		ocl_solver->_run_pcisph_integrate(iterationCount,1/*=velocities_mode*/, config);		helper->watch_report("_runPCISPH: \t\t%9.3f ms\t3 iteration(s)\n");
	}
	else{
		ocl_solver->_run_pcisph_integrate(iterationCount, 2,config);		helper->watch_report("_runPCISPH: \t\t%9.3f ms\t3 iteration(s)\n");
	}
	//Handling of Interaction with membranes
	if(config->numOfMembranes > 0){
		ocl_solver->_run_clearMembraneBuffers(config);
		ocl_solver->_run_computeInteractionWithMembranes(config);
		// compute change of coordinates due to interactions with membranes
		ocl_solver->_run_computeInteractionWithMembranes_finalize(config);
																	helper->watch_report("membraneHadling: \t%9.3f ms\n");
	}
	//END
	ocl_solver->read_position_buffer(position_cpp, config);				helper->watch_report("_readBuffer: \t\t%9.3f ms\n");

	//END PCISPH algorithm
	printf("------------------------------------\n");
	printf("_Total_step_time:\t%9.3f ms\n",helper->getElapsedTime());
	printf("------------------------------------\n");
	if(load_to){
		if(iterationCount == 0){
			owHelper::loadConfigurationToFile(position_cpp,  config,elasticConnectionsData_cpp,membraneData_cpp,true);
		}else{
			if(iterationCount % config->getLogStep() == 0){
				owHelper::loadConfigurationToFile(position_cpp, config, NULL, NULL, false);
			}
		}
	}
	if(owVtkExport::isActive){
		if(iterationCount % config->getLogStep() == 0){
			getvelocity_cpp();
			owVtkExport::exportState(iterationCount, config, position_cpp,
									 elasticConnectionsData_cpp, velocity_cpp,
									 membraneData_cpp, muscle_activation_signal_cpp);
		}
	}

	float correction_coeff;
	float neural_signal;

	int m_counter;
	int ptype_i = 0, ptype_i_2 = 0;
	bool muscle_found = false;
	float pos_x = 0, pos_y = 0, pos_z = 0;
	if (iterationCount==0) {m_counter = 0;}

	std::vector<int> muscle_collection;
	/* determine id and group of each muscle particle */
	for (int i = 0; i < config->numOfElasticP;i=i+1) {
		ptype_i = (i*MAX_NEIGHBOR_COUNT*4)+2; // 3 is interval where ptype index is stored
		muscle_found = false;
		for (int j = 0; j < MAX_NEIGHBOR_COUNT; j++) {
			ptype_i_2 =  ptype_i + (j*4); // increase by 4 is for next neighbor entry
			if (i == 150 or i == 149 or i == 151 or i == 561 or i == 559 or i == 560) {
				elasticConnectionsData_cpp[ptype_i_2] = 1.1;
			}
			if (elasticConnectionsData_cpp[ptype_i_2] >= 1 && elasticConnectionsData_cpp[ptype_i_2] < 2) {

				if (iterationCount==0) {
					pos_x = position_cpp[(i*4)+0];
					pos_y = position_cpp[(i*4)+1];
					pos_z = position_cpp[(i*4)+2];
					//if (pos_y <= 69.5) {
					//if (pos_x <= 73) {
					if ((pos_x <= 73 & pos_z <= 71.9) or \
						(pos_x >= 73 & pos_z >= 71.9)) {
					//if (pos_x <= 65.5) {
					//if (pos_z <= 71.9) {					
						elasticConnectionsData_cpp[ptype_i_2] = 1.5;
						//m_counter++;
					}
					else {
						elasticConnectionsData_cpp[ptype_i_2] = 1.1;
					}
				}

				muscle_found = true;
			}
			//std::cout<<elasticConnectionsData_cpp[ptype_i_2]<<" ";
		}
		if (muscle_found == true) {
			muscle_collection.push_back(i);
			//m_counter++;
		}
	}

	if (iterationCount==0) {
		ocl_solver->updateElasticConnectionsData(elasticConnectionsData_cpp, config);
	}

	int muscles_size = muscle_collection.size();
	//	std::vector<std::vector<int> > muscle_groups[4][muscles_size];
	//std::vector<std::vector<int> > muscle_groups(muscles_size, std::vector<int>(muscles_size/4));
	/*std::vector<float> muscle_groups(muscles_size);
	std::vector<int> muscle_pids(muscles_size);*/
	/*std::cout<<"mc \n";
	std::cout<<"\nsize "<<muscles_size<<"\n";
	std::cout<<"\n"<<muscle_groups[1].size()<<" ";*/
	int muscle_pid = 0;
	if (iterationCount==0) {
		//m_counter = 0;	
		for (int i = 0; i < muscles_size; i++) {
			muscle_pid = muscle_collection[i];
			neural_signal = config->get_example_angle_out(iterationCount);
			pos_x = position_cpp[(muscle_pid*4)+0];
			pos_y = position_cpp[(muscle_pid*4)+1];
			pos_z = position_cpp[(muscle_pid*4)+2];

			//if (pos_y <= 69.5) {
			//if (pos_x <= 60.0) {
			//if (pos_x <= 65.5) {
			if (pos_x <= 73.0) {
			//if (pos_z <= 71.9) {
				muscle_groups_oc[i] = 2.0f;
				m_counter++;
			}
			//std::cout<<"|"<<pos_x<<" "<<pos_y<<" "<<pos_z;

			muscle_pids_oc[i] = muscle_pid;		
		}
		muscle_groups_oc[319] = m_counter;
	}

	/*
	Find anchor points
	*/
	muscle_pid = 150;
	float vertical_start[] = {position_cpp[(muscle_pid*4)+0], position_cpp[(muscle_pid*4)+1], position_cpp[(muscle_pid*4)+2]};
	muscle_pid = 561;
	float vertical_end[] = {position_cpp[(muscle_pid*4)+0], position_cpp[(muscle_pid*4)+1], position_cpp[(muscle_pid*4)+2]};
	muscle_pid = 87;
	float horizontal_start[] = {position_cpp[(muscle_pid*4)+0], position_cpp[(muscle_pid*4)+1], position_cpp[(muscle_pid*4)+2]};
	muscle_pid = 85;
	float horizontal_end[] = {position_cpp[(muscle_pid*4)+0], position_cpp[(muscle_pid*4)+1], position_cpp[(muscle_pid*4)+2]};	
	//muscle_pid = center;
	//float center_point[] = {position_cpp[(muscle_pid*4)+0], position_cpp[(muscle_pid*4)+1], position_cpp[(muscle_pid*4)+2]};
	float forward_move[] = {(horizontal_end[0] - horizontal_start[0]), (horizontal_end[1] - horizontal_start[1]), (horizontal_end[2] - horizontal_start[2])};
	//float reverse_move[] = {(horizontal_start[0] - center_point[0]), (horizontal_start[1] - center_point[1]), (horizontal_start[2] - center_point[2])};	
	muscle_groups_oc[315] = forward_move[0];
	muscle_groups_oc[316] = forward_move[1];
	muscle_groups_oc[317] = forward_move[2];
	float move_scaling = 0.05;

	int p_type;
	int m;
	for (int i = 0; i < muscles_size; i++) {
		neural_signal = config->get_example_angle_out(iterationCount);
		p_type = muscle_groups_oc[i];
		m = muscle_collection[i];
		if ((p_type >= 1.8 and p_type <= 2.2) or \
			(p_type >= 2.8 and p_type <= 3.2)) {
			////////// last commented out 11/05/16 //////////
			muscle_activation_signal_cpp[i] = neural_signal*-1;
			/////////////////////////////////////////////////			
			/*
			position_cpp[(m*4)+0] += forward_move[0] * move_scaling * neural_signal * -1; // x
			position_cpp[(m*4)+1] += forward_move[1] * move_scaling * neural_signal * -1; // y
			position_cpp[(m*4)+2] += forward_move[2] * move_scaling * neural_signal * -1; // z
			*/
		}
		else {
			////////// last commented out 11/05/16 //////////
			muscle_activation_signal_cpp[i] = neural_signal;//*-1;	
			/////////////////////////////////////////////////
			/*
			position_cpp[(m*4)+0] += forward_move[0] * move_scaling * neural_signal; // x
			position_cpp[(m*4)+1] += forward_move[1] * move_scaling * neural_signal; // y
			position_cpp[(m*4)+2] += forward_move[2] * move_scaling * neural_signal; // z			
			*/
		}	
	}
	
	/*for(unsigned int i=0;i<=1;++i)//config->MUSCLE_COUNT;++i)//i<=0;++i)//
	{ 
		correction_coeff = sqrt( 1.f - ((1+i%24-12.5f)/12.5f)*((1+i%24-12.5f)/12.5f) );
		//printf("\n%d\t%d\t%f\n",i,1+i%24,correction_coeff);
		neural_signal = config->get_example_angle_out(iterationCount);
		muscle_activation_signal_cpp[1] += neural_signal*.07;//.7;
		//muscle_activation_signal_cpp[0] = -20.0*.07;//neural_signal*.07;//.7;
		muscle_activation_signal_cpp[i] *= correction_coeff;//0.39191836   0.5425864		
		//muscle_activation_signal_cpp[i] *= 0.39191836;//   0.5425864		

		//if (i == 0) {
		//	std::cout<<"\nCurrent timestep:\t"<<iterationCount<<"\tp1\t"<<muscle_activation_signal_cpp[2]<<"\tm count\t"<<sizeof(muscle_activation_signal_cpp)/sizeof(float)<<"\t"<<muscle_activation_signal_cpp[i]<<"\t"<<config->get_example_angle_out(iterationCount)<<"\n";
		//	std::cout<<"\nm_counter:"<<m_counter<<"\n";
		//}

	}*/
	//std::cout<<muscle_groups[0].size()<<" ";
	//std::cout<<"\n\nmuscle pids: "<<sizeof(muscle_pids_oc)/sizeof(int())<<" \n\n";
	//std::cout<<"\n\nmuscle pids: "<<muscles_size<<" \n\n";
	std::cout<<"\n\nmuscle counter: "<<m_counter<<" \n\n";
	std::cout<<"\n\nmuscle_groups_oc[319]: "<<muscle_groups_oc[319]<<" \n\n";
	//std::cout<<"\n";
	//uint muscle_number_cpp = 5;
	//owOpenCLSolver::copy_buffer_to_device(muscle_number_cpp, muscle_number, sizeof( uint ));
	config->set_muscle_number(config->get_muscle_number()+1);	

	config->updateNeuronSimulation(muscle_activation_signal_cpp);

	ocl_solver->updateMuscleActivityData(muscle_activation_signal_cpp, config);
	//ocl_solver->updateMuscleGroupsOc((float*) &muscle_groups_oc, config);	
	muscle_groups_oc[318] = iterationCount;//iterationCount;	
	std::cout<<"\nmuscle_groups_oc[318] "<<muscle_groups_oc[318]<<"\n";
	ocl_solver->updateMuscleGroupsOc(muscle_groups_oc, config);	
	ocl_solver->updateMusclePidsOc(muscle_pids_oc, config);	
	/*std::cout<<(float*) &muscle_groups_oc[0]<<" "<<muscle_groups_oc[1]<<" "<<muscle_groups_oc[2]<<" " \
	<<(float*) &muscle_groups_oc[3]<<" "<<(float*) &muscle_groups_oc[4]<<" "<<(float*) &muscle_groups_oc[5]<<" " \
	<<(float*) &muscle_groups_oc[6]<<" "<<(float*) &muscle_groups_oc[7]<<" "<<(float*) &muscle_groups_oc[8]<<" ";
	*/
	/*
	float * muscle_groups_cpp = getMuscleGroups();
	std::cout<<"\n\n"<<muscle_groups_cpp[10]<<"\n\n";
	*/
	//std::cout<<"\n\nnumOfElasticP: "<<config->numOfElasticP<<"\n\n";
	iterationCount++;
	return helper->getElapsedTime();
}
/** Prepare data and log it to special configuration
 *  file you can run your simulation from place you snapshoted it
 *
 *  @param fileName - name of file where saved configuration will be stored
 */
void owPhysicsFluidSimulator::makeSnapshot(){
	getvelocity_cpp();
	std::string fileName = config->getSnapshotFileName();
	owHelper::loadConfigurationToFile(position_cpp, velocity_cpp, elasticConnectionsData_cpp, membraneData_cpp, particleMembranesList_cpp, fileName.c_str(), config);
}

//Destructor
owPhysicsFluidSimulator::~owPhysicsFluidSimulator(void)
{
	destroy();
	delete config;
	delete ocl_solver;
}

void owPhysicsFluidSimulator::destroy(){
	delete [] position_cpp;
	delete [] velocity_cpp;
	delete [] density_cpp;
	delete [] particleIndex_cpp;
	if(elasticConnectionsData_cpp != NULL)
		delete [] elasticConnectionsData_cpp;
	delete [] muscle_activation_signal_cpp;
	if(membraneData_cpp != NULL) {
		delete [] membraneData_cpp;
		delete [] particleMembranesList_cpp;
	}
}
