import numpy as np
import scipy as sc
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import numba
import ast
from numba import njit, prange
import argparse
from pathlib import Path
import pandas as pd
import json
import h5py
import time

###
#Numba Section
###

@njit
def numba_col_sum(array,mu_index):
    total = 0.0
    for i in range(array.shape[0]):
        total+=array[i,mu_index]
    return total

@njit
def delta(i,j):
    return 1 if i==j else 0
"""Just a standard dirac delta function to be used in most of the update rules"""

@njit
def update_phi(signals_meaning_array, alpha_val, alpha_on_s,signal_index,num_meanings):
    """Update rule for phi is based upon the number of observed counts, i.e., equation 12 in the notes.
     As I understand the equation, the updates should only take place for the row corresponding to the signal observed!
     So if the possible signals are dog and cat, and we observe signal cat, we do not touch the phi array values corresponding
     to signal dog.

     The signals are updated according to the following:
     phi_i(s|m;t) = [(n_i(s|m;t)+alpha/S)]/(sum(n_i(s'|m;t))+alpha)
     where sum(n_i(s'|m;t)) corresponds to the sum over signals for that possible meaning, or in othern words the sum of the column

     
     Calculates the updated phi row, and then returns this to be sliced back into the array
       """
    new_phi_row = np.empty(num_meanings)
    signals_meanings_cut = signals_meaning_array[signal_index,:]
    for m in range(num_meanings):
        new_phi_row[m] = (signals_meanings_cut[m] +alpha_on_s)/(signals_meaning_array[:,m].sum() +alpha_val)

    return new_phi_row


@njit(parallel = True)
def update_signal_meaning(counts_array, lambda_val, meaning_index, signal_index, num_signals):
    """Update rule according to equation 14 in document Evolution of Communications Through Fluctuations"""
    for i in prange(num_signals):
        counts_array[i,meaning_index] = delta(i=signal_index, j=i) + (1-lambda_val)*counts_array[i,meaning_index]
    return counts_array
    
    
@njit(parallel = True)
def create_phi_tensor(num_agents, agents_list,num_signals,num_meanings):
    """Compiles all the phi arrays into a singular tensor, to make calculating gain values easier"""
    phi_tensor = np.empty((num_agents,num_signals,num_meanings))
    for i in range(num_agents):
        for j in prange(num_signals):
            for k in prange(num_meanings):
                phi_tensor[i,j,k] = agents_list[i].phi_array[j,k]

@njit
def blind_success_inner_loop(agenti_phi,agentj_phi,num_signals,num_meanings):
    """I am defining the inner loop to be the sum over all meanings and signals of the selected agent's
    phi matrices.  I have decided to break up the blind_success calculations into two parts as I believe it 
    will allow for easier modification of the formula, as well as an easier way to think about the process
    actually being done."""
    inst_sum = 0.0
    for a in range(num_meanings):
        signal_sum = 0.0
        for b in range(num_signals):
            denom_term = agentj_phi[b,:].sum()
            signal_sum+=(agenti_phi[b,a]*agentj_phi[b,a])/denom_term
        inst_sum+=(1/num_meanings)*signal_sum
    return inst_sum

@njit
def blind_success_outer_loop(num_agents,ensemble_phi,num_signals,num_meanings):
    """The outer loop of the blind success metric calculation, summing over all interacting agent pairs"""
    p_s =0.0
    const = 1/(num_agents*(num_agents-1))
    for i in range(num_agents):
        for j in range(num_agents):
            if i==j:
                continue
            else:
                agenti_phi = ensemble_phi[i,:]
                agentj_phi = ensemble_phi[j,:]
                p_s += blind_success_inner_loop(agenti_phi,agentj_phi,num_signals,num_meanings)
    p_s*=const
    return p_s


@njit
def numba_random_choice(array,prob):
    """Numba does not accept the probability distribution for the np.random.choice() function, as such we need a workaround to be able to handle 
    this efficiently.  This function was taken from the numba support issue 2539, specifically from commentor Mike Fenton
    
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.

    
    """
    return array[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


############################################################################################################################
@njit
def select_signal_njit(alpha_dist, signals,meanings, phi_array, meanings_list_forIDX):
    """A numba function for fast meaning selection and signal emission once provided the rho (attentional weight dirichlet) distribution.
    The attentional weight distribution is calculated using sci-py, which is not supported in numba, thus we have no way to entirely numba-fy
    this process unless we decide to make a numba calculator of the dirichlet distribution. 
    """
    rho_dist = np.random.dirichlet(alpha=alpha_dist)
    selected_mu = numba_random_choice(meanings,rho_dist)
    mu_idx = meanings_list_forIDX.index(selected_mu)
    signal_prob = phi_array[:,mu_idx]
    selected_signal= numba_random_choice(signals,signal_prob)
    return selected_signal, rho_dist


@njit
def select_signal_integers_njit(alpha_dist, phi_array,meanings_integers,signals_integers):
    """A numba function for fast meaning selection and signal emission once provided the rho (attentional weight dirichlet) distribution.
    The attentional weight distribution is calculated using sci-py, which is not supported in numba, thus we have no way to entirely numba-fy
    this process unless we decide to make a numba calculator of the dirichlet distribution. 
    """
    rho_dist = np.random.dirichlet(alpha=alpha_dist)
    selected_mu = numba_random_choice(meanings_integers,rho_dist)
    signal_prob = phi_array[:,selected_mu]
    selected_signal_idx= numba_random_choice(signals_integers,signal_prob)
    return selected_signal_idx, rho_dist

@njit
def receive_signal_integers_njit(signal_idx,passed_rho_dist,phi_array,meanings_integers,num_meanings):
    """A numba version of the signal receiving and interpretation process, confined to one generation"""

    rho_dist = passed_rho_dist
    phi_s_mu = phi_array[int(signal_idx),:]
    # print("phi_s_mu")
    # print(phi_s_mu)
    denom = (phi_s_mu*rho_dist).sum()
    # print("Denominator")
    # print(denom)
    posterior_dist = np.zeros(num_meanings, dtype=np.float64)
    for j in range(num_meanings):
        # inst_val = (phi_s_mu[j]*rho_dist[j])/denom
        # print("inst_val")
        # print(inst_val)
        posterior_dist[j] = ((phi_s_mu[j]*rho_dist[j])/denom)

    # print("Posterior Chain")
    # print(posterior_dist)
    nu_idx = numba_random_choice(meanings_integers,posterior_dist)
    return nu_idx

@njit
def receive_signal_intergers_intergen_njit(signal_idx,passed_alpha_dist,phi_array,meanings_integers,num_meanings):
    """Numba function to model the dynamics of signal interpretation between two generations"""
    rho_dist = np.random.dirichlet(alpha=passed_alpha_dist)
    phi_s_mu = phi_array[int(signal_idx),:]
    # print("phi_s_mu")
    # print(phi_s_mu)
    denom = (phi_s_mu*rho_dist).sum()
    # print("Denominator")
    # print(denom)
    posterior_dist = np.zeros(num_meanings, dtype=np.float64)
    for j in range(num_meanings):
        # inst_val = (phi_s_mu[j]*rho_dist[j])/denom
        # print("inst_val")
        # print(inst_val)
        posterior_dist[j] = ((phi_s_mu[j]*rho_dist[j])/denom)

    # print("Posterior Chain")
    # print(posterior_dist)
    nu_idx = numba_random_choice(meanings_integers,posterior_dist)
    return nu_idx


@njit
def random_choice_test(upper,amount):
    for _ in range(amount):
        print(np.random.choice(upper,size = 2))


"""The way I am handling the alpha tensor would make it difficult to make an optimal function for both intergen, 
and one gen cases, solution is to make 2 different functions"""
@njit(nopython = True)
def one_interaction_multiple_gens(counts,phi,speaker_info, meaning_integers, signal_integers,num_agents, 
                                  alpha_tensor,M, S, lambda_val,alpha_val):
    selected_agents = np.random.choice(num_agents,size =2)
    speaker = int(selected_agents[0])
    receiver = int(selected_agents[1])

    speaker_gen_tag = int(speaker_info[speaker,0])
    receiver_gen_tag = int(speaker_info[receiver,0])
    speaker_alpha = alpha_tensor[speaker_gen_tag,:]
    speaker_phi = phi[speaker,:,:]
    receiver_phi = phi[receiver,:,:]

    receiver_counts = counts[receiver,:,:]

    selected_signal_idx, rho_dist = select_signal_integers_njit(speaker_alpha,speaker_phi,meaning_integers,signal_integers)

    if speaker_gen_tag==receiver_gen_tag:
        """Communication between two speakers of same gen, rho is passed"""
        nu_idx = receive_signal_integers_njit(selected_signal_idx,rho_dist,receiver_phi,meaning_integers,M)
    else:
        receiver_alpha = alpha_tensor[receiver_gen_tag,:]
        """Communication between two different gens, they each just use their own alpha"""
        receiver_rho = np.random.dirichlet(alpha=receiver_alpha)
        nu_idx = receive_signal_integers_njit(selected_signal_idx,receiver_rho,receiver_phi,meaning_integers,M)

    """Updating the corresponding arrays"""
    counts[receiver,:,:] = update_signal_meaning(receiver_counts,lambda_val,nu_idx,selected_signal_idx,S)

    new_phi_row = update_phi(counts[receiver,:,:],alpha_val,alpha_val/S,selected_signal_idx,M)
    phi[receiver,selected_signal_idx,:] = new_phi_row
    return phi,counts

@njit
def one_collection_loop_multiple_gens(counts,phi,speaker_info, meaning_integers, signal_integers,num_agents, 
                                  alpha_tensor,M, S, lambda_val,alpha_val,iterations):
    """Runs the data collection/communication process for a specified amount of iterations, then returns the counts and phi arrays"""
    for _ in range(iterations):
        new_phi, new_counts = one_interaction_multiple_gens(counts,phi,speaker_info, meaning_integers, signal_integers,num_agents, 
                                  alpha_tensor,M, S, lambda_val,alpha_val)
        counts = new_counts
        phi = new_phi

    return phi, counts


"""Now doing the single generation interaction processes"""
@njit
def one_interaction_one_population(counts,phi, meaning_integers, signal_integers,num_agents, 
                                  alpha_vector,M, S, lambda_val,alpha_val, alignment):
    """Interaction rules for an ensemble of speakers all from one population"""
    
    selected_agents = np.random.choice(num_agents,size =2)
    speaker = int(selected_agents[0])
    receiver = int(selected_agents[1])
    speaker_phi = phi[speaker,:,:]
    receiver_phi = phi[receiver,:,:]

    receiver_counts = counts[receiver,:,:]

    selected_signal_idx, rho_dist = select_signal_integers_njit(alpha_vector,speaker_phi,meaning_integers,signal_integers)
    #Above taken nearly directly from multi gen interactions

    #alignment rolls
    alignment_roll = np.random.random()
    if alignment_roll<= alignment:
        """If the roll is less than the specified value then pass the rho distribution, else reroll it"""
        receiver_rho = rho_dist
    else:
        receiver_rho = np.random.dirichlet(alpha = alpha_vector)

    nu_idx = receive_signal_integers_njit(selected_signal_idx,receiver_rho,receiver_phi,meaning_integers,M)
    """Updating the corresponding arrays"""
    counts[receiver,:,:] = update_signal_meaning(receiver_counts,lambda_val,nu_idx,selected_signal_idx,S)

    new_phi_row = update_phi(counts[receiver,:,:],alpha_val,alpha_val/S,selected_signal_idx,M)
    phi[receiver,selected_signal_idx,:] = new_phi_row

    return phi,counts

@njit
def one_collection_loop_one_pop(counts,phi, meaning_integers, signal_integers,num_agents, 
                                  alpha_vector,M, S, lambda_val,alpha_val, alignment,iterations):
    """Runs the data collection/communication process for a specified amount of iterations, then returns the counts and phi arrays"""
    for _ in range(iterations):
        new_phi, new_counts = one_interaction_one_population(counts,phi, meaning_integers, signal_integers,num_agents, 
                                  alpha_vector,M, S, lambda_val,alpha_val, alignment)
        counts = new_counts
        phi = new_phi

    return phi, counts

    




class Ensemble:
    def __init__(self,read_dir,output_dir, iterations, alignment,beta, alpha, num_signals, num_meanings, 
                 lambda_val, time_resolution = 100000, import_agents = True, generation_overlap = False, 
                 export_agents = False, new_gamma = None, passed_focuses = None, non_input_initialisation_agents = None):
        print("Initialising")
        self.param_dir = Path(read_dir)
        self.raw_dir = Path(read_dir)/"raw"
        self.output_dir = Path(output_dir)
        self.export_agents = export_agents


        self.A = alignment
        self.beta = beta
        self.C = 1/(1+beta)
        self.lambda_val = lambda_val
        self.gamma = new_gamma
        self.alpha = alpha
        self.alpha_on_s = alpha/num_signals

        self.num_agents = 0
        self.S= num_signals
        self.M = num_meanings
        self.meanings_integers = np.arange(num_meanings,dtype=int)
        self.signals_integers = np.arange(num_signals,dtype=int)

        self.time_res = time_resolution
        self.tot_iterations = iterations
        self.num_measurements = iterations//time_resolution

        self.new_gen_focuses = passed_focuses
        self.possible_focuses = np.arange(self.M).tolist()
        print(f"Debug, self.possible_focuses is {self.possible_focuses}")
        self.gen_focuses = []

        self.gen_alpha_dists = []
        #List to store the alpha arrays of each generation
        self.avg_map = []
        #Guide to which indices of the tensors correspond to each generation

        if import_agents:
            self.read_param_file()
            print("Read file")
            assert self.imported_meanings== self.M, "Imported meanings and inputted meanings do not match"
            assert self.imported_signals== self.S, "Imported signals and inputted signals do not match"
            self.num_gens = len(self.imported_gen_focuses)-1
            self.create_initial_tensors_from_inputs()
            self.gen_focuses.append(self.imported_gen_focuses)
            print(f"Debug, self.speaker_info is {self.speaker_info}")

        else:
            self.num_gens =0
            self.create_initial_tensors(num_agents=non_input_initialisation_agents, passed_focuses= passed_focuses)
            print(f"Debug, self.speaker_info is {self.speaker_info}")

        """Debugging sections below while the code is still being made"""
        # print(f"Number of imported agents is {self.num_imported_agents}")
        print(f"Population phi has shape {self.population_phi.shape}")
        print(f"Population counts has shape {self.population_counts.shape}")
        print(f"Alpha tensor has shape {self.alpha_tensor.shape} and is {self.alpha_tensor} ")
        print(f"Number of agents is {self.num_agents}")
        self.numba_warmup()

    def read_param_file(self):
        """Reads the inputted parameter.csv file and updates/records the relevant ensemble parameter values accordingly"""
        params_df = pd.read_csv(self.param_dir/"simulation_parameters.csv")
        self.imported_meanings = params_df['num_meanings'][0]
        # print(self.M)
        self.imported_signals = params_df['num_signals'][0]
        self.num_imported_agents = params_df['num_agents'][0]
        self.imported_gens_lambda = params_df['lambda'][0]
        self.imported_beta = params_df['beta'][0]
       
        old_gamma = params_df['gamma'][0]
        if self.gamma is None:
            self.gamma = old_gamma
        else:
            self.imported_gamma = old_gamma

        self.imported_gen_focuses = params_df['generation_focuses'].apply(ast.literal_eval)[0]
        self.imported_A = params_df['alignment'][0]
        #Need the extra handling to convert the csv's string back into an actual python list
        self.imported_certainty = params_df['certainty'][0]

    def create_generation_focuses(self,number_focuses = 2, supplied_focuses = None, remove_focuses = True, gamma = None):
        """A function to return attentional counts, modified by a select number of attentional focuses, for a generation.  
        Focuses are either selected randomly from an inputted list of allowed focuses, or they are supplied directly.  The degree to which these counts are modifed
        ('focused on') comes from the parameter gamma, which is a straighforward multiplier of the number of counts.  All meanings
        not within the group of selected focuses have a count value corresponding to beta/M where M is total number of meanings.


        If remove_focuses is set to true, the function removes the selected or inputted focuses from the possible focuses list

        Note: likely need to add gamma as an arg, if no gamma is supplied use the self.gamma, just to allow for further control
        """

        if gamma is None:
            gamma = self.gamma
        
        if supplied_focuses is None:
            selected_focuses = np.random.choice(self.possible_focuses,size = number_focuses, replace =False)
            self.gen_focuses.append(selected_focuses.tolist())
        else:
            selected_focuses = supplied_focuses


        gen_counts = np.full(self.M,(self.beta/self.M))
        for focus in selected_focuses:
            gen_counts[focus] *=gamma
            if remove_focuses:
                # print(f"Debug, inst_focus is {focus}")
                self.possible_focuses.remove(focus)
        
        return gen_counts, selected_focuses


    #############################################################################################################################################
    def create_initial_tensors_from_inputs(self):
        """Function to create the initial ensemble tensors from imported data"""
        with h5py.File(self.raw_dir/"array_data.h5", 'r') as f:
            final_phis = f['agents_phi']['agents_phi_data']
            final_counts = f['agents_counts']['agents_counts_data']

            self.population_phi = final_phis[:]
            self.population_counts = final_counts[:]
            self.avg_map.append(self.num_imported_agents)

            #############################################################################################################################################
            gen_counts, focuses = self.create_generation_focuses(2, supplied_focuses= self.imported_gen_focuses[0], gamma=self.imported_gamma)
            
            """NOTE: Above line will absolutely break if importing more than one generation, special care will be needed when that is done"""
            #############################################################################################################################################
            self.alpha_tensor = np.array(gen_counts)
        f.close()

        num_added = self.population_phi.shape[0]
        print(f"Num added is {self.population_phi.shape[0]}") 
        self.num_agents+=num_added

        self.speaker_info = np.empty((num_added,2))
        """Creating a list to store information about the speakers, i.e., what generation they belong to now, but later 
        if I want differ alignment/certainty etc, it will all be stored here
        
        speaker_info:
        0 - generation
        1 - Lambda
        
        """
        self.speaker_info[:,0] = int(self.num_gens)
        self.speaker_info[:,1] = self.imported_gens_lambda




    def create_initial_tensors(self, num_agents, passed_focuses = None):
        """Function to create the initial ensemble tensors at their base values, i.e., in the case of no imported agents"""
        if passed_focuses is not None:
            gen_counts, selected_focuses = self.create_generation_focuses(number_focuses=2, supplied_focuses = passed_focuses)
            self.alpha_tensor = np.array(gen_counts)
        else:
            self.alpha_tensor = np.full((self.M), fill_value=(self.beta/self.S))

        self.population_phi = np.full((num_agents, self.S,self.M), fill_value=1/self.S)
        self.population_counts = np.zeros((num_agents, self.S,self.M))
        self.avg_map.append(num_agents)
        self.num_agents+=num_agents

        """Creating a list to store information about the speakers, i.e., what generation they belong to now, but later 
        if I want differ alignment/certainty etc, it will all be stored here
        
        speaker_info:
        0 - generation
        1 - Lambda
        
        """

        self.speaker_info = np.empty((num_agents, 2))
        self.speaker_info[:,0] = int(self.num_gens)
        self.speaker_info[:,1] = self.lambda_val

    def numba_warmup(self):
        test_array = np.array([0,1])
        dummy_matrix = self.population_phi[0,:,:]
        dummy_alpha = self.alpha_tensor

        print("Initialising Numba Functions")
        print("Testing delta")
        delta(1,1)
        print("Testing update_phi")
        update_phi(dummy_matrix,
                   self.alpha,self.alpha_on_s,1,self.M)

        print("Testing update_signal meaning")
        update_signal_meaning(dummy_matrix,self.lambda_val,1,1,self.S)  

        print("Testing numba_random_choice")
        numba_random_choice(test_array,test_array)
        
        print("Testing select_signal_integers_njit")
        discard, rho_dist = select_signal_integers_njit(dummy_alpha,dummy_matrix,
                                    self.meanings_integers,self.signals_integers)

        print("Testing receive_signal_integers_njit")
        receive_signal_integers_njit(1,rho_dist,dummy_matrix,self.meanings_integers,
                                     self.M)
        
        print("Test blind success inner loop")
        blind_success_inner_loop(dummy_matrix,dummy_matrix,self.S,self.M)
        print("Test blind success outer loop")
        blind_success_outer_loop(self.num_agents,self.population_phi,self.S,self.M)

    def generate_initial_h5_file(self):
            """Generates the h5_file used to store data as the experiment runs, should only be 
            used after the initial ensemble arrays have been calculated"""
            save_dir = self.output_dir/"raw"
            save_dir.mkdir(parents = True, exist_ok = True)
            self.h5_file = h5py.File(name =save_dir/"array_data.h5", mode = 'w')

            """Ensemble_Phi Data"""
            self.h5_phi_group = self.h5_file.create_group(name = 'Phi')
            self.h5_phi_dataset = self.h5_phi_group.create_dataset(name = 'phi_dataset', shape = (self.num_measurements,self.S, self.M), 
                                                        dtype = np.float64, chunks = True)
            self.h5_phi_dataset.attrs['description'] = 'Ensemble phi arrays over time'
            self.h5_phi_dataset.attrs['num_signals'] = self.S
            self.h5_phi_dataset.attrs['num_meanings'] = self.M

            self.h5_phi_dataset[0,:,:] = self.population_phi.mean(axis =0)


            """Generation specific phi data"""
            self.h5_phi_gens_dataset = self.h5_phi_group.create_dataset(name = 'phi_gens_dataset', shape = (self.num_measurements,self.num_gens+1,self.S,
                        self.M),maxshape = (self.num_measurements,None, self.S, self.M),
                        dtype = np.float64, chunks = True)
            #Want it to be none in both the first and second generation to allow in simulation updates later on
            self.h5_phi_gens_dataset.attrs['description'] = 'Generation specific average phi arrays over time'
            self.h5_phi_gens_dataset.attrs['num_signals'] = self.S
            self.h5_phi_gens_dataset.attrs['num_meanings'] = self.M
            self.h5_phi_gens_dataset.attrs['num_gens'] = self.num_gens


            # print(f"Starting h5_phi_gens values")
            # print(f"Keys of gens dict is {self.gen_agent_dict.keys()}")
            # print(f"Avg map is {self.avg_map}")
            start_idx = 0
            for i in range(self.num_gens+1):
                # print(i)
                end_idx = start_idx+self.avg_map[i]
                gens_phi = self.population_phi[start_idx:end_idx,:,:]
                # print(f"Shape of gens_phi is {gens_phi.shape}")

                self.h5_phi_gens_dataset[0,i,:,:] = gens_phi.mean(axis =0)
                start_idx = end_idx

            
            """Ensemble_n(S|M) Data"""
            self.h5_counts_group = self.h5_file.create_group(name = 'counts')
            self.h5_counts_dataset = self.h5_counts_group.create_dataset(name = 'counts_dataset', 
                        shape = (self.num_measurements, self.S, self.M), dtype = np.float64, chunks = True)
            self.h5_counts_dataset.attrs['description'] = "Ensemble signal meaning count array over time "
            self.h5_counts_dataset.attrs['num_signals'] = self.S
            self.h5_counts_dataset.attrs['num_meanings'] = self.M

            self.h5_counts_dataset[0,:,:] = self.population_counts.mean(axis =0)


            """Generation specific counts data, copying same overall structure as the gens phi"""
            self.h5_counts_gens_dataset = self.h5_counts_group.create_dataset(name = 'counts_gens_dataset', 
                shape = (self.num_measurements,self.num_gens+1,self.S, self.M),
                maxshape = (self.num_measurements,None, self.S, self.M),dtype = np.float64, chunks = True)
            #Want it to be none in both the first and second generation to allow in simulation updates later on
            self.h5_counts_gens_dataset.attrs['description'] = 'Generation specific average counts arrays over time'
            self.h5_counts_gens_dataset.attrs['num_signals'] = self.S
            self.h5_counts_gens_dataset.attrs['num_meanings'] = self.M
            self.h5_counts_gens_dataset.attrs['num_gens'] = self.num_gens


            # print(f"Starting h5_counts_gens values")
            # print(f"Keys of gens dict is {self.gen_agent_dict.keys()}")
            # print(f"Avg map is {self.avg_map}")
            start_idx = 0
            for i in range(self.num_gens+1):
                print(i)
                end_idx = start_idx+self.avg_map[i]
                gens_counts = self.population_counts[start_idx:end_idx,:,:]
                print(f"Shape of gens_counts is {gens_counts.shape}")

                self.h5_counts_gens_dataset[0,i,:,:] = gens_counts.mean(axis =0)
                start_idx = end_idx

            """If self.export_agents set to true, we need an array to save the final phi and count arrays from all agents """
            if self.export_agents:
                self.h5_agents_phi_group = self.h5_file.create_group(name = 'agents_phi')
                self.h5_agents_phi_data =self.h5_agents_phi_group.create_dataset(name = 'agents_phi_data', 
                                                                                shape = (self.num_agents,self.S,self.M), dtype = np.float64)
                self.h5_agents_phi_data.attrs['description'] = "Each agent in the simulation's final phi array"
                self.h5_agents_phi_data.attrs['num_signals'] = self.S
                self.h5_agents_phi_data.attrs['num_meanings'] = self.M


                self.h5_agents_counts_group = self.h5_file.create_group(name = 'agents_counts')
                self.h5_agents_counts_data =self.h5_agents_counts_group.create_dataset(name = 'agents_counts_data', 
                                                                                shape = (self.num_agents,self.S,self.M), dtype = np.float64)
                self.h5_agents_counts_data.attrs['description'] = "Each agent in the simulation's final phi array"
                self.h5_agents_counts_data.attrs['num_signals'] = self.S
                self.h5_agents_counts_data.attrs['num_meanings'] = self.M

    def save_to_file(self,idx_iter):
        """Save the experimental data to a file as it is calculated/experiment is performed"""
        save_dir = self.output_dir/"raw"
        save_dir.mkdir(parents = True, exist_ok = True)

        meta_data = {
            "num_agents": self.num_agents,
            "num_signals": int(self.S),
            "num_meanings": int(self.M),
            "lambda_val": self.speaker_info[0,1]
            
        }
        with open(save_dir/"metadata.json","w") as f:
            json.dump(meta_data,f,indent=4)


        if hasattr(self,"gain_values"):
            pd.DataFrame({"epoch": self.gain_epochs, "gain": self.gain_values}).to_csv(save_dir / "gain_values.csv",
                                                                                        index=False)

        if idx_iter is not None:
            """If iteration is not none, we append the data we found to the h5py file we initialised earlier"""

            self.h5_phi_dataset[idx_iter,:,:] = self.population_phi.mean(axis = 0)
            self.h5_counts_dataset[idx_iter,:,:] = self.population_counts.mean(axis = 0)

            start_idx = 0
            for k in range(self.num_gens+1):
                end_idx = self.avg_map[k] + start_idx
                ensemble_phi = self.population_phi[start_idx:end_idx,:,:]
                ensemble_counts = self.population_counts[start_idx:end_idx,:,:]
                self.h5_phi_gens_dataset[idx_iter,k,:,:] = ensemble_phi.mean(axis = 0)
                self.h5_counts_gens_dataset[idx_iter,k,:,:] = ensemble_counts.mean(axis = 0)
                start_idx = end_idx

    def save_agents_data(self):
        """A function to be called if export agents is set to true, saves the individual agent's phi and count arrays to the h5 file"""
        file_dir = self.output_dir/"raw"
        with h5py.File(file_dir/"array_data.h5", 'a') as f:
            agents_phi_group = f['agents_phi']
            agents_phi_data = agents_phi_group['agents_phi_data']
            agents_counts_group = f['agents_counts']
            agents_counts_data = agents_counts_group['agents_counts_data']
            for i in range(self.num_agents):
                agents_phi_data[i,:,:] = self.population_phi[i,:,:]
                agents_counts_data[i,:,:] = self.population_counts[i,:,:]

            f.close()
        print("Agent data saved")

    def save_params(self,iterations):
            params_dict = {'num_meanings': self.M,
                        'num_signals': self.S,
                        'num_agents': self.num_agents,
                        'lambda':self.lambda_val,
                            'alpha': self.alpha,
                            'beta': self.beta,
                            'gamma':self.gamma,
                            'generation_focuses':self.gen_focuses,
                            'alignment':self.A,
                            'iterations':iterations,
                            'number_of_gens':self.num_gens,
                            'imported_file': self.param_dir,
                            'avg_map': [float(x) for x in self.avg_map],
                            'gen_alphas':  np.asarray(self.alpha_tensor).tolist(),
            }
            params_df = pd.DataFrame([params_dict])
            params_file = self.output_dir / "simulation_parameters.csv"
            params_df.to_csv(params_file, index=False)
            print(f"Saved simulation parameters to {params_file}")        

    def add_speakers(self, speakers_added, speakers_focuses = None):
        """A function to add speakers into the system.  Extends the ensemble tensors by the appropriate amount, and then 
        fills them with the values corresponding to a new ('young'/'blank') speaker.  Additionally, adjusts both the alpha 
        tensor, and the avg map list accordingly.  """

        new_speakers_phi = np.full((speakers_added, self.S,self.M), fill_value=1/self.S)
        new_speakers_counts = np.zeros((speakers_added, self.S,self.M))

        self.population_phi = np.vstack((self.population_phi, new_speakers_phi))
        self.population_counts = np.vstack((self.population_counts, new_speakers_counts))

        self.avg_map.append(speakers_added)

        if speakers_focuses is not None:
            gen_counts, selected_focuses = self.create_generation_focuses(number_focuses=2, supplied_focuses = speakers_focuses)
            self.gen_focuses.append(selected_focuses)
            new_alpha = np.array(gen_counts)
        else:
            new_alpha = np.full((self.M), fill_value=(self.beta/self.S))
    
        self.alpha_tensor = np.vstack((self.alpha_tensor,new_alpha))
        self.num_agents+=speakers_added
        self.num_gens+=1

        new_speaker_info = np.full((speakers_added,2), fill_value=(int(self.num_gens),self.lambda_val))
        print(f"Debug, new speaker info is {new_speaker_info}")
        self.speaker_info = np.vstack((self.speaker_info,new_speaker_info))
        print(f"Debug, speaker info is {self.speaker_info}")


#############################################################################################################################################
    def add_intergen_alpha(self,gen_0_weight = 0.5,gen_1_weight = 0.5):
        assert self.alpha_tensor.shape[0] != 1, " Only one gen's alpha in tensor"
        inter_gen_alpha =  gen_0_weight * self.alpha_tensor[0] + gen_1_weight * self.alpha_tensor[1]
        self.alpha_tensor = np.vstack((self.alpha_tensor,inter_gen_alpha))
        """Will need a way of thinking about how to do this in the case we have 3+ generations communicating, as it currently is written it would fail"""

    def one_interaction_multiple_gens_numba_test(self):
        print(f"Debug, testing one interaction numba")
        new_phi,new_counts = one_interaction_multiple_gens(self.population_counts,self.population_phi,self.speaker_info,self.meanings_integers,
                                                           self.signals_integers,self.num_agents,self.alpha_tensor,self.M,
                                                           self.S,self.lambda_val,self.alpha)

    def one_loop_multiple_gens_numba_test(self):
        print(f"Debug, testing one data loop numba")
        new_phi,new_counts = one_collection_loop_multiple_gens(self.population_counts,self.population_phi,self.speaker_info,self.meanings_integers,
                                                                   self.signals_integers,self.num_agents,self.alpha_tensor,self.M,
                                                                   self.S,self.lambda_val,self.alpha, iterations=2)

    def one_interaction_one_pop_numba_test(self):
        print(f"Debug, testing one population numba tests")

        newphi,new_counts = one_interaction_one_population(self.population_counts,self.population_phi,self.meanings_integers,
                                                           self.signals_integers,self.num_agents,self.alpha_tensor,self.M,
                                                           self.S,self.lambda_val,alpha_val=self.alpha,alignment=self.A)

        print(f"Testing training loop")
        new_phi,new_counts = one_collection_loop_one_pop(self.population_counts,self.population_phi,self.meanings_integers,
                                                           self.signals_integers,self.num_agents,self.alpha_tensor,self.M,
                                                           self.S,self.lambda_val,self.alpha,alignment=self.A, iterations=2)

    def training_loop_one_pop(self):
        print(f"Caching required numba functions")
        self.one_interaction_one_pop_numba_test()

        self.generate_initial_h5_file()
        for i in range(self.num_measurements):
            new_population_phi, new_population_counts = one_collection_loop_one_pop(self.population_counts,
                                                                    self.population_phi,self.meanings_integers,
                                                                   self.signals_integers,self.num_agents,self.alpha_tensor,self.M,
                                                                   self.S,self.lambda_val,self.alpha,alignment=self.A, iterations=self.time_res)
            self.save_to_file(idx_iter=i)
            self.population_phi = new_population_phi
            self.population_counts = new_population_counts
            print(f"Completed iteration {(i+1) * self.time_res}")
        
        if self.export_agents:
            self.save_agents_data()

        self.save_params(iterations = self.tot_iterations)

    def training_loop_multiple_gens(self):
        print(f"Caching required numba functions")
        self.one_interaction_multiple_gens_numba_test()
        self.one_loop_multiple_gens_numba_test()

        self.generate_initial_h5_file()
        for i in range(self.num_measurements):
            new_population_phi, new_population_counts = one_collection_loop_multiple_gens(self.population_counts,
                                                                    self.population_phi,self.speaker_info,self.meanings_integers,
                                                                   self.signals_integers,self.num_agents,self.alpha_tensor,self.M,
                                                                   self.S,self.lambda_val,self.alpha, iterations=self.time_res)
            self.save_to_file(idx_iter=i)
            self.population_phi = new_population_phi
            self.population_counts = new_population_counts
            print(f"Completed iteration {(i+1) * self.time_res}")
        
        if self.export_agents:
            self.save_agents_data()

        self.save_params(iterations = self.tot_iterations)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--NumMeanings', type = int, help="Determines the numbers of meanings to be created/used in the case where meanings are not" \
    "hardcoded", default=12)
    parser.add_argument('--NumSignals', type = int, help="Determines the numbers of signals to be created/used in the case where signals are not" \
    "hardcoded", default=12)
    parser.add_argument('--NumAgents', type = int, help="Determines the numbers of communicating/learning agents", default=5)
    parser.add_argument('--Beta', type = int, help="Determines the value of parameter beta", default=49)
    parser.add_argument('--Alpha', type = float, help="Determines the value of parameter alpha", default=0.1)
    parser.add_argument('--lambda_val', type = float, help="Determines the value of lambda, or the forgetting rate of the models", default=0.01)
    parser.add_argument('--alignment', type = float, help="Determines the alignment between models", default=1.0)
    parser.add_argument('--iterations', type=float, help = "Determines how many millions of iterations to run the code for", default = 4.0)
    parser.add_argument('--OutputDir', type = str, help = 'Determines the file output of saved plots, data, etc', 
                            default =str(Path.home()/"Downloads"/"MPhys/Code_Runs"))
    parser.add_argument('--Filename', type = str, help='Determines the file to save data to', default= 'Unsorted')
    parser.add_argument('--ImportFile', type = str, help='Determines the file/folder to read the imported data from', default= 'Unsorted')
    parser.add_argument('--Generations', action='store_true', help='If set, have independent generations rho distributions')
    parser.add_argument('--GenCount', type = int, help="Determines the number of generations", default=1)
    parser.add_argument('--gamma', type = int, help="Determines the parameter gamma, which controls \
                        how much the generational focus is multiplied by", default=2)
    parser.add_argument('--Focuses', type = int, help="Determines how many focuses are in each generation", default=2)
    parser.add_argument('--PassedFocuses', type = int,nargs='+', help ="If given inputs, sets the specific focuses of the generation", default = None)
    parser.add_argument('--TimeRes', type = float, help="Determines how often to collect data, in multipliers of 1 hundred thousand", default = 1.0)
    parser.add_argument('--Export', action='store_true', help='If set, exports the agents')
    args = parser.parse_args()

    print("Past args")

    input_dir = Path(args.OutputDir)/args.ImportFile

    output_directory = Path(args.OutputDir)/args.Filename
    output_directory.mkdir(parents = True, exist_ok = True)#Creates the folder if it does not already exist

    time_res = int(args.TimeRes * 100000)
    total_iterations = int(args.iterations * 1000000)

    
    debugging_ensemble = Ensemble(input_dir, output_directory, alignment=args.alignment, beta=args.Beta, alpha = args.Alpha,
                                  lambda_val=args.lambda_val,num_meanings=args.NumMeanings,num_signals=args.NumSignals,
                                   time_resolution=time_res, iterations=total_iterations, new_gamma=args.gamma, 
                                   passed_focuses=args.PassedFocuses, import_agents=False, non_input_initialisation_agents=args.NumAgents,
                                   export_agents=args.Export)
    
    
    

    # debugging_ensemble.generate_initial_h5_file()
    # print(f"Debug Shape of intergen_tensor = {debugging_ensemble.inter_gen_alpha.shape}, and is {debugging_ensemble.inter_gen_alpha}")
    train_start_time = time.time()
    debugging_ensemble.training_loop_one_pop()
    train_end_time = time.time()
    print(f"Runtime is {train_end_time-train_start_time}" )