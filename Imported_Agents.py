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
def receive_signal_integers_njit(signal_idx,A,passed_rho_dist,alpha_dist,phi_array,meanings_integers,num_meanings):
    """A numba version of the signal receiving and interpretation process, confined to one generation"""
    # rho_roll = np.random.rand()
    # if rho_roll<=A:
    #     rho_dist = passed_rho_dist
    # else:
    #     rho_dist = np.random.dirichlet(alpha=alpha_dist)

    """Above is commented out as we have not yet begun experimenting with different inter-agent alignment"""

    # print("Rho dist")
    # print(rho_dist)
    

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


class Agent:
    """The agent parent class will be able to act as both the signaller and the receiver, as such it needs to have the set of meanings
    and signals available to it, as well as it's own attentional weight distribution.
    Following the instructions put out in the "Evolution of communication through fluctuations" background reading, the signals (S) and 
    meanings (M) form a matrix, this matrix is initialised to be empty.

    Additionally, the weight array will be initialised to be constant, I am not quite sure how to introduce the variation as a result of 
    certainty into it at this moment
    
    """
    def __init__(self,meanings,signals, lambda_val, alpha, beta,num_meanings, num_signals,passed_counts = None, generation_tag =None):
        self.num_meanings = num_meanings
        self.num_signals = num_signals
        
        self.lambda_val = lambda_val

        self.meanings = meanings
        
        self.signals = signals

        self.generation_tag = generation_tag #A identifier for which generation this particular agent corresponds to

        self.meanings_integers = np.arange(self.num_meanings, dtype=int)
        self.signals_integers = np.arange(self.num_signals,dtype=int)
        """In most cases, signal and meaning integers are entirely the same.  It has however been left in the code,
        as if there is ever any desire to further complicate signals/meanings (multi-character signals, more than one signal
        being communicated in one instance of communication, etc.), this is a valuable array to simplify computations"""


        self.Alpha = alpha
        self.Alpha_s = alpha/self.num_signals

        self.beta = beta

        self.certainty = 1/(1+beta)
        if passed_counts is not None:
            self.alpha_dist = passed_counts
        else:
            self.alpha_dist = np.full(self.num_meanings, (beta/self.num_meanings))
        # print(self.alpha_dist)
        
        #Above is to be used for the rho distributions, it has no bearing on the self.Alpha value
        # print(f"The agent alpha array  is of form {self.alpha_dist.dtype}, and shape {self.alpha_dist.shape}")

        self.signal_meaning_array = np.zeros((self.num_signals,self.num_meanings)) #Initialises a zero array of size S X M
        """The signal meaning array tracks how many times an agent has received signal s when they interpreted meaning m"""
        self.phi_array = np.full((self.num_signals,self.num_meanings),fill_value= (1/self.num_signals))
        """At initialisation the phi array is uniformally distributed such that all signals are equally likely"""

    def select_signal_numba(self):
        selected_signal, inst_rho = select_signal_integers_njit(self.alpha_dist,self.phi_array,self.meanings_integers,self.signals_integers)
        return selected_signal,inst_rho

    def receive_signal_numba(self,signal_idx,rho_dist,A):
        nu_idx = receive_signal_integers_njit(signal_idx,A,rho_dist,self.alpha_dist,self.phi_array,self.meanings_integers,self.num_meanings)
        return nu_idx

    def receive_signal_intergen(self,signal_idx,intergen_alpha):
        nu_idx = receive_signal_intergers_intergen_njit(signal_idx,intergen_alpha,self.phi_array,self.meanings_integers,self.num_meanings)
        return nu_idx
    
    def update_counts(self, nu_idx, signal_idx):
        """Update rule is that all values in the signal_meaning_array decay by (1-lambda)*current_value, 
        with the exception of the actual signal, which while it does decay, is also incremented by 1. """

        self.signal_meaning_array = update_signal_meaning(self.signal_meaning_array,self.lambda_val,
                                                          nu_idx,signal_idx,self.num_signals)

        """Phi is calcualted based upon the counts array, so updating phi must come after we update the signal meaning array"""
        new_phi_row = update_phi(self.signal_meaning_array,self.Alpha,self.Alpha_s,signal_idx,self.num_meanings)
        self.phi_array[signal_idx,:] = new_phi_row
        


class Imported_Ensemble:
    """A class very similar to the base ensmeble class, except for the fact that it allows for the importing of already established agent networks.
    This allows us to skip tedious, uninteresting monogenerational runs to baseline, speeding up computation time significantly.  Additionally, this 
     creates the ability to control our studies, as all increased generational trials can be done on the same configuration. """
    def __init__(self, read_dir,agent_fn,output_dir, iterations, time_resolution = 100000, 
                 generation_overlap = False, export_agents = False, new_gamma = None, passed_focuses = None):
        
        self.param_dir = Path(read_dir)
        self.raw_dir = Path(read_dir)/"raw"
        self.output_dir = Path(output_dir)
        self.agent_fn = agent_fn
        self.export_agents = export_agents
        self.gamma = new_gamma

        self.time_resolution = time_resolution
        self.iterations = iterations
        self.num_measurements = iterations/time_resolution

        self.new_gen_focuses = passed_focuses

        self.agents_list = []


        self.gen_alpha_dists = []

        #List to store the alpha arrays of each generation

        self.gen_agent_dict = {}

        self.read_param_file()
        print(self.gen_focuses) #test
        self.possible_focuses = np.arange(self.num_meanings).tolist()
        print(self.possible_focuses) # test

        # print(f"number of different generation focus sets is {len(self.gen_focuses)}, the amount of focuses per each gen is {len(self.gen_focuses[0])}")
        self.num_gens = len(self.gen_focuses)-1
        #Set to be -1 so I can use it as a dict index easier


        if hasattr(self,'imported_gamma'):
            print("Gamma supplied, using imported_gamma")
            self.recreate_agents(gamma = self.imported_gamma)
        else:
            print(f"No gamma supplied")
            self.recreate_agents()

        # print(f"Gen alpha distribution is {self.gen_alpha_dists}")
        self.num_agents = len(self.agents_list)

        self.numba_warmup()

    def read_param_file(self):
        """Reads the inputted parameter.csv file and updates/records the relevant ensemble parameter values accordingly"""
        params_df = pd.read_csv(self.param_dir/"simulation_parameters.csv")
        self.num_meanings = params_df['num_meanings'][0]
        # print(self.num_meanings)
        self.num_signals = params_df['num_signals'][0]
        self.num_imported_agents = params_df['num_agents'][0]
        self.imported_gens_lambda = params_df['lambda'][0]
        self.beta = params_df['beta'][0]
       
        old_gamma = params_df['gamma'][0]
        if self.gamma is None:
            self.gamma = old_gamma
        else:
            self.imported_gamma = old_gamma

        self.gen_focuses = params_df['generation_focuses'].apply(ast.literal_eval)[0]
        self.A = params_df['alignment'][0]
        #Need the extra handling to convert the csv's string back into an actual python list
        self.certainty = params_df['certainty'][0]

        # print(self.num_signals,self.num_imported_agents,self.imported_gens_lambda,self.beta,self.gamma,self.imported_gens_focuses,self.certainty)

    def create_generation_counts(self,number_focuses = 2, supplied_focuses = None,remove_focuses = True, gamma = None):
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


        gen_counts = np.full(self.num_meanings,(self.beta/self.num_meanings))
        for focus in selected_focuses:
            gen_counts[focus] *=gamma
            if remove_focuses:
                self.possible_focuses.remove(focus)
        
        # print(f"Gen_counts has datatype {gen_counts.dtype}")
        return gen_counts, selected_focuses

    def recreate_agents(self, gamma = None):
        """Recreates the agents that were passed to the imported ensemble class.  Does so by using the final phi and count arrays stored 
        in the h5 file, and using the generational_focuses that were passed to the imported_ensemble class via the params.csv file.  
        This function also isntantiates the ensemble phi and counts arrays


        NOTE: Special care will be needed later if we are ever passing a configuration with more than 1 generation in it, as it will fail 
        if doing so
        """

        if gamma is None:
            gamma = self.gamma

        for i in range(len(self.gen_focuses)):
            inst_focus = self.gen_focuses[i]
            number_focus = len(inst_focus)
            gen_counts, selected_focuses = self.create_generation_counts(number_focuses=number_focus, supplied_focuses=inst_focus, gamma=gamma)
            print(f"Debug, gen_counts made in recreate agents")

            self.gen_alpha_dists.append(gen_counts.tolist())

        inst_agent_list = []

        self.ensemble_phi = np.empty((self.num_imported_agents,self.num_signals,self.num_meanings))
        self.ensemble_counts = np.empty_like(self.ensemble_phi)

        with h5py.File(self.raw_dir/"array_data.h5", 'r') as f:
            final_phis = f['agents_phi']['agents_phi_data']
            final_counts = f['agents_counts']['agents_counts_data']
            # print(f"Final counts is {final_counts}")
        
            for k in range(self.num_imported_agents):
                inst_agent = self.agent_fn(passed_counts = gen_counts, generation_tag = self.num_gens)
                self.agents_list.append(inst_agent)
                inst_agent_list.append(inst_agent)
                self.agents_list[k].phi_array = final_phis[k,:,:]
                self.ensemble_phi[k,:,:] = final_phis[k,:,:]
                self.agents_list[k].signal_meaning_array = final_counts[k,:,:]
                # print(final_counts[k,:,:])
                self.ensemble_counts[k,:,:] = final_counts[k,:,:]
            
            f.close()

        #Now to add these new agents to the dict
        self.gen_agent_dict[self.num_gens] = inst_agent_list

        self.avg_map = self.make_avg_map()

    def numba_warmup(self):
        test_array = np.array([0,1])

        print("Initialising Numba Functions")
        selected_agent = self.agents_list[0]
        print("Testing delta")
        delta(1,1)
        print("Testing update_phi")
        update_phi(selected_agent.signal_meaning_array,
                   selected_agent.Alpha,selected_agent.Alpha_s,1,selected_agent.num_meanings)

        print("Testing update_signal meaning")
        update_signal_meaning(selected_agent.signal_meaning_array,selected_agent.lambda_val,1,1,selected_agent.num_signals)  

        print("Testing numba_random_choice")
        numba_random_choice(test_array,test_array)
        
        print("Testing select_signal_integers_njit")
        discard, rho_dist = select_signal_integers_njit(selected_agent.alpha_dist,selected_agent.phi_array,
                                    selected_agent.meanings_integers,selected_agent.signals_integers)

        print("Testing receive_signal_integers_njit")
        receive_signal_integers_njit(1,1,rho_dist,test_array,selected_agent.phi_array,selected_agent.meanings_integers,
                                     selected_agent.num_meanings)
        
        print("Test blind success inner loop")
        blind_success_inner_loop(self.agents_list[0].phi_array,self.agents_list[1].phi_array,self.num_signals,self.num_meanings)
        print("Test blind success outer loop")
        blind_success_outer_loop(self.num_agents,self.ensemble_phi,self.num_signals,self.num_meanings)




   ##########################################################################
    def add_agents(self, number_added):
        """Adds new agents (of a new generation) to the agents list.  Removes corresponding new focuses from the possible focuses,
        increases the counter for the number of agents, and updates the ensemble arrays accordingly."""
        
        new_generation_counts, selected_focuses = self.create_generation_counts(supplied_focuses=self.new_gen_focuses)
        self.gen_focuses.append(self.new_gen_focuses)
        """Above line in particular will be problematic if adding more than one generation at one time"""
        ##########################################################################

        self.gen_alpha_dists.append(new_generation_counts.tolist())

        self.num_agents+=number_added
        self.num_gens+=1

        # print(f"Length of agent list before is {len(self.agents_list)}") #test
        inst_agent_list = []
        for i in range(number_added):
            inst_agent = self.agent_fn(passed_counts = new_generation_counts, generation_tag = self.num_gens)
            self.agents_list.append(inst_agent)
            inst_agent_list.append(inst_agent)

        self.gen_agent_dict[self.num_gens] = inst_agent_list

        new_gen_phi = self.agents_list[-1].phi_array
        new_gen_count = self.agents_list[-1].signal_meaning_array
        
        phi_block = np.repeat(new_gen_phi[None], repeats=number_added, axis = 0)
        counts_block = np.repeat(new_gen_count[None], repeats=number_added, axis = 0)

        # print(f"Shapes of ensemble phi and counts before adding are {self.ensemble_phi.shape}, and {self.ensemble_counts.shape}")

        self.ensemble_phi = np.concatenate([self.ensemble_phi, phi_block], axis =0)
        self.ensemble_counts = np.concatenate([self.ensemble_counts,counts_block], axis = 0)
        # print(f"Shapes of ensemble phi and counts after adding are {self.ensemble_phi.shape}, and {self.ensemble_counts.shape}")
        # print(f"Length of agent list after is {len(self.agents_list)}") #test
        self.avg_map = self.make_avg_map()

   ##########################################################################


    def make_avg_map(self):
        avg_map = []
        for i in range(len(self.gen_agent_dict.keys())):
            avg_map.append(len(self.gen_agent_dict[i]))
        return avg_map

    def make_inter_gen_alpha(self, weight_0=0.60, weight_1 = 0.40):
        ensemble_alphas = np.array(self.gen_alpha_dists)
        print(ensemble_alphas.shape)
        self.inter_gen_alpha = ensemble_alphas.mean(axis =0)

        # print(np.array(self.gen_alpha_dists[0]))
        # print(np.array(self.gen_alpha_dists[1]))
        # self.inter_gen_alpha = np.array(self.gen_alpha_dists[0]) * weight_0 + np.array(self.gen_alpha_dists[1])*weight_1
        # print(self.inter_gen_alpha)
        # print(f"Length of intergen alpha is {self.inter_gen_alpha.shape}")

    def one_interaction(self,agent_ids):
        """Iteration procedure for one agent to agent interaction, copied directly from the ensmble class"""
        selected_agents = np.random.choice(agent_ids,size=2, replace=False)
        
        signaller_agent = self.agents_list[selected_agents[0]]
        signaller_gen_tag = signaller_agent.generation_tag
        
        receiver_agent = self.agents_list[selected_agents[1]]
        receiver_gen_tag = receiver_agent.generation_tag
        
        selcted_signal_idx,inst_rho = signaller_agent.select_signal_numba()

        if signaller_gen_tag == receiver_gen_tag:
            nu_idx = receiver_agent.receive_signal_numba(selcted_signal_idx,inst_rho,self.A)
        else:
            nu_idx = receiver_agent.receive_signal_intergen(selcted_signal_idx,self.inter_gen_alpha)


        receiver_agent.update_counts(nu_idx,selcted_signal_idx)

    def update_ensemble_arrays(self):
        """Numba cannot be passed an agent_type at all, as such, we need to create tensors for the ensemble average values,
        i.e., values we would otherwise obtain in a numba function from the objects passed."""
        for i in range(self.num_agents):
            self.ensemble_phi[i,:] = self.agents_list[i].phi_array
            self.ensemble_counts[i,:] = self.agents_list[i].signal_meaning_array

    def measure_blind_success(self):
        """Sum over all pairs and then subtract the diagonal elements"""
        p_s = blind_success_outer_loop(self.num_agents,self.ensemble_phi,self.num_signals,self.num_meanings)
        return p_s

    def generate_initial_h5_file(self):
            """Generates the h5_file used to store data as the experiment runs, should only be 
            used after the initial ensemble arrays have been calculated"""
            save_dir = self.output_dir/"raw"
            save_dir.mkdir(parents = True, exist_ok = True)
            self.h5_file = h5py.File(name =save_dir/"array_data.h5", mode = 'w')

            """Ensemble_Phi Data"""
            self.h5_phi_group = self.h5_file.create_group(name = 'Phi')
            self.h5_phi_dataset = self.h5_phi_group.create_dataset(name = 'phi_dataset', shape = (self.num_measurements,self.num_signals, self.num_meanings), 
                                                        dtype = np.float64, chunks = True)
            self.h5_phi_dataset.attrs['description'] = 'Ensemble phi arrays over time'
            self.h5_phi_dataset.attrs['num_signals'] = self.num_signals
            self.h5_phi_dataset.attrs['num_meanings'] = self.num_meanings

            self.h5_phi_dataset[0,:,:] = self.ensemble_phi.mean(axis =0)


            """Generation specific phi data"""
            self.h5_phi_gens_dataset = self.h5_phi_group.create_dataset(name = 'phi_gens_dataset', shape = (self.num_measurements,self.num_gens+1,self.num_signals,
                        self.num_meanings),maxshape = (self.num_measurements,None, self.num_signals, self.num_meanings),
                        dtype = np.float64, chunks = True)
            #Want it to be none in both the first and second generation to allow in simulation updates later on
            self.h5_phi_gens_dataset.attrs['description'] = 'Generation specific average phi arrays over time'
            self.h5_phi_gens_dataset.attrs['num_signals'] = self.num_signals
            self.h5_phi_gens_dataset.attrs['num_meanings'] = self.num_meanings
            self.h5_phi_gens_dataset.attrs['num_gens'] = self.num_gens


            # print(f"Starting h5_phi_gens values")
            # print(f"Keys of gens dict is {self.gen_agent_dict.keys()}")
            # print(f"Avg map is {self.avg_map}")
            start_idx = 0
            for i in range(self.num_gens+1):
                # print(i)
                end_idx = start_idx+self.avg_map[i]
                gens_phi = self.ensemble_phi[start_idx:end_idx,:,:]
                # print(f"Shape of gens_phi is {gens_phi.shape}")

                self.h5_phi_gens_dataset[0,i,:,:] = gens_phi.mean(axis =0)
                start_idx = end_idx

            
            """Ensemble_n(S|M) Data"""
            self.h5_signal_meanings_group = self.h5_file.create_group(name = 'signal_meaning')
            self.h5_signal_meanings_dataset = self.h5_signal_meanings_group.create_dataset(name = 'signal_meanings_dataset', 
                        shape = (self.num_measurements, self.num_signals, self.num_meanings), dtype = np.float64, chunks = True)
            self.h5_signal_meanings_dataset.attrs['description'] = "Ensemble signal meaning count array over time "
            self.h5_signal_meanings_dataset.attrs['num_signals'] = self.num_signals
            self.h5_signal_meanings_dataset.attrs['num_meanings'] = self.num_meanings

            self.h5_signal_meanings_dataset[0,:,:] = self.ensemble_counts.mean(axis =0)


            """Generation specific counts data, copying same overall structure as the gens phi"""
            self.h5_signal_meanings_gens_dataset = self.h5_signal_meanings_group.create_dataset(name = 'signal_meanings_gens_dataset', 
                shape = (self.num_measurements,self.num_gens+1,self.num_signals, self.num_meanings),
                maxshape = (self.num_measurements,None, self.num_signals, self.num_meanings),dtype = np.float64, chunks = True)
            #Want it to be none in both the first and second generation to allow in simulation updates later on
            self.h5_signal_meanings_gens_dataset.attrs['description'] = 'Generation specific average signal_meanings arrays over time'
            self.h5_signal_meanings_gens_dataset.attrs['num_signals'] = self.num_signals
            self.h5_signal_meanings_gens_dataset.attrs['num_meanings'] = self.num_meanings
            self.h5_signal_meanings_gens_dataset.attrs['num_gens'] = self.num_gens


            # print(f"Starting h5_signal_meanings_gens values")
            # print(f"Keys of gens dict is {self.gen_agent_dict.keys()}")
            # print(f"Avg map is {self.avg_map}")
            start_idx = 0
            for i in range(self.num_gens+1):
                print(i)
                end_idx = start_idx+self.avg_map[i]
                gens_signal_meanings = self.ensemble_counts[start_idx:end_idx,:,:]
                print(f"Shape of gens_signal_meanings is {gens_signal_meanings.shape}")

                self.h5_signal_meanings_gens_dataset[0,i,:,:] = gens_signal_meanings.mean(axis =0)
                start_idx = end_idx

            """If self.export_agents set to true, we need an array to save the final phi and count arrays from all agents """
            if self.export_agents:
                self.h5_agents_phi_group = self.h5_file.create_group(name = 'agents_phi')
                self.h5_agents_phi_data =self.h5_agents_phi_group.create_dataset(name = 'agents_phi_data', 
                                                                                shape = (self.num_agents,self.num_signals,self.num_meanings), dtype = np.float64)
                self.h5_agents_phi_data.attrs['description'] = "Each agent in the simulation's final phi array"
                self.h5_agents_phi_data.attrs['num_signals'] = self.num_signals
                self.h5_agents_phi_data.attrs['num_meanings'] = self.num_meanings


                self.h5_agents_counts_group = self.h5_file.create_group(name = 'agents_counts')
                self.h5_agents_counts_data =self.h5_agents_counts_group.create_dataset(name = 'agents_counts_data', 
                                                                                shape = (self.num_agents,self.num_signals,self.num_meanings), dtype = np.float64)
                self.h5_agents_counts_data.attrs['description'] = "Each agent in the simulation's final phi array"
                self.h5_agents_counts_data.attrs['num_signals'] = self.num_signals
                self.h5_agents_counts_data.attrs['num_meanings'] = self.num_meanings

    def save_to_file(self,idx_iter):
        """Save the experimental data to a file as it is calculated/experiment is performed"""
        save_dir = self.output_dir/"raw"
        save_dir.mkdir(parents = True, exist_ok = True)

        # print(type(len(self.agents_list)))
        # print(type(int(self.num_signals)))
        # print(type(int(self.num_meanings)))
        # print(type(self.agents_list[0].lambda_val))

        meta_data = {
            "num_agents": len(self.agents_list),
            "num_signals": int(self.num_signals),
            "num_meanings": int(self.num_meanings),
            "lambda_val": self.agents_list[0].lambda_val
            
        }
        with open(save_dir/"metadata.json","w") as f:
            json.dump(meta_data,f,indent=4)


        if hasattr(self,"gain_values"):
            pd.DataFrame({"epoch": self.gain_epochs, "gain": self.gain_values}).to_csv(save_dir / "gain_values.csv",
                                                                                        index=False)

        if idx_iter is not None:
            """If iteration is not none, we append the data we found to the h5py file we initialised earlier"""
            current_phi_iter_val = self.h5_phi_dataset.shape[0]
            current_counts_iter_val =self.h5_signal_meanings_dataset.shape[0]


            # print(f"Saving to file, saving ensemble data")
            # self.h5_phi_dataset.resize(current_phi_iter_val+1,axis=0)
            # self.h5_signal_meanings_dataset.resize(current_counts_iter_val+1,axis=0)

            # self.h5_phi_gens_dataset.resize(current_phi_iter_val+1,axis = 0)
            # self.h5_signal_meanings_gens_dataset.resize(current_counts_iter_val+1,axis=0)

            self.h5_phi_dataset[idx_iter,:,:] = self.ensemble_phi.mean(axis = 0)
            self.h5_signal_meanings_dataset[idx_iter,:,:] = self.ensemble_counts.mean(axis = 0)

            start_idx = 0
            for i in range(self.num_gens+1):
                end_idx = self.avg_map[i] +start_idx
                ensemble_phi = self.ensemble_phi[start_idx:end_idx,:,:]
                ensemble_counts = self.ensemble_counts[start_idx:end_idx,:,:]
                self.h5_phi_gens_dataset[idx_iter,i,:,:] = ensemble_phi.mean(axis = 0)
                self.h5_signal_meanings_gens_dataset[idx_iter,i,:,:] = ensemble_counts.mean(axis = 0)
                start_idx = end_idx

    
    def save_params(self,iterations,convergence, avg_map):
        params_dict = {'num_meanings': self.num_meanings,
                       'num_signals': self.num_signals,
                       'num_agents': self.num_agents,
                       'lambda':self.agents_list[0].lambda_val,
                        'alpha': self.agents_list[0].Alpha,
                        'beta': self.agents_list[0].beta,
                        'gamma':self.gamma,
                        'generation_focuses':self.gen_focuses,
                        'alignment':self.A,
                        'certainty':self.agents_list[0].certainty,
                        'iterations':iterations,
                        'Steady_State':convergence,
                        'number_of_gens':self.num_gens,
                        'imported_file': self.param_dir,
                        'avg_map': avg_map,
                        'gen_alphas': self.gen_alpha_dists,
                        'inter_gen_alpha':self.inter_gen_alpha
        }
        params_df = pd.DataFrame([params_dict])
        params_file = self.output_dir / "simulation_parameters.csv"
        params_df.to_csv(params_file, index=False)
        print(f"Saved simulation parameters to {params_file}")        

    def print_lambdas(self):
        """Debug function to print all the lambda vals"""
        for i in range(self.num_agents):
            print(f"Lambda val for agent {i} is {self.agents_list[i].lambda_val}")

    def debug_counts(self):
        """Debug the count arrays"""

    def training_loop(self):

        self.update_ensemble_arrays()
        #Initialise the files to save the data in
        self.generate_initial_h5_file()
        agent_ids = np.arange(self.num_agents)
        self.gain_values = []
        self.gain_epochs = []
        converged = False
        idx_iter = 1
        for i in range(self.iterations):
            self.one_interaction(agent_ids)
            if ((i>0) and (i%self.time_resolution ==0)):
                print(f"On iteration {i}")
                self.update_ensemble_arrays()

                p_s = self.measure_blind_success()
                gain = ((self.num_meanings*p_s) -1)/(self.num_signals-1)
                if (gain>=0.80) and (converged ==False):
                    print("Steady state reached")
                    self.steady_state_iteration = i
                    converged = True
                self.gain_values.append(gain)
                self.gain_epochs.append(i)

                
                self.save_to_file(idx_iter=idx_iter) 
                idx_iter+=1


        """Need to make the guide for averaging the ensemble arrays which then gets passed to the plotter"""
        

        self.save_params(self.iterations,converged,self.avg_map)
        




if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--NumMeanings', type = int, help="Determines the numbers of meanings to be created/used in the case where meanings are not" \
    "hardcoded", default=16)
    parser.add_argument('--NumSignals', type = int, help="Determines the numbers of signals to be created/used in the case where signals are not" \
    "hardcoded", default=8)
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


    args = parser.parse_args()

    time_res = int(args.TimeRes * 100000)
    total_iterations = int(args.iterations * 1000000)
    
    input_dir = Path(args.OutputDir)/args.ImportFile

    output_directory = Path(args.OutputDir)/args.Filename
    output_directory.mkdir(parents = True, exist_ok = True)#Creates the folder if it does not already exist


    meanings = np.arange(args.NumMeanings)

    signals = np.arange(args.NumSignals)

    agent_func = lambda **kwargs: Agent(meanings,signals,args.lambda_val,args.Alpha,args.Beta,args.NumMeanings,args.NumSignals,**kwargs)

    read_data_ensemble= Imported_Ensemble(input_dir, agent_func, output_directory, time_resolution=time_res, iterations=total_iterations,
                                          new_gamma=args.gamma, passed_focuses=args.PassedFocuses)

    # print(len(read_data_ensemble.agents_list))
    # print(read_data_ensemble.gen_focuses)
    # print(read_data_ensemble.possible_focuses)
    # read_data_ensemble.update_ensemble_arrays()

    read_data_ensemble.add_agents(number_added=4)

    read_data_ensemble.update_ensemble_arrays()

    read_data_ensemble.make_inter_gen_alpha()
    # print(read_data_ensemble.inter_gen_alpha)

    read_data_ensemble.training_loop()
    # read_data_ensemble.print_lambdas()


