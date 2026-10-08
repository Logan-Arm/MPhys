import numpy as np
import scipy as sc
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import numba
from numba import njit, prange
import argparse
from pathlib import Path
import pandas as pd
import json
import h5py







"""Oftentimes I am passing variables like num_meanings and num_signals; even though these could be recalculated in one line, 
I still believe it is faster to just pass the variable"""

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
def create_phi_tensor(num_agents, agent_list,num_signals,num_meanings):
    """Compiles all the phi arrays into a singular tensor, to make calculating gain values easier"""
    phi_tensor = np.empty((num_agents,num_signals,num_meanings))
    for i in range(num_agents):
        for j in prange(num_signals):
            for k in prange(num_meanings):
                phi_tensor[i,j,k] = agent_list[i].phi_array[j,k]

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
    """A numba version of the signal receiving and interpretation process"""
    rho_roll = np.random.rand()
    if rho_roll<=A:
        rho_dist = passed_rho_dist
    else:
        rho_dist = np.random.dirichlet(alpha=alpha_dist)
    # print("Rho dist")
    # print(rho_dist)
    
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
    def __init__(self,meanings,signals, lambda_val, alpha, beta,num_meanings, num_signals,passed_counts = None, generation_tag =0):
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
    def update_counts(self, nu_idx, signal_idx):
        """Update rule is that all values in the signal_meaning_array decay by (1-lambda)*current_value, 
        with the exception of the actual signal, which while it does decay, is also incremented by 1. """

        self.signal_meaning_array = update_signal_meaning(self.signal_meaning_array,self.lambda_val,
                                                          nu_idx,signal_idx,self.num_signals)

        """Phi is calcualted based upon the counts array, so updating phi must come after we update the signal meaning array"""
        new_phi_row = update_phi(self.signal_meaning_array,self.Alpha,self.Alpha_s,signal_idx,self.num_meanings)
        self.phi_array[signal_idx,:] = new_phi_row
        


class Ensemble:
    """Create a class responsible for the ensemble of agents, such that measuring and analyzing communicative values are easier."""
    def __init__(self, num_agents, agent_fn, alignment, number_signals, number_meanings, savefig,output_dir,showfig, 
                 generation_count_dists= False, num_generations = 1, beta = 49, gamma = 2, generation_probs = None, number_focuses = 2,
                 generation_overlap = False):
        self.A = alignment
        self.agent_ids = np.arange(num_agents)
        self.number_agents = num_agents
        self.num_signals = number_signals
        self.num_meanings = number_meanings
        self.savefig = savefig
        self.show = showfig
        self.output_dir = Path(output_dir)


        if generation_count_dists:
            """Currently this is only configured such that we take 1 generation to start with, once a steady state is reached,
            more generations will be formed, although this will of course not occur in the init"""
            self.possible_focuses = np.arange(self.num_meanings).tolist()
            self.has_generations=True
            self.number_of_gens = num_generations
            self.gamma=gamma
            self.gen_probs = generation_probs
            generation_counts,discard= self.create_generation_counts(beta=beta,gamma=gamma,possible_focuses=self.possible_focuses,number_focuses=number_focuses)

            self.agent_list = [agent_fn(passed_counts = generation_counts) for _ in range(num_agents)]


        else:
            self.has_generations=False
            self.number_of_gens = 0
            self.gamma = 'N/A'
            self.gen_probs = 'N/A'
            self.agent_list = [agent_fn() for _ in range(num_agents)]
        #Creates the initial list of agents
        self.ensemble_phi = np.empty((self.number_agents,self.num_signals,self.num_meanings))
        self.ensemble_counts = np.empty_like(self.ensemble_phi)
        self.update_ensemble_arrays()


        self.numba_warmup()
        self.gain_epochs = []
        self.gain_values = []

    def create_counts_generations(self,num_generations,beta,gamma, number_focuses = 2, unique_focuses = False):
        """A function to create the rho distributions for each generation at an ensemble level, to then distribute to each agent on 
        a probabilistic basis.  """

        meanings = np.arange(self.num_meanings).tolist()
        generation_attentional_counts_array = np.empty((num_generations,self.num_meanings)) #Dimensionality [number of generation, rho array]
        for i in range(num_generations):
            selected_focuses = np.random.choice(meanings,size =number_focuses, replace = False)
            
            
            if unique_focuses:
                """If we want unique focuses, remove the selected focuses from the list of possible meanings to choose from"""
                for j in range(number_focuses):
                    meanings.remove(selected_focuses[j])


            print(f"Focuses of generation {i} are {selected_focuses}")
            base_counts = np.full(self.num_meanings, (beta/self.num_meanings))
            for focus in selected_focuses:
                base_counts[focus]*=gamma
            generation_attentional_counts_array[i,:] = base_counts

        return generation_attentional_counts_array

    def create_generation_counts(self,beta,gamma, possible_focuses,number_focuses =2):
        """A function to return attentional counts, modified by a select number of attentional focuses, for a generation.  
        Focuses are selected randomly from an inputted list of allowed focuses.  The degree to which these counts are modifed
        ('focused on') comes from the parameter gamma, which is a straighforward multiplier of the number of counts.  All meanings
        not within the group of selected focuses have a count value corresponding to beta/M where M is total number of meanings.
        """
        selected_focuses = np.random.choice(possible_focuses,size = number_focuses, replace =False)
        gen_counts = np.full(self.num_meanings,(beta/self.num_meanings))
        for focus in selected_focuses:
            gen_counts[focus] *=gamma
        # print(f"Gen_counts has datatype {gen_counts.dtype}")
        return gen_counts, selected_focuses

    def plot_save_or_show(self, fig, name):
        """A helper function to be called after creating a figure in any of the below functions.  Determines what to do with the created figure"""
        if self.savefig:
            fig.savefig(self.output_dir / f"{name}.png")
        if self.show:
            plt.show()
        plt.close(fig)

    def update_ensemble_arrays(self):
        """Numba cannot be passed an agent_type at all, as such, we need to create tensors for the ensemble average values,
        i.e., values we would otherwise obtain in a numba function from the objects passed."""
        for i in range(self.number_agents):
            self.ensemble_phi[i,:] = self.agent_list[i].phi_array
            self.ensemble_counts[i,:] = self.agent_list[i].signal_meaning_array
    
    def numba_warmup(self):
        test_array = np.array([0,1])

        print("Initialising Numba Functions")
        selected_agent = self.agent_list[0]
        print("Testing delta")
        delta(1,1)
        print("Testing update_phi")
        update_phi(selected_agent.signal_meaning_array,
                   selected_agent.Alpha,selected_agent.Alpha_s,1,selected_agent.num_meanings)

        print("Testing update_signal meaning")
        update_signal_meaning(selected_agent.signal_meaning_array,selected_agent.lambda_val,1,1,selected_agent.num_signals)  

        print("Testing numba_random_choice")
        numba_random_choice(test_array,test_array)

        # print("Testing select_signal_njit")
        # select_signal_njit(selected_agent.alpha_dist,selected_agent.signals,selected_agent.meanings,
        #                    selected_agent.phi_array,selected_agent.meanings_list)
        

        print("Testing select_signal_integers_njit")
        discard, rho_dist = select_signal_integers_njit(selected_agent.alpha_dist,selected_agent.phi_array,
                                    selected_agent.meanings_integers,selected_agent.signals_integers)

        print("Testing receive_signal_integers_njit")
        receive_signal_integers_njit(1,1,rho_dist,test_array,selected_agent.phi_array,selected_agent.meanings_integers,
                                     selected_agent.num_meanings)
        
        # print("Test create_phi_tensor")
        # create_phi_tensor(1,self.agent_list,self.num_signals,self.num_meanings)
        print("Test blind success inner loop")
        blind_success_inner_loop(self.agent_list[0].phi_array,self.agent_list[1].phi_array,self.num_signals,self.num_meanings)
        print("Test blind success outer loop")
        blind_success_outer_loop(self.number_agents,self.ensemble_phi,self.num_signals,self.num_meanings)

    def one_interaction(self):
        selected_agents = np.random.choice(self.agent_ids,size=2, replace=False)
        signaller_agent = self.agent_list[selected_agents[0]]
        receiver_agent = self.agent_list[selected_agents[1]]
        # print("Agents selected")


        selcted_signal_idx,inst_rho = signaller_agent.select_signal_numba()
        # print("Signal and rho sent")
        # print(selcted_signal_idx,inst_rho)

        nu_idx = receiver_agent.receive_signal_numba(selcted_signal_idx,inst_rho,self.A)
        # print("Signal received")
        # print(nu_idx)

        receiver_agent.update_counts(nu_idx,selcted_signal_idx)
        # print("counts updated")

    def measure_blind_success(self):
        """Sum over all pairs and then subtract the diagonal elements"""
        p_s = blind_success_outer_loop(self.number_agents,self.ensemble_phi,self.num_signals,self.num_meanings)
        return p_s

    def plot_ensemble_counts(self,ensemble_avg):
        """Plots the ensemble average signal/meaning count array"""
        lambda_val = self.agent_list[0].lambda_val
        fig,ax = plt.subplots(figsize = (10,6))
        sns.heatmap(ensemble_avg,cmap = 'coolwarm')
        ax.set_xlabel("Meanings")
        ax.set_ylabel("Signals")
        ax.set_title(f"Counts of Signals vs Meaning for $\lambda$ = {lambda_val}")
        name = f"Ensemble_Counts_Lambda{lambda_val}.png"
        self.plot_save_or_show(fig,name)
        # plt.savefig(fr"C:/Users/Logan/Downloads/MPhys/Plots/ensemble_counts_lambda{lambda_val}.png")

    def plot_communication_gain(self):
        lambda_val = self.agent_list[0].lambda_val
        fig,ax = plt.subplots(figsize = (10,6))
        ax.plot(self.gain_epochs, self.gain_values)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Communication Gain")
        ax.set_title(f"Communication Gain vs Time for $\lambda$ = {lambda_val}")
        name = f"Communication_Gain_lambda{lambda_val}"
        self.plot_save_or_show(fig,name)

    def training_loop(self,iterations):

        self.update_ensemble_arrays()
        #Initialise the files to save the data in
        self.generate_initial_h5_file()


        converged = False
        for i in range(iterations):
            self.one_interaction()
            if ((i>0) and (i%100000 ==0)):
                print(f"On iteration {i}")
                self.update_ensemble_arrays()

                if i== 2500000:
                    self.test_phi_h5 = self.ensemble_phi.mean(axis = 0)
                    self.test_counts_h5 = self.ensemble_counts.mean(axis = 0)
                    self.h5_test_i = i//100000
                    
                p_s = self.measure_blind_success()
                gain = ((self.num_meanings*p_s) -1)/(self.num_signals-1)
                if (gain>=0.80) and (converged ==False):
                    print("Steady state reached")
                    self.steady_state_iteration = i
                    converged = True
                self.gain_values.append(gain)
                self.gain_epochs.append(i)

                self.save_to_file(iteration= i) 

        ensemble_average_counts = np.mean(self.ensemble_counts, axis = 0)
        # self.plot_communication_gain()
        # self.plot_ensemble_counts(ensemble_avg=ensemble_average_counts)
        if converged:
            print(f"Steady state iteration is {self.steady_state_iteration}")
        self.save_params(iterations,converged)
        
    def save_params(self,iterations,convergence):
        params_dict = {'num_meanings': self.num_meanings,
                       'num_signals': self.num_signals,
                       'num_agents': self.number_agents,
                       'lambda':self.agent_list[0].lambda_val,
                        'alpha': self.agent_list[0].Alpha,
                        'beta': self.agent_list[0].beta,
                        'gamma':self.gamma,
                        'generation_probs':self.gen_probs,
                        'alignment':self.A,
                        'certainty':self.agent_list[0].certainty,
                        'iterations':iterations,
                        'Steady_State':convergence,
                        'number_of_gens':self.number_of_gens}
        params_df = pd.DataFrame([params_dict])
        params_file = self.output_dir / "simulation_parameters.csv"
        params_df.to_csv(params_file, index=False)
        print(f"Saved simulation parameters to {params_file}")        

    def add_new_gens(self,number_of_added_members, delete_old = 0):
        """A function to add members of a new generation to the agent list, as well as delete older members """
        

    def save_to_file(self, iteration: int = None):
        """Save the experimental data to a file as it is calculated/experiment is performed"""
        save_dir = self.output_dir/"raw"
        save_dir.mkdir(parents = True, exist_ok = True)


        meta_data = {
            "num_agents": len(self.agent_list),
            "num_signals": self.num_signals,
            "num_meanings": self.num_meanings,
            "lambda_val": self.agent_list[0].lambda_val
        }
        with open(save_dir/"metadata.json","w") as f:
            json.dump(meta_data,f,indent=4)


        if hasattr(self,"gain_values"):
            pd.DataFrame({"epoch": self.gain_epochs, "gain": self.gain_values}).to_csv(save_dir / "gain_values.csv",
                                                                                        index=False)

        if iteration is not None:
            """If iteration is not none, we append the data we found to the h5py file we initialised earlier"""
            current_phi_iter_val = self.h5_phi_dataset.shape[0]
            current_counts_iter_val =self.h5_signal_meanings_dataset.shape[0]

            self.h5_phi_dataset.resize(current_phi_iter_val+1,axis=0)
            self.h5_signal_meanings_dataset.resize(current_counts_iter_val+1,axis=0)
            self.h5_phi_dataset[current_phi_iter_val,:,:] = self.ensemble_phi.mean(axis = 0)
            self.h5_signal_meanings_dataset[current_counts_iter_val,:,:] = self.ensemble_counts.mean(axis = 0)
   
    def generate_initial_h5_file(self):
        """Generates the h5_file used to store data as the experiment runs, should only be 
        used after the initial ensemble arrays have been calculated"""
        save_dir = self.output_dir/"raw"
        save_dir.mkdir(parents = True, exist_ok = True)
        self.h5_file = h5py.File(name =save_dir/"array_data.h5", mode = 'w')
        self.h5_phi_group = self.h5_file.create_group(name = 'Phi')
        self.h5_phi_dataset = self.h5_phi_group.create_dataset(name = 'phi_dataset', shape = (1, self.num_signals, self.num_meanings), 
                                                     maxshape = (None, self.num_signals, self.num_meanings),dtype = np.float64, chunks = True)
        self.h5_phi_dataset.attrs['description'] = 'Ensemble phi arrays over time'
        self.h5_phi_dataset.attrs['num_signals'] = self.num_signals
        self.h5_phi_dataset.attrs['num_meanings'] = self.num_meanings

        self.h5_phi_dataset[0,:,:] = self.ensemble_phi.mean(axis =0)


        self.h5_signal_meanings_group = self.h5_file.create_group(name = 'signal_meaning')
        self.h5_signal_meanings_dataset = self.h5_signal_meanings_group.create_dataset(name = 'signal_meanings_dataset', shape = (1, self.num_signals, self.num_meanings), 
                                                     maxshape = (None, self.num_signals, self.num_meanings),dtype = np.float64, chunks = True)
        self.h5_signal_meanings_dataset.attrs['description'] = "Ensemble signal meaning count array over time "
        self.h5_signal_meanings_dataset.attrs['num_signals'] = self.num_signals
        self.h5_signal_meanings_dataset.attrs['num_meanings'] = self.num_meanings

        self.h5_signal_meanings_dataset[0,:,:] = self.ensemble_counts.mean(axis =0)


    def test_h5_file(self):
        file_dir = self.output_dir/"raw"
        with h5py.File(file_dir/"array_data.h5", 'r') as f:
            phi_group = f['Phi']
            counts_group = f['signal_meaning']

            phi_data = phi_group['phi_dataset'][:]
            counts_data = counts_group["signal_meanings_dataset"][:]

        test_recorded_phi = phi_data[self.h5_test_i,:,:]
        test_recorded_counts = counts_data[self.h5_test_i,:,:]

        difference_phi = test_recorded_phi - self.test_phi_h5
        difference_counts = test_recorded_counts-self.test_counts_h5

        print(f"Maximum phi difference is {np.max(difference_phi)}")
        print(f"Maximum counts difference is {np.max(difference_counts)}")


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--NumMeanings', type = int, help="Determines the numbers of meanings to be created/used in the case where meanings are not" \
    "hardcoded", default=12)
    parser.add_argument('--NumSignals', type = int, help="Determines the numbers of signals to be created/used in the case where signals are not" \
    "hardcoded", default=6)
    parser.add_argument('--NumAgents', type = int, help="Determines the numbers of communicating/learning agents", default=5)
    parser.add_argument('--Beta', type = int, help="Determines the value of parameter beta", default=49)
    parser.add_argument('--Alpha', type = float, help="Determines the value of parameter alpha", default=0.1)
    parser.add_argument('--lambda_val', type = float, help="Determines the value of lambda, or the forgetting rate of the models", default=0.01)
    parser.add_argument('--alignment', type = float, help="Determines the alignment between models", default=1.0)
    parser.add_argument('--iterations', type=float, help = "Determines how many millions of iterations to run the code for", default = 4)
    parser.add_argument('--OutputDir', type = str, help = 'Determines the file output of saved plots, data, etc', 
                            default =str(Path.home()/"Downloads"/"MPhys/Code_Runs"))
    parser.add_argument('--Filename', type = str, help='Determines the file to save data to', default= 'Unsorted')
    parser.add_argument('--SaveFig', action='store_true', help='If set, saves figures to args.Filename')
    parser.add_argument('--ShowFig', action='store_true', help='If set, shows figures')
    parser.add_argument('--Generations', action='store_true', help='If set, have independent generations rho distributions')
    parser.add_argument('--GenCount', type = int, help="Determines the number of generations", default=1)
    parser.add_argument('--gamma', type = int, help="Determines the parameter gamma, which controls \
                        how much the generational focus is multiplied by", default=2)
    parser.add_argument('--Focuses', type = int, help="Determines how many focuses are in each generation", default=2)

    args = parser.parse_args()

    total_iterations = int(args.iterations * 1000000)
    
    output_directory = Path(args.OutputDir)/args.Filename
    output_directory.mkdir(parents = True, exist_ok = True)#Creates the folder if it does not already exist



    """Initialise the meanings and signals list if they have not been done"""
    meanings = np.arange(args.NumMeanings)

    signals = np.arange(args.NumSignals)


    gen_probs = [0.50,0.50]
      
    agent_func = lambda **kwargs: Agent(meanings,signals,args.lambda_val,args.Alpha,args.Beta,args.NumMeanings,args.NumSignals,**kwargs)
    #We add the **kwargs so that we can pass the function a predetermined rho if necessary

    Test_ensemble = Ensemble(args.NumAgents,agent_func,args.alignment,args.NumSignals,args.NumMeanings,args.SaveFig,
                             output_directory,args.ShowFig,args.Generations,generation_probs=gen_probs,number_focuses=args.Focuses)
    Test_ensemble.training_loop(iterations=total_iterations)
    # Test_ensemble.test_h5_file()
    # Test_ensemble.one_interaction()
    

    




    


