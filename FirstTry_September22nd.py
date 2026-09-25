import numpy as np
import scipy as sc
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import numba
from numba import njit, prange




#As a test, starting with 3 meanings, 2 signals
meanings = ["dog",
            "cat",
            "bird",
            "pig"]


signals = ["red",
           "blue"]


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

@njit(parallel = True)
def update_phi(phi_array, counts_array, lambda_val, alpha, meaning_index, signal_index,num_signals):
    """Update rule according to equation 15 in document Evolution of Communications Through Fluctuations"""
    
    new_phi = np.copy(phi_array)
    sum_term =0
    for i in range(num_signals):
        sum_term+=counts_array[i,meaning_index]
    #Sum over all signals observed to be used for meaning m

    for j in prange(num_signals):
        phi_array[j,meaning_index] = (
            phi_array[j,meaning_index] + (delta(signal_index,j=j) -phi_array[j,meaning_index] +
                                          lambda_val*alpha*(1/num_signals -phi_array[j,meaning_index]))/
                                            (1+(1-lambda_val)*sum_term +alpha)
        )
    return phi_array

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

# @njit(parallel = True)
# def calculate_ensemble_count_array(num_agents, ensemble_count_array):
#     """Calculating the ensemble average n(s|m) array"""
#     ensemble_count = np.empty_like(ensemble_count_array[0,:]) #create a blank version of the signal_meaning array to store the new ensemble average
#     for i in prange(num_agents):
#         ensemble_count[:] = ensemble_count_array[i,:].sum()
#         ensemble_count[:,:] = agent_list[i].signal_meaning_array[i,:,:].sum()
#     ensemble_count/=num_agents
#     return ensemble_count
#Don't think above is helpful if I cannot pass the agent list

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
        inst_val = (phi_s_mu[j]*rho_dist[j])/denom
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
    def __init__(self,meanings,signals, lambda_val = 0.01, alpha=0.1, beta = 49):
        self.num_meanings = len(meanings)
        self.num_signals = len(signals)
        self.meanings = np.array(meanings)
        self.signals = np.array(signals)
        self.lambda_val = lambda_val

        self.meanings_list = meanings
        self.signals_list = signals

        #Need some list of integers to pass to the numba functions corresponding to the meanings and signal distributions
        self.meanings_integers = np.arange(self.num_meanings, dtype=int)
        print(self.meanings_integers)
        self.signals_integers = np.arange(self.num_signals,dtype=int)
        print(self.signals_integers)

        self.Alpha = alpha
        self.Alpha_s = alpha/self.num_signals

        self.beta = beta

        self.certainty = 1/(1+beta)

        self.alpha_dist = np.full(self.num_meanings, (beta/self.num_meanings))
        # print(self.alpha_dist)
        
        #Above is to be used for the rho distributions, it has no bearing on the self.Alpha value

        self.signal_meaning_array = np.zeros((self.num_signals,self.num_meanings)) #Initialises a zero array of size S X M
        """The signal meaning array tracks how many times an agent has received signal s when they interpreted meaning m
        
        In our specific case, signal_meaning_array[0,1] corresponds to the number of times the agent has received the signal
        'red', when the meaning they believe the signaller intends is 'cat'"""


#####################################################################################
        #Double check this
        self.phi_array = np.full((self.num_signals,self.num_meanings),fill_value= (1/self.num_signals))
        """At initialisation the phi array is uniformally distributed such that all signals are equally likely"""

    def select_signal_numpy(self):
        """A function to be used when this agent is chosen to select a signal to send"""

        inst_rho = sc.stats.dirichlet.rvs(alpha = self.alpha_dist, size = 1).squeeze()
        ###
        #Double check size = 1 is wanted here
        ###
        # print(f"inst_rho is {inst_rho}")
        #Creates an attentional weight distribution over all the meanings

        selected_mu = np.random.choice(self.meanings, p=inst_rho)
        # print(f"Selected mu is {selected_mu}")

        mu_idx = self.meanings_list.index(selected_mu)

        signal_prob = self.phi_array[:,mu_idx]

        selected_signal = np.random.choice(self.signals,p=signal_prob)
        # print(f"The selected signal is {selected_signal}")
        ###
        #Double check this is the correct way to go about choosing the signal
        ###

        return selected_signal, inst_rho,

    def select_signal_numba(self):
        selected_signal, inst_rho = select_signal_integers_njit(self.alpha_dist,self.phi_array,self.meanings_integers,self.signals_integers)
        return selected_signal,inst_rho

    def receive_signal(self, signal, rho_distribution,A):
        """In future, break this up into two functions, one where we need to re-generate rho, and one where we don't"""
        rho_roll = np.random.rand()
        if rho_roll<=A:
            inst_rho = rho_distribution
        else:
            inst_rho = sc.stats.dirichlet.rvs(alpha = self.alpha_dist, size = 1).squeeze()
            print("Recalculating Rho")


        """Interpretation rule works by having each agent work under the framework of calculating their own likelihood of
        using the inputted signal to convey a meaning"""

        ###
        #I imagine this posterior chain is where numba is really necessary
        ###


        signal_idx = self.signals_list.index(signal)
        # print(f"Received signal idx is {signal_idx}")
        phi_s_mu = self.phi_array[signal_idx,:] #Grabs vector containing the likelihood of using signal corresponding to signal idx for each meaning
        denom = (phi_s_mu * inst_rho).sum()
        # if np.abs(denom-1) >= 1e-3:
        #     print("Signal denominator larger than 1") 

        posterior_dist = np.empty((self.num_meanings,))
        # print(f"Shape of posterio dist is {posterior_dist.shape}")

        for j in range(self.num_meanings):
            posterior_dist[j] = (phi_s_mu[j] * inst_rho[j])/denom

        # print(f"Posterior dist is {posterior_dist}")
        # print(f"Sum of post dist is {np.sum(posterior_dist)}")
        
        nu = np.random.choice(self.meanings, p=posterior_dist)
        # print(f"Selected meaning is {nu}")
        return nu, signal_idx

    def receive_signal_numba(self,signal_idx,rho_dist,A):
        nu_idx = receive_signal_integers_njit(signal_idx,A,rho_dist,self.alpha_dist,self.phi_array,self.meanings_integers,self.num_meanings)
        return nu_idx
    def update_counts(self, nu_idx, signal_idx):
        """Update rule is that all values in the signal_meaning_array decay by (1-lambda)*current_value, 
        with the exception of the actual signal, which while it does decay, is also incremented by 1. """
        # print("phi array before")
        # print(self.phi_array)
        #Now updating the posterior distribution
        # nu_idx = self.meanings_list.index(interpreted_nu)

        self.phi_array = update_phi(self.phi_array,self.signal_meaning_array,self.lambda_val,
                                    self.Alpha,nu_idx,signal_idx,self.num_signals)
        # print("phi array after")
        # print(self.phi_array)

        #Now updating the counts array
        # print("Counts array before")
        # print(self.signal_meaning_array)
        self.signal_meaning_array = update_signal_meaning(self.signal_meaning_array,self.lambda_val,
                                                          nu_idx,signal_idx,self.num_signals)
        # print("Counts after")
        # print(self.signal_meaning_array)
        
        


class Ensemble:
    """Create a class responsible for the ensemble of agents, such that measuring and analyzing communicative values are easier."""
    def __init__(self, num_agents, agent_fn, alignment, number_signals, number_meanings):
        self.A = alignment
        self.agent_ids = np.arange(num_agents)
        self.number_agents = num_agents
        self.num_signals = number_signals
        self.num_meanings = number_meanings

        self.agent_list = [agent_fn() for _ in range(num_agents)]
        #Creates the initial list of agents
        self.ensemble_phi = np.empty((self.number_agents,self.num_signals,self.num_meanings))
        self.ensemble_counts = np.empty_like(self.ensemble_phi)
        self.update_ensemble_arrays()


        self.numba_warmup()
        self.gain_epochs = []
        self.gain_values = []



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
        update_phi(selected_agent.phi_array,selected_agent.signal_meaning_array,selected_agent.lambda_val,
                   selected_agent.Alpha,1,1,selected_agent.num_signals)

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
        plt.savefig(fr"C:/Users/Logan/Downloads/MPhys/Plots/ensemble_counts_lambda{lambda_val}.png")
        plt.close()

    def plot_communication_gain(self):
        lambda_val = self.agent_list[0].lambda_val
        fig,ax = plt.subplots(figsize = (10,6))
        ax.plot(self.gain_epochs, self.gain_values)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Communication Gain")
        ax.set_title(f"Communication Gain vs Time for $\lambda$ = {lambda_val}")
        plt.savefig(fr"C:/Users/Logan/Downloads/MPhys/Plots/Gain_{lambda_val}.png")
        plt.close()

    def training_loop(self,iterations):
        for i in range(iterations):
            self.one_interaction()
            if ((i>0) and (i%50000 ==0)):
                print(f"On iteration {i}")
                self.update_ensemble_arrays()
                
                p_s = self.measure_blind_success()
                gain = ((self.num_meanings*p_s) -1)/(self.num_signals-1)
                self.gain_values.append(gain)
                self.gain_epochs.append(i)

        ensemble_average_counts = np.mean(self.ensemble_counts, axis = 0)
        self.plot_communication_gain()
        self.plot_ensemble_counts(ensemble_avg=ensemble_average_counts)
        



if __name__ =="__main__":



    # send_agent = Agent(meanings=meanings,signals=signals,lambda_val=0.001)
    # receive_agent = Agent(meanings=meanings,signals=signals,lambda_val=0.001)
    num_signals = len(signals)
    num_meanings = len(meanings)
    #Initial compiling of numba functions  
    agent_func = lambda : Agent(meanings,signals, lambda_val=0.1)

    Test_ensemble = Ensemble(5,agent_func,1,num_signals,num_meanings)
    Test_ensemble.training_loop(iterations=3000000)
    # Test_ensemble.one_interaction()
    
    #Testing numpy dirichlet dist
    # beta = 49
    # alpha_dist = np.full(num_meanings, (beta/num_meanings))
    # inst_dirichlet = np.random.dirichlet(alpha_dist)
    # print(inst_dirichlet)


    # prob_dist_test = np.array([0.35,0.5,0.15])
    # array_test = np.array([1,2,3])
    # warmup = numba_random_choice(array_test,prob_dist_test)
    # counts_array = np.zeros((3,))
    # for i in range(100000000):
    #     randomdraw = numba_random_choice(array_test,prob_dist_test)
    #     counts_array[randomdraw-1]+=1
    # print(counts_array)


    




    


