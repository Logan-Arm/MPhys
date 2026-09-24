import numpy as np
import scipy as sc
import scipy.stats
import numba
from numba import njit, prange




#As a test, starting with 3 meanings, 2 signals
meanings = ["dog",
            "cat",
            "bird"]


signals = ["red",
           "blue"]


###
#Numba Section
###

@njit
def numba_col_sum(array,mu_index):
    total = 0.0
    for i in range(array.shape[0]):
        total+=array[i,mu_index]
    return total

@njit(parallel = True)
def phi_array(num_sig_meaning_array, alpha, s):
    alpha_on_s = alpha/s
    phi = np.empty_like((num_sig_meaning_array))
    num_meanings = num_sig_meaning_array.shape[1]
    for i in prange(num_meanings):
        pass
        

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

    def generate_phi(self,mu_idx): #NOW OUTDATED
        """Function to be used to generate an agents phi(s|m), or their posterior predictive distribution, from a given mu
        Specifically used in signal generation, and measures of communication gain
        """

        ni_s_m_column = self.signal_meaning_array[:,mu_idx]
        sum_term = ni_s_m_column.sum() #Using the .sum() is faster than using np.sum by ~2.5X, still slower than numba by ~2x, so this provides potential speedup later
        signal_prob = (ni_s_m_column + self.Alpha_s)/(sum_term + self.Alpha)
        print(f"Signal probability is {signal_prob}")
        return signal_prob

    def select_signal(self):
        """A function to be used when this agent is chosen to select a signal to send"""

        inst_rho = sc.stats.dirichlet.rvs(alpha = self.alpha_dist, size = 1).squeeze()
        ###
        #Double check size = 1 is wanted here
        ###
        print(f"inst_rho is {inst_rho}")
        #Creates an attentional weight distribution over all the meanings

        selected_mu = np.random.choice(self.meanings, p=inst_rho)
        print(f"Selected mu is {selected_mu}")

        mu_idx = self.meanings_list.index(selected_mu)

        signal_prob = self.phi_array[:,mu_idx]

        selected_signal = np.random.choice(self.signals,p=signal_prob)
        print(f"The selected signal is {selected_signal}")
        ###
        #Double check this is the correct way to go about choosing the signal
        ###

        return selected_signal, inst_rho,

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
        print(f"Received signal idx is {signal_idx}")
        phi_s_mu = self.phi_array[signal_idx,:] #Grabs vector containing the likelihood of using signal corresponding to signal idx for each meaning
        denom = (phi_s_mu * inst_rho).sum()
        if np.abs(denom-1) >= 1e-3:
            print("Signal denominator larger than 1") 

        posterior_dist = np.empty((self.num_meanings,))
        print(f"Shape of posterio dist is {posterior_dist.shape}")

        for j in range(self.num_meanings):
            posterior_dist[j] = (phi_s_mu[j] * inst_rho[j])/denom

        print(f"Posterior dist is {posterior_dist}")
        print(f"Sum of post dist is {np.sum(posterior_dist)}")
        
        nu = np.random.choice(self.meanings, p=posterior_dist)
        print(f"Selected meaning is {nu}")
        return nu, signal_idx

    def update_counts(self, interpreted_nu, signal_idx):
        """Update rule is that all values in the signal_meaning_array decay by (1-lambda)*current_value, 
        with the exception of the actual signal, which while it does decay, is also incremented by 1. """
        print("phi array before")
        print(self.phi_array)
        #Now updating the posterior distribution
        nu_idx = self.meanings_list.index(interpreted_nu)
        self.phi_array = update_phi(self.phi_array,self.signal_meaning_array,self.lambda_val,
                                    self.Alpha,nu_idx,signal_idx,self.num_signals)
        print("phi array after")
        print(self.phi_array)

        #Now updating the counts array
        print("Counts array before")
        print(self.signal_meaning_array)
        self.signal_meaning_array = update_signal_meaning(self.signal_meaning_array,self.lambda_val,
                                                          nu_idx,signal_idx,self.num_signals)
        print("Counts after")
        print(self.signal_meaning_array)
        
        



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
        self.numba_warmup()
        self.agent_index_array = np.ones((num_agents,num_agents))
        #Store an array of num_agents by num_agents, this allows us to vectorise over num_agents for interacting pair calculations (I think)
    def numba_warmup(self):
        print("Initialising Numba Functions")
        selected_agent = self.agent_list[0]
        delta(1,1)
        update_phi(selected_agent.phi_array,selected_agent.signal_meaning_array,selected_agent.lambda_val,
                   selected_agent.Alpha,1,1,selected_agent.num_signals)
    
        update_signal_meaning(selected_agent.signal_meaning_array,selected_agent.lambda_val,1,1,selected_agent.num_signals)  

    def one_interaction(self):
        selected_agents = np.random.choice(self.agent_ids,size=2, replace=False)
        signaller_agent = self.agent_list[selected_agents[0]]
        receiver_agent = self.agent_list[selected_agents[1]]

        selcted_signal,inst_rho = self.agent_list[signaller_agent].select_signal()

        nu, signal_idx = receiver_agent.receive_signal(selcted_signal,inst_rho,self.A)

        receiver_agent.update_counts(nu,signal_idx)


        

    def measure_blind_success(self):
        """Sum over all pairs and then subtract the diagonal elements"""
        phi_tensor = create_phi_tensor(self.number_agents,self.agent_list,self.num_signals,self.num_meanings)
        
        pass



if __name__ =="__main__":



    # send_agent = Agent(meanings=meanings,signals=signals,lambda_val=0.001)
    # receive_agent = Agent(meanings=meanings,signals=signals,lambda_val=0.001)

    #Initial compiling of numba functions
    delta(1,1)
    update_phi(send_agent.phi_array,send_agent.signal_meaning_array,send_agent.lambda_val,
               send_agent.Alpha,1,1,send_agent.num_signals)

    update_signal_meaning(send_agent.signal_meaning_array,send_agent.lambda_val,1,1,send_agent.num_signals)
    
    selcted_signal, inst_rho = send_agent.select_signal()
    print('########################################')
    nu, signal_idx = receive_agent.receive_signal(selcted_signal,inst_rho, A=1)
    print('########################################')
    receive_agent.update_counts(nu,signal_idx)

    # agents = [Agent(meanings=meanings,signals=signals,lambda_val=0.001) for _ in range(2)]

    



    




    


