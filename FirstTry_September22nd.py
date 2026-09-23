import numpy as np
import scipy as sc
import scipy.stats

"""This is going to be a baseline reading, as such, I am not going to be implementing any memory loss yet"""


#As a test, starting with 3 meanings, 2 signals
meanings = ["dog",
            "cat",
            "bird"]


signals = ["red",
           "blue"]



class Agent:
    """The agent parent class will be able to act as both the signaller and the receiver, as such it needs to have the set of meanings
    and signals available to it, as well as it's own attentional weight distribution.
    Following the instructions put out in the "Evolution of communication through fluctuations" background reading, the signals (S) and 
    meanings (M) form a matrix, this matrix is initialised to be empty.

    Additionally, the weight array will be initialised to be constant, I am not quite sure how to introduce the variation as a result of 
    certainty into it at this moment
    
    """
    def __init__(self,meanings,signals, forget_rate = 0.01, alpha=0.1, beta = 49):
        self.num_meanings = len(meanings)
        self.num_signals = len(signals)
        self.meanings = np.array(meanings)
        self.signals = np.array(signals)
        self.forget_rate = forget_rate

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

    def generate_phi(self,mu_idx):
        """Function to be used to generate an agents phi(s|m), or their posterior predictive distribution, from a given mu
        Specifically used in signal generation, and measures of communication gain
        """
        ni_s_m_column = self.signal_meaning_array[:,mu_idx]
        sum_term = np.sum(ni_s_m_column)

        signal_prob = (ni_s_m_column + self.Alpha_s)/(sum_term + self.Alpha)
        print(f"Signal probability is {signal_prob}")
        

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
        print(f"Mu idx is {mu_idx}")

        ni_s_m_column = self.signal_meaning_array[:,mu_idx]
        sum_term = np.sum(ni_s_m_column)

        signal_prob = (ni_s_m_column + self.Alpha_s)/(sum_term + self.Alpha)
        print(f"Signal probability is {signal_prob}")


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


        phi_j_matrix = np.empty_like(self.signal_meaning_array)
        for i in range(self.num_meanings):
            phi_j_matrix[:,i]= ((self.signal_meaning_array[:,i] + self.Alpha_s)/
                            (np.sum(self.signal_meaning_array[:,i]) +self.Alpha)
            )
        """Need to change above, we are only concerned with the probabilities relating to the signal we already have"""

        signal_idx = self.signals_list.index(signal)
        print(f"Received signal idx is {signal_idx}")
        phi_s_mu = phi_j_matrix[signal_idx,:] #Grabs vector containing the likelihood of using signal corresponding to signal idx for each meaning
        denom = np.sum(phi_s_mu * inst_rho)

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
        print("signal meaning array before")
        print(self.signal_meaning_array)

        nu_idx = self.meanings_list.index(interpreted_nu)
        self.signal_meaning_array = (1-self.forget_rate)*self.signal_meaning_array #Decay 
        self.signal_meaning_array[signal_idx,nu_idx]+=1
        print("signal meaning array after")
        print(self.signal_meaning_array)



class Ensemble:
    """Create a class responsible for the ensemble of agents, such that measuring and analyzing communicative values are easier."""
    def __init__(self, num_agents, agent_fn):

        self.agent_list = [agent_fn() for _ in range(num_agents)]
        #Creates the initial list of agents
        
    def 



if __name__ =="__main__":
    # send_agent = Agent(meanings=meanings,signals=signals,forget_rate=0.001)
    # receive_agent = Agent(meanings=meanings,signals=signals,forget_rate=0.001)
    # selcted_signal, inst_rho = send_agent.select_signal()
    # print('########################################')
    # nu, signal_idx, = receive_agent.receive_signal(selcted_signal,inst_rho, A=1)
    # print('########################################')
    # receive_agent.update_counts(nu,signal_idx)

    agents = [Agent(meanings=meanings,signals=signals,forget_rate=0.001) for _ in range(2)]




    




    


