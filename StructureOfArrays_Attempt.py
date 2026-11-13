import argparse
import ast
import json
import time
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numba import njit, types
from numba.typed import Dict

###
# namedtuple definitions
###
SimulationState = namedtuple("SimulationState", ["phi", "counts"])
"""Namedtuple containing the simulation state (the phi and counts arrays).  Implemented primarily in the numba functions
so as to prevent any problems/errors in accessing the returned tuples.
\\
Two arguments:
phi,
counts
"""
SpeakerInfo = namedtuple("SpeakerInfo", ["gens", "lambda_val", "align"])
"""Collects all the speaker information into one namedtuple
\\
3 arguments:
\\
speaker_gens,
\\
speaker_lambda,
\\
speaker_align
"""

NumbaConsts = namedtuple("NumbaConsts", ["N", "M", "S", "alpha_val", "alpha_on_S"])
"""A namedtuple containing all the key constants we pass to the numba functions continuously.  This is done to avoid having to repeatedly write 
all the arguments of the numba functions over and over again, as well as to simply organize all the constants easily
\\
5 arguments:
\\ 
N (number agents)
\\ 
M
\\ 
S
\\ 
alpha_val
\\ 
alpha_on_s

"""

NumbaBuffers = namedtuple("NumbaBuffers", ["dirichlet", "signal", "meaning"])
"""A namedtuple containing buffer arrays (re-written at each interaction to prevent memory reallocation at computation steps) necessary for the 
numba functions

I have separated the dirichlet and meaning buffers intentionally, so that if there is ever a computation that requires both simultaneously,
we are not invalidating data as we progress.
"""

###
# Numba Section
###


###############################################################
"""Start of numba helper function section.  Just contains helper functions which have been 
made because either their standard NumPy equivalent is not supported in numba, or because 
it would be space inefficient to repeatedly write out."""


@njit(cache = True)
def numba_col_sum(array, mu_index):
    total = 0.0
    for i in range(array.shape[0]):
        total += array[i, mu_index]
    return total


@njit(cache = True)
def delta(i, j):
    """Just a standard dirac delta function to be used in most of the update rules, made in
    numba so we do not have to repeatedly leave and re-enter nonpython mode, which would
    otherwise be a slowdown"""
    return 1 if i == j else 0


@njit(cache = True)
def numba_random_choice(array:np.ndarray, prob:np.ndarray):
    """Numba does not accept the probability distribution for the np.random.choice() function, as such we need a workaround to be able to handle
    this efficiently.  This function was taken from the numba support issue 2539, specifically from commentor Mike Fenton

    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    #np.searchsorted(a,v) finds indices where the value v (since np.cumsum returns a scaler) should be inserted to maintain an order
    #The order is specified by the 'side' arg, 'right' returns the last index to satisfy the following:
    # a[i-1]<=v<a[i] 
    #This is very good for large arrays, but not great on small arrays, as the overhead in allocating a new array from np.cumsum 
    # may surpass the actual computation time

    return array[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@njit(cache = True)
def numba_random_choice_linear(array:np.ndarray, prob:np.ndarray):
    """A more explicit, linear search system to recreate np.random.choice in numba.  The previous iteration (from Mike Fenton in numba support issue
    2539) is very good for large systems; however, for smaller systems (like the ones typically working with on this project), the heap strain from
    the instantiation of a new array from np.cumsum can match or surpass the actual work required.  Hence the need for a linear search.

    Will use a similar process as before, i.e., 'right' sampling
    a[i-1]<=p<a[i] 

    
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    rand_roll = np.random.random()
    A = array.shape[0]

    cumulative_prob = 0.0
    for i in range(A):
        cumulative_prob+=prob[i] #cumulative probability at point i in arr
        if rand_roll>= cumulative_prob:
            #if our random roll is >= the cumulative prob we continue iterating as described by the 'right' rules
            continue
        else:
            #if our random roll < cumulative prob, then this is the insertion index, and we return the value of the array with this index
            return array[i]

    #It should not be possible for the random roll to pass through the entirety of the search, as by design random.random() returns on half open
    #interval [0.0,1.0), and by construction the dirichlet distribution sums to 1. However, adding a safety precaution to simply return the last 
    # value of the array in this case is harmless and protects against potential floating point imprecision
    return array[-1] #Returning last component 


@njit(cache = True)
def random_choice_test(upper, amount):
    for _ in range(amount):
        print(np.random.choice(upper, size=2))


"""End of the numba helper function section"""
###############################################################
"""Start of the core mechanics of the model.  The following section contains all the key 
interaction rules: signal selection, signal interpretation, and update of own communication 
structure.  

It should be explicitly noted that very little of these functions are parallel, this is intentional.
At the scale these matrices tend to be working on (no more than ~15-20 signals or meanings), the overhead
of assigning cpu cores for the parallelisation is slower than the pure jit runtime.  I can say this confidently
as this was actually parallelised before, and it ran remarkably slower.  

Furthermore, very little numpy functions are used throughout. This is also intentional.  Most numpy functions 
when ran in the numba interpretor will have a small unpacking/overhead cost.  This cost is generally speaking 
quite insignificant, however it is best to avoid it whenever possible.  The remaining numpy functions that are dominantly 
used are the np.random.choice, and np.random.dirichlet. 
"""


@njit(cache = True)
def update_phi_col(
    signals_meaning_array, alpha_val, alpha_on_s, meaning_index, num_signals
):
    """ "Update rule for phi is based upon the number of observed counts, i.e., equation 12 in the notes.
    As I understand the equation, the updates should only take place for the column corresponding to the meaning interpreted!
    So if the possible interpreted meanings are dog and cat, and we interpret cat, we do not touch the phi array values corresponding
    to meaning dog.

    The signals are updated according to the following:
    phi_i(s|m;t) = [(n_i(s|m;t)+alpha/S)]/(sum(n_i(s'|m;t))+alpha)
    where sum(n_i(s'|m;t)) corresponds to the sum over signals for that possible meaning, or in othern words the sum of the column


    Calculates the updated phi col, and then returns this to be sliced back into the array
    """

    new_col = np.empty(num_signals)
    col_to_sum = signals_meaning_array[:, meaning_index]
    denom = 0.0
    for i in range(num_signals):
        denom += col_to_sum[i]

    denom += alpha_val

    for j in range(num_signals):
        new_col[j] = (col_to_sum[j] + alpha_on_s) / denom

    return new_col


@njit(cache = True)
def update_signal_meaning(
    counts_array:np.ndarray,
    lambda_val:float, 
    meaning_index:int, 
    signal_index:int, 
    num_signals:int
):
    """Update rule according to equation 14 in document Evolution of Communications Through Fluctuations"""
    for i in range(num_signals):
        counts_array[i, meaning_index] = (
            delta(i=signal_index, j=i)
            + (1 - lambda_val) * counts_array[i, meaning_index]
        )
    return counts_array


@njit(cache = True)
def blind_success_inner_loop(agenti_phi, agentj_phi, num_signals, num_meanings):
    """I am defining the inner loop to be the sum over all meanings and signals of the selected agent's
    phi matrices.  I have decided to break up the blind_success calculations into two parts as I believe it
    will allow for easier modification of the formula, as well as an easier way to think about the process
    actually being done."""
    inst_sum = 0.0
    for a in range(num_meanings):
        signal_sum = 0.0
        for b in range(num_signals):
            denom_term = agentj_phi[b, :].sum()
            signal_sum += (agenti_phi[b, a] * agentj_phi[b, a]) / denom_term
        inst_sum += (1 / num_meanings) * signal_sum
    return inst_sum


@njit(cache = True)
def blind_success_outer_loop(num_agents, ensemble_phi, num_signals, num_meanings):
    """The outer loop of the blind success metric calculation, summing over all interacting agent pairs"""
    p_s = 0.0
    const = 1 / (num_agents * (num_agents - 1))
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
            else:
                agenti_phi = ensemble_phi[i, :, :]
                agentj_phi = ensemble_phi[j, :, :]
                p_s += blind_success_inner_loop(
                    agenti_phi, agentj_phi, num_signals, num_meanings
                )
    p_s *= const
    return p_s


@njit(cache = True)
def select_signal_integers_njit(
    alpha_dist:np.ndarray, 
    phi_array:np.ndarray, 
    meanings_integers:np.ndarray, 
    signals_integers:np.ndarray,
):
    """A numba function for fast meaning selection and signal emission once provided the rho (attentional weight dirichlet) distribution.
    The attentional weight distribution is calculated using sci-py, which is not supported in numba, thus we have no way to entirely numba-fy
    this process unless we decide to make a numba calculator of the dirichlet distribution.
    """
    rho_dist = np.random.dirichlet(alpha=alpha_dist)
    selected_mu = numba_random_choice_linear(meanings_integers, rho_dist)
    signal_prob = phi_array[:, selected_mu]
    selected_signal_idx = numba_random_choice_linear(signals_integers, signal_prob)
    return selected_signal_idx, rho_dist


@njit(cache = True)
def receive_signal_integers_njit(
    signal_idx, passed_rho_dist, phi_array, meanings_integers, num_meanings
):
    """A numba version of the signal receiving and interpretation process, confined to one generation"""

    rho_dist = passed_rho_dist
    phi_s_mu = phi_array[int(signal_idx), :]
    # print("phi_s_mu")
    # print(phi_s_mu)
    denom = (phi_s_mu * rho_dist).sum()
    # print("Denominator")
    # print(denom)
    posterior_dist = np.zeros(num_meanings, dtype=np.float64)
    for j in range(num_meanings):
        # inst_val = (phi_s_mu[j]*rho_dist[j])/denom
        # print("inst_val")
        # print(inst_val)
        posterior_dist[j] = (phi_s_mu[j] * rho_dist[j]) / denom

    # print("Posterior Chain")
    # print(posterior_dist)
    nu_idx = numba_random_choice_linear(meanings_integers, posterior_dist)
    return nu_idx


@njit(cache = True)
def receive_signal_intergers_intergen_njit(
    signal_idx, passed_alpha_dist, phi_array, meanings_integers, num_meanings
):
    """Numba function to model the dynamics of signal interpretation between two generations"""
    rho_dist = np.random.dirichlet(alpha=passed_alpha_dist)
    phi_s_mu = phi_array[int(signal_idx), :]
    # print("phi_s_mu")
    # print(phi_s_mu)
    denom = (phi_s_mu * rho_dist).sum()
    # print("Denominator")
    # print(denom)
    posterior_dist = np.zeros(num_meanings, dtype=np.float64)
    for j in range(num_meanings):
        # inst_val = (phi_s_mu[j]*rho_dist[j])/denom
        # print("inst_val")
        # print(inst_val)
        posterior_dist[j] = (phi_s_mu[j] * rho_dist[j]) / denom

    # print("Posterior Chain")
    # print(posterior_dist)
    nu_idx = numba_random_choice_linear(meanings_integers, posterior_dist)
    return nu_idx


"""End of numba core function section"""
###############################################################
"""Beginning of numba interaction dynamics section.  This section contains functions which guide/govern
 the interactions for different ensemble domains i.e., single generation, multi-generation, mutli-generation
 with an intergen alpha etc etc. More concretely: this section is the collection of functions from the previous 
 section, put together in a nice wrapper, so that the actual calls within python stay nice and organized"""


"""The way I am handling the alpha tensor would make it difficult to make an optimal function for both intergen, 
and one gen cases, solution is to make 2 different functions"""


@njit(cache = True)
def one_interaction_multiple_gens(
    counts,
    phi,
    speaker_info,
    meaning_integers,
    signal_integers,
    num_agents,
    alpha_tensor,
    M,
    S,
    lambda_val,
    alpha_val,
    alpha_on_S,
):

    speaker = np.random.randint(num_agents)
    receiver = np.random.randint(num_agents - 1)
    if receiver >= speaker:
        receiver += 1

    speaker_gen_tag = speaker_info.gens[speaker]
    receiver_gen_tag = speaker_info.gens[receiver]
    speaker_alpha = alpha_tensor[speaker_gen_tag, :]
    speaker_phi = phi[speaker, :, :]
    receiver_phi = phi[receiver, :, :]

    receiver_counts = counts[receiver, :, :]

    selected_signal_idx, rho_dist = select_signal_integers_njit(
        speaker_alpha, speaker_phi, meaning_integers, signal_integers
    )

    if speaker_gen_tag == receiver_gen_tag:
        """Communication between two speakers of same gen, rho is passed"""
        nu_idx = receive_signal_integers_njit(
            selected_signal_idx, rho_dist, receiver_phi, meaning_integers, M
        )
    else:
        receiver_alpha = alpha_tensor[receiver_gen_tag, :]
        """Communication between two different gens, they each just use their own alpha"""
        receiver_rho = np.random.dirichlet(alpha=receiver_alpha)
        nu_idx = receive_signal_integers_njit(
            selected_signal_idx, receiver_rho, receiver_phi, meaning_integers, M
        )

    """Updating the corresponding arrays"""
    counts[receiver, :, :] = update_signal_meaning(
        receiver_counts, lambda_val, nu_idx, selected_signal_idx, S
    )

    new_phi_col = update_phi_col(
        counts[receiver, :, :], alpha_val, alpha_on_S, nu_idx, S
    )
    phi[receiver, :, nu_idx] = new_phi_col

    return SimulationState(phi=phi, counts=counts)


@njit(cache = True)
def one_collection_loop_multiple_gens(
    counts,
    phi,
    speaker_info,
    meaning_integers,
    signal_integers,
    num_agents,
    alpha_tensor,
    M,
    S,
    lambda_val,
    alpha_val,
    alpha_on_S,
    iterations,
):
    """Runs the data collection/communication process for a specified amount of iterations, then returns the counts and phi arrays"""
    for _ in range(iterations):
        state = one_interaction_multiple_gens(
            counts,
            phi,
            speaker_info,
            meaning_integers,
            signal_integers,
            num_agents,
            alpha_tensor,
            M,
            S,
            lambda_val,
            alpha_val,
            alpha_on_S,
        )
        counts = state.counts
        phi = state.phi

    return phi, counts


"""Intergen interactions"""


@njit(cache = True)
def one_interaction_intergen(
    counts: np.ndarray,
    phi: np.ndarray,
    speaker_info: namedtuple,
    meaning_integers: np.ndarray,
    signal_integers: np.ndarray,
    numba_consts: namedtuple,
    alpha_tensor: np.ndarray,
    intra_gen_alignment: float,
    intergen_alignment: float,
    num_gens: int,
    gen_dict: Dict,
    intergen_prob: float,
    gen_choice: list
):
    """If they are of the same population they pass their rho's with prob ~ alignment A, if they are of different generations, then BOTH
    agents use the intergen alpha counts, however this is still passed with alignment A"""

    ###########################################################################################
    # Note yet configured for nonuniform A
    ###########################################################################################

    interaction_type_roll = np.random.rand()
    if interaction_type_roll < intergen_prob:
        """Intergen interaction, speaker (selected via the selected_speaker_gen) will communicate with a speaker from the opposite gen"""
        selected_agent_gens = np.random.choice(gen_choice, size=2, replace = False)

        selected_speaker_gen = selected_agent_gens[0]
        selected_receiver_gen = selected_agent_gens[1]

        speaker_arr = gen_dict[selected_speaker_gen]
        speaker = np.random.choice(speaker_arr)

        receiver_arr = gen_dict[selected_receiver_gen]
        receiver = np.random.choice(receiver_arr)

        speaker_phi = phi[speaker, :, :]
        receiver_phi = phi[receiver, :, :]

        receiver_counts = counts[receiver, :, :]

        """If the speaker and receiver do not have the same gen tag, then we must use the intergen alpha for both"""
        intergen_alpha = alpha_tensor[-1, :]
        """NOTE: Above will need to change as soon as dealing with more complex systems (i.e., more than 2 gens)"""
        selected_signal_idx, rho_dist = select_signal_integers_njit(
            intergen_alpha, speaker_phi, meaning_integers, signal_integers
        )

        if np.random.rand() <= intergen_alignment:
            """Pass rho dist"""
            nu_idx = receive_signal_integers_njit(
                selected_signal_idx,
                rho_dist,
                receiver_phi,
                meaning_integers,
                numba_consts.M,
            )
        else:
            receiver_rho = np.random.dirichlet(alpha=intergen_alpha)
            nu_idx = receive_signal_integers_njit(
                selected_signal_idx,
                receiver_rho,
                receiver_phi,
                meaning_integers,
                numba_consts.M,
            )

    else:
        """Intragen interaction, need to select a speaker and receiver from the same generation"""

        selected_gen = np.random.randint(low=0, high=(num_gens))

        agent_arr = gen_dict[selected_gen]
        speaker_receiver_idx = np.random.choice(agent_arr, size=2, replace=False)
        speaker = speaker_receiver_idx[0]
        receiver = speaker_receiver_idx[1]
        speaker_gen_tag = selected_gen

        speaker_phi = phi[speaker, :, :]
        receiver_phi = phi[receiver, :, :]

        receiver_counts = counts[receiver, :, :]

        """Same generation branch, we use the alpha from the speaker/receiver gen_tag, and pass the inst rho off with prob ~ Alignment"""
        speaker_alpha = alpha_tensor[speaker_gen_tag, :]
        selected_signal_idx, rho_dist = select_signal_integers_njit(
            speaker_alpha, speaker_phi, meaning_integers, signal_integers
        )

        if np.random.rand() <= intra_gen_alignment:
            """Pass the rho distribution"""
            nu_idx = receive_signal_integers_njit(
                selected_signal_idx,
                rho_dist,
                receiver_phi,
                meaning_integers,
                numba_consts.M,
            )

        else:
            """Receiver must create their own rho distribution"""
            receiver_rho = np.random.dirichlet(
                alpha=speaker_alpha
            )  # Can still reuse speaker alpha since they were same gen
            nu_idx = receive_signal_integers_njit(
                selected_signal_idx,
                receiver_rho,
                receiver_phi,
                meaning_integers,
                numba_consts.M,
            )

    """Updating the corresponding arrays"""
    counts[receiver, :, :] = update_signal_meaning(
        receiver_counts,
        speaker_info.lambda_val[receiver],
        nu_idx,
        selected_signal_idx,
        numba_consts.S,
    )

    new_phi_col = update_phi_col(
        counts[receiver, :, :],
        numba_consts.alpha_val,
        numba_consts.alpha_on_S,
        nu_idx,
        numba_consts.S,
    )
    phi[receiver, :, nu_idx] = new_phi_col

    return phi, counts


@njit(cache = True)
def one_collection_loop_intergens(
    counts: np.ndarray,
    phi: np.ndarray,
    speaker_info: namedtuple,
    meaning_integers: np.ndarray,
    signal_integers: np.ndarray,
    numba_consts: namedtuple,
    alpha_tensor: np.ndarray,
    intra_gen_alignment: float,
    intergen_alignment: float,
    gen_dict: Dict,
    intergen_prob: float,
    gen_maps: np.ndarray,
    iterations: int,
    gen_choice:list
):
    """Runs the data collection/communication process for a specified amount of iterations, then returns the counts and phi arrays"""

    gen_num = gen_maps.shape[0]
    # gen_select_arr = np.arange(gen_num)
    for _ in range(iterations):
        new_phi, new_counts = one_interaction_intergen(
            counts,
            phi,
            speaker_info,
            meaning_integers,
            signal_integers,
            numba_consts,
            alpha_tensor,
            intra_gen_alignment,
            intergen_alignment,
            gen_num,
            gen_dict,
            intergen_prob,
            gen_choice
        )
        counts = new_counts
        phi = new_phi

    return phi, counts


"""Now doing the single generation interaction processes"""


@njit(cache = True)
def one_interaction_one_population(
    counts: np.ndarray,
    phi: np.ndarray,
    meaning_integers: np.ndarray,
    signal_integers: np.ndarray,
    numba_consts: namedtuple,
    alpha_vector: np.ndarray,
    speaker_info: namedtuple,
):
    """Interaction rules for an ensemble of speakers all from one population"""

    speaker = np.random.randint(numba_consts.N)
    receiver = np.random.randint(numba_consts.N - 1)
    if receiver >= speaker:
        receiver += 1

    speaker_A = speaker_info.align[speaker]
    receiver_A = speaker_info.align[receiver]

    receiver_lambda = speaker_info.lambda_val[receiver]
    avg_align = 0.5 * (speaker_A + receiver_A)

    speaker_phi = phi[speaker, :, :]
    receiver_phi = phi[receiver, :, :]

    receiver_counts = counts[receiver, :, :]

    selected_signal_idx, rho_dist = select_signal_integers_njit(
        alpha_vector, speaker_phi, meaning_integers, signal_integers
    )
    # Above taken nearly directly from multi gen interactions

    # alignment rolls
    alignment_roll = np.random.random()
    if alignment_roll <= avg_align:
        """If the roll is less than the specified value then pass the rho distribution, else reroll it"""
        receiver_rho = rho_dist
    else:
        receiver_rho = np.random.dirichlet(alpha=alpha_vector)

    nu_idx = receive_signal_integers_njit(
        selected_signal_idx,
        receiver_rho,
        receiver_phi,
        meaning_integers,
        numba_consts.M,
    )
    """Updating the corresponding arrays"""
    counts[receiver, :, :] = update_signal_meaning(
        receiver_counts, receiver_lambda, nu_idx, selected_signal_idx, numba_consts.S
    )

    new_phi_col = update_phi_col(
        counts[receiver, :, :],
        numba_consts.alpha_val,
        numba_consts.alpha_on_S,
        nu_idx,
        numba_consts.S,
    )
    phi[receiver, :, nu_idx] = new_phi_col

    return phi, counts


@njit(cache = True)
def one_collection_loop_one_pop(
    counts: np.ndarray,
    phi: np.ndarray,
    meaning_integers: np.ndarray,
    signal_integers: np.ndarray,
    numba_consts: namedtuple,
    alpha_vector: np.ndarray,
    speaker_info: namedtuple,
    iterations: int,
):
    """Runs the data collection/communication process for a specified amount of iterations, then returns the counts and phi arrays"""
    for _ in range(iterations):
        new_phi, new_counts = one_interaction_one_population(
            counts,
            phi,
            meaning_integers,
            signal_integers,
            numba_consts,
            alpha_vector,
            speaker_info,
        )
        counts = new_counts
        phi = new_phi

    return phi, counts


#Adding a separate interaction dynamic in the case where we have non_uniform alignment --> Alignment is selected via the alignment matrix
#Treating dimension 0 (row) as speaker, and dimension 1 (col) as receiver

@njit(cache = True)
def one_interaction_one_pop_NuA(counts: np.ndarray,
    phi: np.ndarray,
    meaning_integers: np.ndarray,
    signal_integers: np.ndarray,
    numba_consts: namedtuple,
    alpha_vector: np.ndarray,
    speaker_info: namedtuple,):
    """Procedure is largely the same as the previous case, just with alignment selected from the matrices as 
    opposed to being assigned to a vector"""
    speaker = np.random.randint(numba_consts.N)
    receiver = np.random.randint(numba_consts.N - 1)
    if receiver >= speaker:
        receiver += 1

    receiver_lambda = speaker_info.lambda_val[receiver]

    speaker_phi = phi[speaker, :, :]
    receiver_phi = phi[receiver, :, :]

    receiver_counts = counts[receiver, :, :]

    selected_signal_idx, rho_dist = select_signal_integers_njit(
        alpha_vector, speaker_phi, meaning_integers, signal_integers
    )
    # Above taken nearly directly from multi gen interactions

    # alignment rolls
    alignment_roll = np.random.random()
    if alignment_roll <= speaker_info.align[speaker,receiver]:
        """If the roll is less than the specified value then pass the rho distribution, else reroll it"""
        receiver_rho = rho_dist
    else:
        receiver_rho = np.random.dirichlet(alpha=alpha_vector)

    nu_idx = receive_signal_integers_njit(
        selected_signal_idx,
        receiver_rho,
        receiver_phi,
        meaning_integers,
        numba_consts.M,
    )
    """Updating the corresponding arrays"""
    counts[receiver, :, :] = update_signal_meaning(
        receiver_counts, receiver_lambda, nu_idx, selected_signal_idx, numba_consts.S
    )

    new_phi_col = update_phi_col(
        counts[receiver, :, :],
        numba_consts.alpha_val,
        numba_consts.alpha_on_S,
        nu_idx,
        numba_consts.S,
    )
    phi[receiver, :, nu_idx] = new_phi_col

    return phi, counts


@njit(cache = True)
def one_collection_loop_one_pop_NuA(
    counts: np.ndarray,
    phi: np.ndarray,
    meaning_integers: np.ndarray,
    signal_integers: np.ndarray,
    numba_consts: namedtuple,
    alpha_vector: np.ndarray,
    speaker_info: namedtuple,
    iterations: int,
):
    """Runs the data collection/communication process for a specified amount of iterations, then returns the counts and phi arrays"""
    for _ in range(iterations):
        new_phi, new_counts = one_interaction_one_pop_NuA(
            counts,
            phi,
            meaning_integers,
            signal_integers,
            numba_consts,
            alpha_vector,
            speaker_info,
        )
        counts = new_counts
        phi = new_phi

    return phi, counts

"""End of interaction dynamics section"""
###############################################################


@dataclass(slots=True)
class SimulationBaseConfig:
    """Data class to hold all the basic values and parameters the simulation must take, things like the number of signals and meanings,
    the number of agents, alpha, beta, and lambda.  Another separate dataclass is used for other dependencies of the simulation"""

    num_meanings: int
    num_signals: int
    num_agents: int
    alignment: float
    beta: int
    alpha: float
    lambda_val: float
    output_dir: str
    iterations: int
    export: bool
    time_res: int = 100000


@dataclass(slots=True)
class SimulationGensConfig:
    """Dataclass responsible for storing and passing all the required information for the generation simulations"""

    havefocuses: bool
    import_data: bool
    read_dir: str
    gamma: float
    focuses:int
    passed_focuses: list
    intergenA: float
    generation_overlap: bool = False


@dataclass(slots=True)
class SimulationNoiseConfig:
    """Dataclass holding all pertinent information regarding alignment noise.  To be expanded upon and fleshed out more
    as I understand more about this parameter space"""

    uniformA: bool = True
    means: list = None
    stds: list = None
    weight: list = None

class FileSaver:
    """Class responsible for saving all the data from the ensemble dynamics"""
    def __init__(self,base_config):
        self.output_dir = base_config.output_dir
        self.save_dir = self.output_dir / "raw"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_measurements = int(
            base_config.iterations // base_config.time_res
        )
        self.S = base_config.num_signals
        self.M = base_config.num_meanings

    def generate_h5_file(self,num_gens, export = False, num_agents = None):
        """Generates the h5_file used to store data as the experiment runs, should only be
        used after the initial ensemble arrays have been calculated"""
        self.h5_dir = Path(self.save_dir)/"array_data.h5"
        with h5py.File(name = self.h5_dir, mode="w") as f:
            """Ensemble_Phi Data"""
            phi = f.create_dataset(name = "Phi/phi_dataset",shape=(self.num_measurements, self.S, self.M),
                            dtype=np.float64,
                            chunks=True,)

            phi.attrs["description"] = "Ensemble phi arrays over time"
            phi.attrs["num_signals"] = self.S
            phi.attrs["num_meanings"] = self.M

            """Generation specific phi data"""
            phi_gens = f.create_dataset(name = "Phi/phi_gens_dataset",shape=(self.num_measurements, num_gens + 1, self.S, self.M),
                            maxshape=(self.num_measurements, None, self.S, self.M),
                            dtype=np.float64,
                            chunks=True,)

            # Want it to be none in both the first and second generation to allow in simulation updates later on
            phi_gens.attrs["description"] = (
                "Generation specific average phi arrays over time"
            )
            phi_gens.attrs["num_signals"] = self.S
            phi_gens.attrs["num_meanings"] = self.M
            phi_gens.attrs["num_gens"] = num_gens


            """Ensemble_n(S|M) Data"""
            counts = f.create_dataset(name = "counts/counts_dataset",
                                    shape=(self.num_measurements, self.S, self.M),
                                                    dtype=np.float64,
                                                    chunks=True,)
            counts.attrs["description"] = (
                "Ensemble signal meaning count array over time "
            )
            counts.attrs["num_signals"] = self.S
            counts.attrs["num_meanings"] = self.M

            """Generation specific counts data, copying same overall structure as the gens phi"""

            counts_gens = f.create_dataset(name ="counts/counts_gens_dataset",
                            shape=(self.num_measurements, num_gens + 1, self.S, self.M),
                            maxshape=(self.num_measurements, None, self.S, self.M),
                            dtype=np.float64,
                            chunks=True,)
            # Want it to be none in both the first and second generation to allow in simulation updates later on
            counts_gens.attrs["description"] = (
                "Generation specific average counts arrays over time"
            )
            counts_gens.attrs["num_signals"] = self.S
            counts_gens.attrs["num_meanings"] = self.M
            counts_gens.attrs["num_gens"] = num_gens

            # print(f"Starting h5_counts_gens values")
            # print(f"Keys of gens dict is {self.gen_agent_dict.keys()}")
            # print(f"Avg map is {self.avg_map}")

            """If self.export_agents set to true, we need an array to save the final phi and count arrays from all agents """
            if export:
                agents_phi = f.create_dataset(
                    name="agents_phi/agents_phi_data",
                    shape=(num_agents, self.S, self.M),
                    dtype=np.float64,
                )
                agents_phi.attrs["description"] = (
                    "Each agent in the simulation's final phi array"
                )
                agents_phi.attrs["num_signals"] = self.S
                agents_phi.attrs["num_meanings"] = self.M


                agents_counts = f.create_dataset(name = "agents_counts/agents_counts_data",
                    shape=(num_agents, self.S, self.M),
                    dtype=np.float64,)
                
                agents_counts.attrs["description"] = (
                    "Each agent in the simulation's final phi array"
                )
                agents_counts.attrs["num_signals"] = self.S
                agents_counts.attrs["num_meanings"] = self.M
            f.close()

    def open_h5_file(self,mode  = 'r+'):
        if not self.h5_dir.exists():
            raise FileNotFoundError(f"{self.h5_dir} file does not exist, create_h5_file must be called before open_h5 file")
        return h5py.File(self.h5_dir,mode)

    def write_to_h5(self,idx_iter:int, phi:np.ndarray,counts:np.ndarray,num_gens:int,
                    avg_map:np.ndarray = None,gain_values:list=None, gens_gain:dict=None, gain_epochs: list = None):
        """Writes the ensemble data to the h5 file.  If have multiple gens and want data stored separately, the average map must be passed to the function.
        If agent data is meant to be exported, such that the configuration can be continued at a later date, export must be set to true."""

        if gain_values is not None:
            """Only one set of gain values, corresponds to 1 gen case"""
            pd.DataFrame({"epoch": gain_epochs, "gain": gain_values}).to_csv(
                self.save_dir / "gain_values.csv", index=False
            )
        ###########################################################################################################################
        """Number of gens is currently hardcoded into how the pandas file handles the data, in the future, it may be better
        to shift this sort of thing into a numpy array, and use h5"""
        if gens_gain is not None:
            """Multiple generation gains, corresponds to system of multiple communicators"""
            pd.DataFrame(
                {
                    "epoch": gain_epochs,
                    "ensemble_gain": gens_gain[0],
                    "gen_0_gain": gens_gain[1],
                    "gen_1_gain": gens_gain[2],
                }
            ).to_csv(self.save_dir / "gen_gain_values.csv", index=False)
        ###########################################################################################################################

        if idx_iter is not None:
            """If iteration is not none, we append the data we found to the h5py file we initialised earlier"""

            with self.open_h5_file() as f:

                f["Phi/phi_dataset"][idx_iter, :, :] = phi.mean(axis=0)
                f["counts/counts_dataset"][idx_iter, :, :] = counts.mean(axis=0)

                if avg_map is not None:
                    """If an average map is passed then we save to gen specific h5 structures"""
                    start_idx = 0
                    for k in range(num_gens + 1):
                        end_idx = avg_map[k] + start_idx
                        ensemble_phi = phi[start_idx:end_idx, :, :]
                        ensemble_counts = counts[start_idx:end_idx, :, :]
                        f["Phi/phi_gens_dataset"][idx_iter, k, :, :] = ensemble_phi.mean(axis=0)
                        f["counts/counts_gens_dataset"][idx_iter, k, :, :] = ensemble_counts.mean(
                            axis=0
                        )
                        start_idx = end_idx
                f.close()

    def save_agents_data(self, num_agents:int, phi:np.ndarray, counts:np.ndarray):
        """A function to be called if export agents is set to true, saves the individual agent's phi and count arrays to the h5 file"""
        with h5py.File(self.h5_dir, "a") as f:
            agents_phi_group = f["agents_phi"]
            agents_phi_data = agents_phi_group["agents_phi_data"]
            agents_counts_group = f["agents_counts"]
            agents_counts_data = agents_counts_group["agents_counts_data"]
            for i in range(num_agents):
                agents_phi_data[i, :, :] = phi[i, :, :]
                agents_counts_data[i, :, :] = counts[i, :, :]

            f.close()
        print("Agent data saved")

    def save_params(self,params_df:pd.DataFrame):
        """Saves a passed data frame to a csv file."""
        params_file = self.output_dir / "simulation_parameters.csv"
        params_df.to_csv(params_file, index=False)
        print(f"Saved simulation parameters to {params_file}")


class Constructor:
    """A constructor helper class which is responsible for instantiating the phi and count arrays either from a fresh array, or from file imports.
    The constructor takes the basic config values passed to it from the ensemble, from which point it instantiates a base state.  This base state is 
    then further modified depending on the specific configuration we are wanting to create.  All these state values are then packed into a dict,
    and then handed back to the ensemble class to unpack.  This is done intentionally, as I want all the tensors, and values to be ensemble specific items,
    not ensemble.constructor___ items. 

    Basic flow is this:
    Ensemble_init ->
    Constructor_init ->
    Constructor_base_state ->
    Constructor_specific_state -->
    return specific_state-->
    for key, value in state.items():
        setattr(self, key, value)
    
    """

    def __init__(self,base_config:SimulationBaseConfig,gens_config:SimulationGensConfig,noise_config:SimulationNoiseConfig):
        """Instantiates all the major constructor variables from the given configs"""
        
        self.base_config = base_config
        self.gens_config = gens_config
        self.noise_config = noise_config

        # Extract all the commonly used values for convenience
        self.M = base_config.num_meanings
        self.S = base_config.num_signals
        self.num_agents = 0
        self.alpha = base_config.alpha
        self.beta = base_config.beta
        self.lambda_val = base_config.lambda_val
        self.A = base_config.alignment

        # State parameters
        self.avg_map = [] #Guide to which indices in the tensors belong to which generation
        self.num_gens = 0
        self.gen_focuses = []
        self.possible_focuses = np.arange(self.M).tolist()
        self.gen_speaker_gens = None
        self.gen_speaker_lambda = None
        self.gen_speaker_align = None
        self.gen_agent_dict = None
        self.gamma = gens_config.gamma


        self.read_dir = Path(gens_config.read_dir)

    def create_generation_focuses(
        self, number_focuses:int=2, supplied_focuses:list=None, remove_focuses:bool=True, gamma:float=None
    ):
        """A function to return attentional counts, modified by a select number of attentional focuses, for a generation.
        Focuses are either selected randomly from an inputted list of allowed focuses, or they are supplied directly.  The degree to which these counts are modifed
        ('focused on') comes from the parameter gamma, which is a straighforward multiplier of the number of counts.  All meanings
        not within the group of selected focuses have a count value corresponding to beta/M where M is total number of meanings.

        If remove_focuses is set to true, the function removes the selected or inputted focuses from the possible focuses list
        """
        if gamma is None:
            gamma = self.gamma

        if supplied_focuses is None:
            selected_focuses = np.random.choice(
                self.possible_focuses, size=number_focuses, replace=False
            )
            self.gen_focuses.append(selected_focuses.tolist())
        else:
            selected_focuses = supplied_focuses
            self.gen_focuses.append(selected_focuses)

        gen_counts = np.full(self.M, (self.beta / self.M))
        for focus in selected_focuses:
            gen_counts[focus] *= gamma
            if remove_focuses:
                # print(f"Debug, inst_focus is {focus}")
                self.possible_focuses.remove(focus)

        return gen_counts, selected_focuses

    def generate_non_uniform_A(self,N,means, stds, weight):
            """
            In a non-uniform alignment situation, alignment between each agent/communicator is selected via a bimodal distribution.
            It is important to note that these alignments are not assigned to agents directly, but assigned however to the actual 
            interactions between specific agents.   
    
            We accomplish this by leveraging the fact that N(mu,sigma) = sigma * Z + mu where Z selected from N(0,1)
    
            """
    
            rng = np.random.default_rng()
            
            z_vals = rng.standard_normal(size = (N,N))
            weight_draws = rng.random(size = (N,N))
            draw_map = weight_draws>=weight
            draw_means = np.where(draw_map,means[0],means[1])
            draw_std = np.where(draw_map,stds[0],stds[1])
    
            bimod_draw = draw_std*z_vals + draw_means
            filtered_vals = np.where((bimod_draw>1.0),1.0,bimod_draw)
            print(f"Debug: filtered_vals has shape {filtered_vals.shape}")
            print(filtered_vals)
            return filtered_vals
    
    def generate_non_uniform_A_symmetric(self,N,means,stds,weight):
        """Similar procedure to the base version, except that this generates the alignment matrix symmetrically, such that interactions
        between agents are symmetric pairwise (prevents possibility that for same 2 agents, one speaker is preferred)
        
        N = number of speakers
        means = means of bimodal distribution
        stds = sigma of bimodal distribution
        weight = weight of first dist in bimodal distribution (1-weight is weight of other)
        """

        upper_triangle_indices = np.triu_indices(N,k=1) #start from k =1, self interactions are not possible -> main diagonal should be zero
        num_pairs = len(upper_triangle_indices[0])

        #Rest of procedure very similar
        rng = np.random.default_rng()

        z_vals = rng.standard_normal(num_pairs)
        draw_map = rng.random(num_pairs)>=weight

        drawed_means = np.where(draw_map,means[0],means[1])
        drawed_stds =  np.where(draw_map,stds[0],stds[1])

        draws = np.clip(drawed_means+(drawed_stds*z_vals),0.0,1.0)

        A = np.zeros((N,N))
        A[upper_triangle_indices] = draws
        A = A + A.T
        return A

    def read_param_file(self, params_dir):
        """
        dir: params_directory

        Reads the inputted parameter.csv file and updates/records the relevant ensemble parameter values accordingly"""
        params_df = pd.read_csv(params_dir / "simulation_parameters.csv")
        self.imported_meanings = params_df["num_meanings"][0]
        # print(self.M)
        self.imported_signals = params_df["num_signals"][0]
        self.num_imported_agents = params_df["num_agents"][0]
        self.imported_gens_lambda = params_df["lambda"][0]
        self.imported_beta = params_df["beta"][0]

        old_gamma = params_df["gamma"][0]
        if self.gamma is None:
            self.gamma = old_gamma
        else:
            self.imported_gamma = old_gamma

        self.imported_gen_focuses = params_df["generation_focuses"].apply(
            ast.literal_eval
        )[0]
        self.imported_A = params_df["alignment"][0]
        # Need the extra handling to convert the csv's string back into an actual python list

    def _construct_uniform_A(self,state):
        """Helper function that takes a base state and returns a simulation state configured with uniform alignment for each agent.
           """

        print("Instantiating speakers with no focuses and uniform alignment")
        num_added = baseconfig.num_agents
        self.num_agents += num_added

        self.alpha_tensor = np.full((self.M), fill_value=(self.beta / self.M))

        self.population_phi = np.full(
            (self.num_agents, self.S, self.M), fill_value=1 / self.S
        )
        self.population_counts = np.zeros((self.num_agents, self.S, self.M))
        self.avg_map.append(self.num_agents)

        self.speaker_gens = np.zeros(
            self.num_agents, dtype=np.int16
        )  # As of now, only really operating with gens listed from 0-1 (maybe 2)
        # no need for high bit count
        self.speaker_gens[:] = int(self.num_gens)
        self.speaker_lambda = np.zeros(self.num_agents, dtype=np.float64)
        self.speaker_lambda[:] = self.lambda_val

        self.speaker_align = np.full((self.num_agents), fill_value=self.A)

        state.update({
            'num_agents': self.num_agents,
            'population_phi': self.population_phi,
            'population_counts': self.population_counts,
            'alpha_tensor': self.alpha_tensor,
            'speaker_gens': self.speaker_gens,
            'speaker_lambda': self.speaker_lambda,
            'speaker_align': self.speaker_align,
            'avg_map': self.avg_map
            })

        return state

    def _construct_NuA(self,state):
        """Helper function that takes a base state and returns a simulation state configured for a non-uniform (typically bimodal) alignment
        matrix for interactions"""
        print("Instantiating speakers with no focuses and non uniform alignment")
        num_added = baseconfig.num_agents
        self.num_agents += num_added



        self.speaker_align = self.generate_non_uniform_A_symmetric(self.num_agents,self.noise_config.means,self.noise_config.stds,self.noise_config.weight)
        print(f"Debug, in _construct_NuA, self.speaker_align has shape {self.speaker_align.shape}")
        
        self.alpha_tensor = np.full((self.M), fill_value=(self.beta / self.M))

        self.population_phi = np.full(
            (self.num_agents, self.S, self.M), fill_value=1 / self.S
        )
        self.population_counts = np.zeros((self.num_agents, self.S, self.M))
        self.avg_map.append(self.num_agents)
        self.speaker_gens = np.zeros(
            self.num_agents, dtype=np.int16
        )  # As of now, only really operating with gens listed from 0-1 (maybe 2)
        # no need for high bit count
        self.speaker_gens[:] = int(self.num_gens)
        self.speaker_lambda = np.zeros(self.num_agents, dtype=np.float64)
        self.speaker_lambda[:] = self.lambda_val
        state.update({
            'num_agents': self.num_agents,
            'population_phi': self.population_phi,
            'population_counts': self.population_counts,
            'alpha_tensor': self.alpha_tensor,
            'speaker_gens': self.speaker_gens,
            'speaker_lambda': self.speaker_lambda,
            'speaker_align': self.speaker_align,
            'means':self.noise_config.means,
            'stds':self.noise_config.stds,
            'weight':self.noise_config.weight,
            'avg_map': self.avg_map
            })
        return state

    def _construct_w_focuses(self,state):
        """Helper function that takes a base state and returns a simulation state configured with focused alpha vectors (i.e., modified by gamma)"""

        print("Instantiating speakers with focuses and uniform alignment")
        num_added = baseconfig.num_agents
        self.num_agents += num_added
        gen_counts,_ =self.create_generation_focuses(self.gens_config.focuses,self.gens_config.passed_focuses,gamma=self.gens_config.gamma)
        self.alpha_tensor = np.array(gen_counts)
        
        self.population_phi = np.full(
            (self.num_agents, self.S, self.M), fill_value=1 / self.S
        )
        self.population_counts = np.zeros((self.num_agents, self.S, self.M))
        self.avg_map.append(self.num_agents)
        

        self.speaker_gens = np.zeros(
            self.num_agents, dtype=np.int16
        )  # As of now, only really operating with gens listed from 0-1 (maybe 2)
        # no need for high bit count
        self.speaker_gens[:] = int(self.num_gens)
        self.speaker_lambda = np.zeros(self.num_agents, dtype=np.float64)
        self.speaker_lambda[:] = self.lambda_val

        self.speaker_align = np.full((self.num_agents), fill_value=self.A)
        # self.speaker_align = self.A
        state.update({
            'num_agents': self.num_agents,
            'population_phi': self.population_phi,
            'population_counts': self.population_counts,
            'alpha_tensor': self.alpha_tensor,
            'speaker_gens': self.speaker_gens,
            'speaker_lambda': self.speaker_lambda,
            'speaker_align': self.speaker_align,
            'gen_focuses': self.gen_focuses,
            'avg_map': self.avg_map, 
            'possible_focuses':self.possible_focuses
            })
        return state

    def _construct_from_imports(self,state):
        """Helper function that takes a base state and returns a simulation state configured to the imported agents"""
        print("Instantiating speakers from ImportFile")
        
        self.read_param_file(params_dir=self.read_dir)

        with h5py.File(self.read_dir/"raw"/ "array_data.h5", "r") as f:
            final_phis = f["agents_phi"]["agents_phi_data"]
            final_counts = f["agents_counts"]["agents_counts_data"]

            self.population_phi = final_phis[:]
            self.population_counts = final_counts[:]
            self.avg_map.append(self.num_imported_agents)

            #############################################################################################################################################
            if len(self.imported_gen_focuses) != 0:
                gen_counts, _focuses = self.create_generation_focuses(
                    len(self.imported_gen_focuses),
                    supplied_focuses=self.imported_gen_focuses[0],
                    gamma=self.imported_gamma,
                )
                self.alpha_tensor = np.array(gen_counts)
            else:
                self.alpha_tensor = np.full((self.M), fill_value=(self.beta / self.M))
            """NOTE: Above line will absolutely break if importing more than one generation, special care will be needed when that is done"""
            #############################################################################################################################################

        f.close()

        num_added = self.population_phi.shape[0]
        print(f"Num added is {self.population_phi.shape[0]}")
        self.num_agents += num_added

        self.speaker_gens = np.zeros(
            self.num_agents, dtype=np.int16
        )  # As of now, only really operating with gens listed from 0-1 (maybe 2)
        # no need for high bit count
        self.speaker_gens[:] = int(self.num_gens)
        self.speaker_lambda = np.zeros(self.num_agents, dtype=np.float64)
        self.speaker_lambda[:] = self.imported_gens_lambda

        self.speaker_align = np.full(num_added,self.imported_A, dtype=np.float64)
        """As of now no imported speakers are being passed the alignment"""
        # self.speaker_align = self.A
        
        state.update({
            'num_agents': self.num_agents,
            'population_phi': self.population_phi,
            'population_counts': self.population_counts,
            'alpha_tensor': self.alpha_tensor,
            'speaker_gens': self.speaker_gens,
            'speaker_lambda': self.speaker_lambda,
            'speaker_align': self.speaker_align,
            'gen_focuses': self.gen_focuses,
            'avg_map': self.avg_map,
            'possible_focuses':self.possible_focuses
            })
        return state

    def construct_simulation_state(self):
        """Generates the simulation specific state.  Does this first via generating a base state dict, which is then passed
        to a more specific helper function for the specific configuration desired (i.e., non-uniform alignment, multi-gen, single-pop, etc)"""

        state = {
        'M': self.M,
        'S': self.S,
        'num_agents': 0,
        'alpha': self.alpha,
        'beta':self.beta,
        'gamma':self.gamma,
        'alpha_on_s': self.alpha / self.S,
        'lambda_val': self.lambda_val,
        'A': self.A,
        'time_res': self.base_config.time_res,
        'tot_iterations': self.base_config.iterations,
        'export_agents': self.base_config.export,
        'num_gens': 0,
        'avg_map': [],
        'gen_focuses': [],
        'intergen_A': self.gens_config.intergenA,
        'param_dir':self.read_dir,
        }
        if self.gens_config.havefocuses:
            if self.gens_config.import_data:
                return self._construct_from_imports(state)
            else:
                return self._construct_w_focuses(state)
        else:
            if self.noise_config.uniformA:
                return self._construct_uniform_A(state)
            else:
                return self._construct_NuA(state)




class Ensemble:
    def __init__(
        self,
        basic_config: SimulationBaseConfig,
        gens_config: SimulationGensConfig,
        noise_config: SimulationNoiseConfig,
    ):
        print("Initializing")

        """Unpacking all elements from the base config NOT unpacked by the constructor"""

        print("Unpacking Base Config")
    
        self.meanings_integers = np.arange(basic_config.num_meanings, dtype=int)
        self.signals_integers = np.arange(basic_config.num_signals, dtype=int)
        self.C = 1 / (1 + basic_config.beta)
        self.time_res = basic_config.time_res
        self.tot_iterations = basic_config.iterations
        self.num_measurements = int(
            self.tot_iterations // self.time_res
        )  # This was being read as a float despite the integer division, not sure why

        self.gain_epochs = []
        # List to store the alpha arrays of each generation

        print("Initialising Constructor")
        self.constructor = Constructor(basic_config,gens_config,noise_config)
        self.constructed_state = self.constructor.construct_simulation_state() #Keep the state such that we can pass it back into the constructor later to change state
        for key, value in self.constructed_state.items():
            setattr(self,key,value) #writing the constructor state variables to ensemble variables

        """Debugging sections below while the code is still being made"""
        # print(f"Number of imported agents is {self.num_imported_agents}")
        print(f"Population phi has shape {self.population_phi.shape}")
        print(f"Population counts has shape {self.population_counts.shape}")
        print(
            f"Alpha tensor has shape {self.alpha_tensor.shape} and is {self.alpha_tensor} "
        )
        print(f"Number of agents is {self.num_agents}")
        # self.numba_warmup()

        self.file_saver = FileSaver(basic_config)

    def read_param_file(self, params_dir):
        """
        dir: params_directory

        Reads the inputted parameter.csv file and updates/records the relevant ensemble parameter values accordingly"""
        params_df = pd.read_csv(params_dir / "simulation_parameters.csv")
        self.imported_meanings = params_df["num_meanings"][0]
        # print(self.M)
        self.imported_signals = params_df["num_signals"][0]
        self.num_imported_agents = params_df["num_agents"][0]
        self.imported_gens_lambda = params_df["lambda"][0]
        self.imported_beta = params_df["beta"][0]

        old_gamma = params_df["gamma"][0]
        if self.gamma is None:
            self.gamma = old_gamma
        else:
            self.imported_gamma = old_gamma

        self.imported_gen_focuses = params_df["generation_focuses"].apply(
            ast.literal_eval
        )[0]
        self.imported_A = params_df["alignment"][0]
        # Need the extra handling to convert the csv's string back into an actual python list

    def create_generation_focuses(
        self, number_focuses=2, supplied_focuses=None, remove_focuses=True, gamma=None
    ):
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
            selected_focuses = np.random.choice(
                self.possible_focuses, size=number_focuses, replace=False
            )
            self.gen_focuses.append(selected_focuses.tolist())
        else:
            selected_focuses = supplied_focuses
            self.gen_focuses.append(selected_focuses)

        gen_counts = np.full(self.M, (self.beta / self.M))
        for focus in selected_focuses:
            gen_counts[focus] *= gamma
            if remove_focuses:
                # print(f"Debug, inst_focus is {focus}")
                self.possible_focuses.remove(focus)

        return gen_counts, selected_focuses

    def compile_speaker_info(self):
        """Compiles all the separate 1d speaker arrays into one speaker_info namedtuple, to make passing information to the jitted
        numba functions far easier to read.  Should be called before any numba functions (including the warmups!)"""
        return SpeakerInfo(self.speaker_gens, self.speaker_lambda, self.speaker_align)

    def numba_warmup(self):
        test_array = np.array([0, 1])
        dummy_matrix = np.full((self.S, self.M), 1.0 / self.S)
        dummy_alpha = np.full(self.M, self.beta / self.M)

        print("Initialising Numba Functions")
        print("Testing delta")
        delta(1, 1)
        print("Testing update_phi")
        update_phi_col(dummy_matrix, self.alpha, self.alpha_on_s, 1, self.S)

        print("Testing update_signal meaning")
        update_signal_meaning(dummy_matrix, self.lambda_val, 1, 1, self.S)

        print("Testing numba_random_choice")
        numba_random_choice_linear(test_array, test_array)

        print("Testing select_signal_integers_njit")
        _discard, rho_dist = select_signal_integers_njit(
            dummy_alpha, dummy_matrix, self.meanings_integers, self.signals_integers
        )

        print("Testing receive_signal_integers_njit")
        receive_signal_integers_njit(
            1, rho_dist, dummy_matrix, self.meanings_integers, self.M
        )

        print("Test blind success inner loop")
        blind_success_inner_loop(dummy_matrix, dummy_matrix, self.S, self.M)
        print("Test blind success outer loop")
        blind_success_outer_loop(self.num_agents, self.population_phi, self.S, self.M)

    def generate_DF(self,iterations):
        """Creates a pandas data frame to then be passed to the file saver parameter saving function """
        if hasattr(self, "means"):
            params_dict = {
                "num_meanings": self.M,
                "num_signals": self.S,
                "num_agents": self.num_agents,
                "lambda": self.lambda_val,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "generation_focuses": self.gen_focuses,
                "alignment": self.A,
                "Intergen_A": self.intergen_A,
                "iterations": iterations,
                "number_of_gens": self.num_gens,
                "imported_file": self.param_dir,
                "avg_map": [float(x) for x in self.avg_map],
                "gen_alphas": np.asarray(self.alpha_tensor).tolist(),
                "NUA_means": self.means,
                "NUA_stds": self.stds,
                "NUA_weight": self.weight,
            }
            print("NuA param df made")
        else:
            params_dict = {
                "num_meanings": self.M,
                "num_signals": self.S,
                "num_agents": self.num_agents,
                "lambda": self.lambda_val,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "generation_focuses": self.gen_focuses,
                "alignment": self.A,
                "Intergen_A": self.intergen_A,
                "iterations": iterations,
                "number_of_gens": self.num_gens,
                "imported_file": self.param_dir,
                "avg_map": [float(x) for x in self.avg_map],
                "gen_alphas": np.asarray(self.alpha_tensor).tolist(),
            }
        params_df = pd.DataFrame([params_dict])
        return params_df

    def add_speakers(self, speakers_added, speakers_focuses=None, align_vec=None):
        """A function to add speakers into the system.  Extends the ensemble tensors by the appropriate amount, and then
        fills them with the values corresponding to a new ('young'/'blank') speaker.  Additionally, adjusts both the alpha
        tensor, and the avg map list accordingly."""

        new_speakers_phi = np.full(
            (speakers_added, self.S, self.M), fill_value=1 / self.S
        )
        new_speakers_counts = np.zeros((speakers_added, self.S, self.M))

        self.population_phi = np.vstack((self.population_phi, new_speakers_phi))
        self.population_counts = np.vstack(
            (self.population_counts, new_speakers_counts)
        )

        self.avg_map.append(speakers_added)

        if speakers_focuses is not None:
            gen_counts, _selected_focuses = self.create_generation_focuses(
                number_focuses=2, supplied_focuses=speakers_focuses
            )
            # self.gen_focuses.append(selected_focuses)
            new_alpha = np.array(gen_counts)
        else:
            new_alpha = np.full((self.M), fill_value=(self.beta / self.M))

        self.alpha_tensor = np.vstack((self.alpha_tensor, new_alpha))
        self.num_agents += speakers_added
        self.num_gens += 1

        new_speaker_gens = np.full(speakers_added, self.num_gens, dtype=np.int8)
        new_speaker_lambda = np.full(speakers_added, self.lambda_val, dtype=np.float64)

        if align_vec is not None:
            # new_speaker_align = np.zeros((speakers_added,speakers_added), dtype=np.float64)
            new_speaker_align = align_vec
        else:
            new_speaker_align = np.full(speakers_added, self.A, dtype=np.float64)

        self.speaker_align = np.concat((self.speaker_align, new_speaker_align),axis = 0)
        self.speaker_lambda = np.concat((self.speaker_lambda, new_speaker_lambda))
        self.speaker_gens = np.concat((self.speaker_gens, new_speaker_gens))

    def measure_gen_gain(self, gen_gain_dict):
        """Function to measure the gain for systems with more than one generation.  It is valueable to
        know how the ensemble gain differs compared to the individual populations

        Ensemble data is always stored in the first dict list position, then generation data is stored in
        increasing gen number order
        """

        keys = gen_gain_dict.keys()
        num_gens = len(keys) - 1
        # print(f"Deug: num_gens is {num_gens}")

        p_s = blind_success_outer_loop(
            self.num_agents, self.population_phi, self.S, self.M
        )
        gain = ((self.M * p_s) - 1) / (self.S - 1)
        gen_gain_dict[0].append(gain)
        start_iter = 0
        for i in range(num_gens):
            end_iter = self.avg_map[i] + start_iter
            number_agents = end_iter - start_iter
            generation_phi = self.population_phi[start_iter:end_iter, :, :]
            p_s = blind_success_outer_loop(
                number_agents, generation_phi, self.S, self.M
            )
            gain = ((self.M * p_s) - 1) / (self.S - 1)
            gen_gain_dict[i + 1].append(gain)
            start_iter = end_iter

    def measure_froben_norm(self):
        """
        Takes frobenius norm of the matrix product between the generation speaker average phi 
        of two gens
        
        Measures the frobenius norm of the product of ensemble matrices.  Meant to provide another 
        way to evaluate the homogeneity of the generations' communication.

        """
        save_indices = False
        #Most of the values in this tensor will stay empty, this will not cost any memory however so not worried about overhead
        frob_norm_tens= np.empty((self.num_gens+1,self.num_gens+1))
        if not hasattr(self,"indices"):
            self.indices = []
            save_indices = True
        buffer = np.empty((len(self.avg_map),self.S,self.M))


        start_idx = 0
        for z in range(len(self.avg_map)):
            end_idx = self.avg_map[z] + start_idx
            buffer[z,:,:] = self.population_phi[start_idx:end_idx, :, :].mean(axis =0)

        rows,cols = np.triu_indices(self.num_gens+1,k=1) #K=1 to avoid self interactions
        for j in range(len(cols)):
            inst_product = buffer[rows[j],:,:] @ buffer[cols[j],:,:]
            frob_norm_tens[rows[j],cols[j]] =  np.linalg.norm(inst_product,ord='fro')
            if save_indices:
                self.indices.append([rows[j],cols[j]])
        return frob_norm_tens

    #############################################################################################################################################
    def add_intergen_alpha(self, gen_0_weight: float = 0.5, gen_1_weight: float = 0.5):
        """Creates a weighted intergenerational alpha distribution to be used in interactions between two separate populations/gens.
        The weights correspond to which generation's alpha vector is favored (default of 50/50), although this can be tweaked as needed
        to explore different situations.  For example, if one was interested in what occurs if there is a thought leader so to speak, who
        is a minority but dominates most social interactions, they could preferentially favor that vector via adjusting the weight accordingly.
        """
        assert self.alpha_tensor.shape[0] != 1, " Only one gen's alpha in tensor"
        inter_gen_alpha = (
            gen_0_weight * self.alpha_tensor[0] + gen_1_weight * self.alpha_tensor[1]
        )
        self.alpha_tensor = np.vstack((self.alpha_tensor, inter_gen_alpha))
        """Will need a way of thinking about how to do this in the case we have 3+ generations communicating, as it currently is written it would fail"""

    def one_interaction_one_pop_numba_test(self):
        print("Debug, testing one population numba tests")
        dummy_counts = np.zeros((self.num_agents, self.S, self.M))
        dummy_phi = np.full((self.num_agents, self.S, self.M), 1.0 / self.S)
        dummy_alpha = np.full(shape=(self.M), fill_value=self.beta / self.M)

        newphi, new_counts = one_interaction_one_population(
            dummy_counts,
            dummy_phi,
            self.meanings_integers,
            self.signals_integers,
            self.numba_consts,
            dummy_alpha,
            speaker_info=self.speaker_info,
        )

        print("Testing training loop")
        new_phi, new_counts = one_collection_loop_one_pop(
            dummy_counts,
            dummy_phi,
            self.meanings_integers,
            self.signals_integers,
            self.numba_consts,
            dummy_alpha,
            speaker_info=self.speaker_info,
            iterations=2,
        )

    def training_loop_one_pop(self):
        print("Caching required numba functions")

        self.speaker_info = SpeakerInfo(
            self.speaker_gens, self.speaker_lambda, self.speaker_align
        )
        self.numba_consts = NumbaConsts(
            self.num_agents, self.M, self.S, self.alpha, self.alpha_on_s
        )
        # self.one_interaction_one_pop_numba_test()
        self.gain_values = []

        """Need a check to collapse the alpha tensor into one vector if there are more than one alphas.  Logic for doing so is it 
        is easier to implement a check here, than to implement checks at every single instantiation of an agent/speaker group"""

        print(f"Debug: self.alpha_tensor.ndim is {self.alpha_tensor.ndim}")
        if self.alpha_tensor.ndim > 1:
            """If the alpha tensor is greater than 1, we have more than one alpha tensor, and thus it needs to be reduced to a single vector"""
            self.alpha_tensor = self.alpha_tensor.mean(axis=0)

        self.file_saver.generate_h5_file(self.num_gens,self.export_agents,self.num_agents) #trying the new file saving regime

        #Save initial data
        self.file_saver.write_to_h5(0,self.population_phi,self.population_counts,self.num_gens)

        for i in range(self.num_measurements):
            new_population_phi, new_population_counts = one_collection_loop_one_pop(
                self.population_counts,
                self.population_phi,
                self.meanings_integers,
                self.signals_integers,
                self.numba_consts,
                self.alpha_tensor,
                speaker_info=self.speaker_info,
                iterations=self.time_res,
            )
            self.population_phi = new_population_phi
            self.population_counts = new_population_counts
            p_s = blind_success_outer_loop(
                self.num_agents, self.population_phi, self.S, self.M
            )
            gain = ((self.M * p_s) - 1) / (self.S - 1)
            self.gain_values.append(gain)
            self.gain_epochs.append((i + 1) * self.time_res)

            print(f"Completed iteration {(i + 1) * self.time_res}")
            assert np.allclose(
                self.population_phi.sum(axis=1), 1.0
            )  # Check to make sure all the phi columns sum to 1

            #Write to file
            self.file_saver.write_to_h5(i,self.population_phi,self.population_counts,self.num_gens,gain_values=self.gain_values, gain_epochs=self.gain_epochs)

        if self.export_agents:
            self.file_saver.save_agents_data(self.num_agents,self.population_phi,self.population_counts)

        params_dataframe = self.generate_DF(iterations=self.tot_iterations)
        self.file_saver.save_params(params_dataframe)

    def numba_warmup_intergen(self, gen_dict, gen_choice):
        print("Performing numba warmups for intergen")
        dummy_counts = np.zeros((self.num_agents, self.S, self.M))
        dummy_phi = np.full((self.num_agents, self.S, self.M), 1.0 / self.S)
        dummy_alpha = np.full(
            shape=(self.num_gens + 1, self.M), fill_value=self.beta / self.M
        )
        # print(f"Debug : numba_warmup_intergen : dummy alpha is {dummy_alpha}")

        new_phi, new_counts = one_interaction_intergen(
            dummy_counts,
            dummy_phi,
            self.speaker_info,
            self.meanings_integers,
            self.signals_integers,
            self.numba_consts,
            dummy_alpha,
            intra_gen_alignment=1.0,
            intergen_alignment=1.0,
            num_gens=1,
            gen_dict=gen_dict,
            intergen_prob=0.0,
        )
        new_phi, new_counts = one_collection_loop_intergens(
            dummy_counts,
            dummy_phi,
            self.speaker_info,
            self.meanings_integers,
            self.signals_integers,
            self.numba_consts,
            dummy_alpha,
            iterations=2,
            intra_gen_alignment=1.0,
            intergen_alignment=1.0,
            gen_maps=gen_choice,
            gen_dict=gen_dict,
            intergen_prob=0.0,
        )

    def training_loop_intergen(self, intergen_prob):
        """Making a more explicit average map for the purposes of passing to the numba functions for better control over who interacts with whom
        Note this is being more or less hardcoded, in future
        """

        explicit_map = np.empty((len(self.avg_map), 2), dtype=np.int64)
        gen_dict = Dict.empty(key_type=types.int64, value_type=types.int64[:])

        
        start_idx = 0
        print(f"Debug: len(self.avg_map) is {len(self.avg_map)}")
        for z in range(len(self.avg_map)):
            end_idx = self.avg_map[z]
            end_val = start_idx + end_idx
            # explicit_map.append([start_idx,end_val])
            explicit_map[z, 0] = start_idx
            explicit_map[z, 1] = end_val
            gen_dict[z] = np.arange(start_idx, end_val)
            start_idx = end_val

        print(f"Debug: Explicit map is {explicit_map}")
        print(f"Debug: Gen Dict is {gen_dict}")

        print("Creating namedtuples for numba functions")
        self.speaker_info = SpeakerInfo(
            self.speaker_gens, self.speaker_lambda, self.speaker_align
        )
        self.numba_consts = NumbaConsts(
            self.num_agents, self.M, self.S, self.alpha, self.alpha_on_s
        )
        self.numba_buffers = NumbaBuffers(np.zeros(self.M,dtype=np.float64),np.zeros(self.S,dtype=np.float64),np.zeros(self.M,dtype=np.float64))


        print(
            f"Debug: speaker_info.lambda_val looks like {self.speaker_info.lambda_val}"
        )

        print(f"Debug: self.num_gens is {self.num_gens}")
        self.gens_gain = {i: [] for i in range(self.num_gens + 2)}
        # +2 because we need to include ensemble data as well
        print(f"Debug:self.gens_gains has keys {self.gens_gain.keys()}")

        print("Caching required numba functions")
        inst_gen_choice = np.arange(len(self.avg_map))

        print(f"Debug inst_gen_choice is {inst_gen_choice}")
        # self.numba_warmup_intergen(gen_dict=gen_dict, gen_choice=inst_gen_choice)

        self.file_saver.generate_h5_file(self.num_gens,self.export_agents,self.num_agents) #trying the new file saving regime
        
        #Save initial data
        self.file_saver.write_to_h5(0,self.population_phi,self.population_counts,self.num_gens)

        for i in range(self.num_measurements):
            new_population_phi, new_population_counts = one_collection_loop_intergens(
                self.population_counts,
                self.population_phi,
                self.speaker_info,
                self.meanings_integers,
                self.signals_integers,
                self.numba_consts,
                self.alpha_tensor,
                iterations=self.time_res,
                intra_gen_alignment=self.A,
                intergen_alignment=self.intergen_A,
                gen_maps=explicit_map,
                gen_dict=gen_dict,
                intergen_prob=intergen_prob,
                gen_choice = inst_gen_choice
            )
            self.population_phi = new_population_phi
            self.population_counts = new_population_counts


            self.measure_gen_gain(self.gens_gain)
            # print(f"Debug:Self.gens_gain[0] looks like {self.gens_gain[0]}")

            # frob_test = self.measure_froben_norm()
            # print(f"Debug: frob test is {frob_test}")

            self.gain_epochs.append((i + 1) * self.time_res)
            print(f"Completed iteration {(i + 1) * self.time_res}")
            self.file_saver.write_to_h5(i,self.population_phi,self.population_counts,self.num_gens,self.avg_map,
                                        gens_gain=self.gens_gain,
                                        gain_epochs=self.gain_epochs)
            assert np.allclose(
                self.population_phi.sum(axis=1), 1.0
            )  # Check to make sure all the phi columns sum to 1
        if self.export_agents:
            self.file_saver.save_agents_data(self.num_agents,self.population_phi,self.population_counts)


        params_df = self.generate_DF(self.tot_iterations)
        # self.save_params(iterations=self.tot_iterations)
        self.file_saver.save_params(params_df)

    def training_loop_NuA(self):
        self.speaker_info = SpeakerInfo(
            self.speaker_gens, self.speaker_lambda, self.speaker_align
        )
        self.numba_consts = NumbaConsts(
            self.num_agents, self.M, self.S, self.alpha, self.alpha_on_s
        )
        
        self.gain_values = []

        self.file_saver.generate_h5_file(self.num_gens,self.export_agents,self.num_agents) #trying the new file saving regime
        
        #Save initial data
        self.file_saver.write_to_h5(0,self.population_phi,self.population_counts,self.num_gens)

        for i in range(self.num_measurements):
            new_population_phi, new_population_counts = one_collection_loop_one_pop_NuA(
                self.population_counts,
                self.population_phi,
                self.meanings_integers,
                self.signals_integers,
                self.numba_consts,
                self.alpha_tensor,
                speaker_info=self.speaker_info,
                iterations=self.time_res,
            )
            self.population_phi = new_population_phi
            self.population_counts = new_population_counts
            p_s = blind_success_outer_loop(
                self.num_agents, self.population_phi, self.S, self.M
            )
            gain = ((self.M * p_s) - 1) / (self.S - 1)
            self.gain_values.append(gain)
            self.gain_epochs.append((i + 1) * self.time_res)

            print(f"Completed iteration {(i + 1) * self.time_res}")
            assert np.allclose(
                self.population_phi.sum(axis=1), 1.0
            )  # Check to make sure all the phi columns sum to 1

            self.file_saver.write_to_h5(i,self.population_phi,self.population_counts,self.num_gens,self.avg_map,
                                        gain_values=self.gain_values,
                                        gain_epochs=self.gain_epochs)

        if self.export_agents:
            self.file_saver.save_agents_data(self.num_agents,self.population_phi,self.population_counts)
        
        
        params_df = self.generate_DF(self.tot_iterations)
        # self.save_params(iterations=self.tot_iterations)
        self.file_saver.save_params(params_df)

        np.save(self.file_saver.output_dir/"align_matrix.npy",self.speaker_align) #since this is temporary just hardcode it
        print("Debug:Speaker align saved")

    """
    Now outdated, unlikely to be utilized, helpful to keep around in case they are of interest again

    def numba_warmup_multiple_gens(self):
        dummy_counts = np.zeros((self.num_agents,self.S, self.M))
        dummy_phi    = np.full((self.num_agents,self.S, self.M), 1.0/self.S)
        dummy_alpha  = np.full(shape=(self.num_gens+1,self.M), fill_value=self.beta/self.M)

        
        print("Debug, testing one interaction numba")
        new_phi,new_counts = one_interaction_multiple_gens(dummy_counts,dummy_phi,self.speaker_info,self.meanings_integers,
                                                            self.signals_integers,self.num_agents,dummy_alpha,self.M,
                                                            self.S,self.lambda_val,self.alpha,self.alpha_on_s)
        print("Debug, testing one data loop numba")
        new_phi,new_counts = one_collection_loop_multiple_gens(dummy_counts,dummy_phi,self.speaker_info,self.meanings_integers,
                                                                            self.signals_integers,self.num_agents,dummy_alpha,self.M,
                                                                        self.S,self.lambda_val,self.alpha, self.alpha_on_s, iterations=2)

        
    
    def training_loop_multiple_gens(self):
        print("Caching required numba functions")
        self.numba_warmup_multiple_gens()

        self.generate_initial_h5_file()
        for i in range(self.num_measurements):
            new_population_phi, new_population_counts = one_collection_loop_multiple_gens(self.population_counts,
                                                                    self.population_phi,self.speaker_info,self.meanings_integers,
                                                                   self.signals_integers,self.num_agents,self.alpha_tensor,self.M,
                                                                   self.S,self.lambda_val,self.alpha,self.alpha_on_s, iterations=self.time_res)
            self.save_to_file(idx_iter=i)
            self.population_phi = new_population_phi
            self.population_counts = new_population_counts
            print(f"Completed iteration {(i+1) * self.time_res}")
            assert np.allclose(self.population_phi.sum(axis=1), 1.0) #Check to make sure all the phi columns sum to 1
        
        if self.export_agents:
            self.save_agents_data()

        self.save_params(iterations = self.tot_iterations)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--NumMeanings",
        type=int,
        help="Determines the numbers of meanings to be created/used in the case where meanings are not"
        "hardcoded",
        default=12,
    )
    parser.add_argument(
        "--NumSignals",
        type=int,
        help="Determines the numbers of signals to be created/used in the case where signals are not"
        "hardcoded",
        default=12,
    )
    parser.add_argument(
        "--NumAgents",
        type=int,
        help="Determines the numbers of communicating/learning agents",
        default=5,
    )
    parser.add_argument(
        "--Beta", type=int, help="Determines the value of parameter beta", default=49
    )
    parser.add_argument(
        "--Alpha",
        type=float,
        help="Determines the value of parameter alpha",
        default=0.1,
    )
    parser.add_argument(
        "--lambda_val",
        type=float,
        help="Determines the value of lambda, or the forgetting rate of the models",
        default=0.01,
    )
    parser.add_argument(
        "--alignment",
        type=float,
        help="Determines the alignment between models",
        default=1.0,
    )
    parser.add_argument(
        "--GenAlign",
        type=float,
        help="Determines the alignment between models of different generations",
        default=0.50,
    )
    parser.add_argument(
        "--IntergenProb",
        type=float,
        help="Determines the frequency of intergen interactions",
        default=0.50,
    )

    parser.add_argument(
        "--iterations",
        type=float,
        help="Determines how many millions of iterations to run the code for",
        default=4.0,
    )
    parser.add_argument(
        "--OutputDir",
        type=str,
        help="Determines the file output of saved plots, data, etc",
        default=str(Path.home() / "Downloads" / "MPhys/Code_Runs"),
    )
    parser.add_argument(
        "--Filename",
        type=str,
        help="Determines the file to save data to",
        default="Unsorted",
    )
    parser.add_argument(
        "--ImportFile",
        type=str,
        help="Determines the file/folder to read the imported data from",
        default="Unsorted",
    )
    parser.add_argument(
        "--Generations",
        action="store_true",
        help="If set, have independent generations rho distributions",
    )
    parser.add_argument(
        "--GenCount", type=int, help="Determines the number of generations", default=1
    )
    parser.add_argument(
        "--gamma",
        type=int,
        help="Determines the parameter gamma, which controls \
                        how much the generational focus is multiplied by",
        default=2,
    )
    parser.add_argument(
        "--Focuses",
        type=int,
        help="Determines how many focuses are in each generation",
        default=2,
    )
    parser.add_argument(
        "--PassedFocuses",
        type=int,
        nargs="+",
        help="If given inputs, sets the specific focuses of the generation",
        default=None,
    )
    parser.add_argument(
        "--TimeRes",
        type=float,
        help="Determines how often to collect data, in multipliers of 1 hundred thousand",
        default=1.0,
    )
    parser.add_argument(
        "--Export", action="store_true", help="If set, exports the agents"
    )
    parser.add_argument(
        "--Import", action="store_true", help="If set, lets us import agents"
    )
    parser.add_argument(
        "--Means",
        type=float,
        nargs="+",
        help="If given inputs, sets the means for the bimodal distribution "
        "that alignments are drawn from in the non-uniform alignment case",
        default=[0.5, 0.8],
    )
    parser.add_argument(
        "--Stds",
        type=float,
        nargs="+",
        help="If given inputs, sets the stds for the bimodal distribution "
        "that alignments are drawn from in the non-uniform alignment case",
        default=[0.01, 0.05],
    )
    parser.add_argument(
        "--W",
        type=float,
        help="If given inputs, sets the weight for the bimodal distribution "
        "that alignments are drawn from in the non-uniform alignment case",
        default=0.5,
    )
    parser.add_argument(
        "--NonUniform",
        action="store_true",
        help="If set, we use non_uniform alignments",
    )

    args = parser.parse_args()
    print("Past args")

    if args.NonUniform:
        UniformA = False
    else:
        UniformA = True

    input_dir = Path(args.OutputDir) / args.ImportFile

    output_directory = Path(args.OutputDir) / args.Filename
    output_directory.mkdir(
        parents=True, exist_ok=True
    )  # Creates the folder if it does not already exist

    time_res = int(args.TimeRes * 100000)
    total_iterations = int(args.iterations * 1000000)

    baseconfig = SimulationBaseConfig(
        args.NumMeanings,
        args.NumSignals,
        args.NumAgents,
        args.alignment,
        args.Beta,
        args.Alpha,
        args.lambda_val,
        output_directory,
        total_iterations,
        args.Export,
        time_res,
    )

    genconfig = SimulationGensConfig(
        args.Generations,
        args.Import,
        input_dir,
        args.gamma,
        args.Focuses,
        generation_overlap=False,
        passed_focuses=args.PassedFocuses,
        intergenA=args.GenAlign,
    )

    noiseconfig = SimulationNoiseConfig(UniformA, args.Means, args.Stds, args.W)

    debugging_ensemble = Ensemble(baseconfig, genconfig, noiseconfig)
    print(f"debug ens has align info {debugging_ensemble.speaker_align}")

    # debugging_ensemble.generate_initial_h5_file()
    # print(f"Debug Shape of intergen_tensor = {debugging_ensemble.inter_gen_alpha.shape}, and is {debugging_ensemble.inter_gen_alpha}")
    train_start_time = time.time()

    # debugging_ensemble.training_loop_one_pop()
    debugging_ensemble.training_loop_NuA()


    # debugging_ensemble.add_speakers(args.NumAgents, 
    #                                 speakers_focuses=args.PassedFocuses
    #                                 )
    

    # debugging_ensemble.add_intergen_alpha(gen_0_weight=0.50, gen_1_weight=0.50)
    # debugging_ensemble.training_loop_intergen(args.IntergenProb)

    train_end_time = time.time()
    print(f"Runtime is {train_end_time - train_start_time}")
