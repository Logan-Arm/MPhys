import argparse
import ast
import json
import time
from functools import partial
from joblib import cpu_count,Parallel, delayed
from pathlib import Path

import h5py
import matplotlib


matplotlib.use('Agg') #Need to set matplotlib's backend to be non-interactive such that it can be done in parallel


import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numba
import numpy as np
import pandas as pd
import scipy as sc
import scipy.stats
import seaborn as sns
from numba import njit, prange

from MPhys_Plotter_File_Oct7th import File_Interpreter
from Nov13th_Cleaned import (
    Constructor,
    Ensemble,
    FileSaver,
    NumbaBuffers,
    NumbaConsts,
    SimulationBaseConfig,
    SimulationGensConfig,
    SimulationNoiseConfig,
    SimulationState,
)


def run_single_sim(args,inst_w:float,
                       inst_high_align:np.ndarray,
                       sub_dir:str,
                       time_res:int):
        """A function to be passed to joblib which iterates over all the high_align values runs the simulation in parallel."""

        means = [Low_align,inst_high_align]
        inst_filename = f"W_{inst_w:.3f}_low_{Low_align}_high{inst_high_align:.3f}"
        inst_output_directory = Path(sub_dir)/inst_filename
        inst_output_directory.mkdir(parents = True, exist_ok = True)

        if not args.Analysis:
            #Simulation loop
            inst_base_config = SimulationBaseConfig(
            args.NumMeanings,
            args.NumSignals,
            args.NumAgents,
            args.alignment,
            args.Beta,
            args.Alpha,
            args.lambda_val,
            inst_output_directory,
            total_iterations,
            args.Export,
            time_res,
            )
            inst_noiseconfig = SimulationNoiseConfig(UniformA, means, stds, inst_w)
            inst_ensemble = Ensemble(inst_base_config,genconfig,inst_noiseconfig)
            inst_ensemble.training_loop_NuA()      
            # inst_ensemble.training_loop_one_pop() 

        Plotter = File_Interpreter(inst_output_directory,args.SaveFig,args.ShowFig,inst_output_directory ,args.fps, args.Gens, NonUniform=True)
        Plotter.plot_final_phi()
        Plotter.plot_final_counts()
        Plotter.plot_gain()
        # Plotter.plot_NUA_dists(args.NumAgents)
        Plotter.generate_nx_plot()

        read_raw_dir = Path(inst_output_directory)/"raw"
        
        gain_df = pd.read_csv(read_raw_dir/"gain_values.csv")
        gain_vals = gain_df['gain']
        
        final_20_vals = np.array(gain_vals[-20:])
        print(f"Shape of final_20 is {final_20_vals.shape}")
        
        uncert = np.std(final_20_vals)/np.sqrt(args.NumAgents)
        return final_20_vals.mean(), uncert
    

def run_sim_for_w(args,inst_w:float,
                      high_align:np.ndarray,
                      output_dir:str,
                      time_res:float):
        """Runs simulation for all high_align values for a given inst_w in parallel"""
        sub_dir = Path(output_dir)/f"W_{inst_w:.3f}"
        sub_dir.mkdir(
        parents=True, exist_ok=True)

        results  = Parallel(n_jobs=num_cores)(delayed(run_single_sim)(args,inst_w,inst_high_align,sub_dir,time_res) for inst_high_align in high_align)


        final_gain_avg, final_gain_uncert = zip(*results) #* unpacks first, zip reorders
        np_final_gain_avg = np.array(final_gain_avg)
        np_final_gain_uncert = np.array(final_gain_uncert)
        fig, ax = plt.subplots(figsize = (10,6))
        ax.scatter(high_align,np_final_gain_avg, label = "Average")
        ax.errorbar(high_align, np_final_gain_avg,yerr=np_final_gain_uncert, label = "Standard Error", alpha = 0.4)
        ax.set_xlabel("High Alignment")
        ax.set_ylabel("Average of Metastable Gain")
        ax.set_title(f" High Alignment vs Avg of Metastable Gain at Weight {inst_w}")
        ax.legend()
        plt.savefig(Path(sub_dir)/"Avg_Gain.png")



def run_replica(args, total_iterations,time_res, gen_config,noise_config, replica_id, output_base_dir):
    """
    Runs a single simulation instance. 

    While we do not need to pass in a specific random seed since numba handles all its random variables on a per instance basis, 
    we do need to pass some sort of directory dientifier to avoid all ensemble members actively overwriting the data as it is being generated
    thus undoing the whole purpose of an ensemble scan

    """
    # Create a unique subdirectory for this replica to avoid overwrites
    replica_dir = Path(output_base_dir) / f"replica_{replica_id}"
    replica_dir.mkdir(parents=True, exist_ok=True)


    replica_base_config = SimulationBaseConfig(
                args.NumMeanings,
                args.NumSignals,
                args.NumAgents,
                args.alignment,
                args.Beta,
                args.Alpha,
                args.lambda_val,
                replica_dir,
                total_iterations,
                args.Export,
                time_res,
                )
    
    # Instantiate your simulation class here
    # Since numba handles seeding, no explicit seed argument is needed
    sim = Ensemble(
        replica_base_config,
        gen_config,
        noise_config, # Pass the unique directory to avoid overwriting
    )
    
    # Run simulation
    if not args.Analysis:
        sim.training_loop_NuA()
    Plotter = File_Interpreter(replica_dir,args.SaveFig,args.ShowFig,replica_dir ,args.fps, args.Gens, NonUniform=True)
    Plotter.plot_final_phi()
    Plotter.plot_gain()



    read_raw_dir = Path(replica_dir)/"raw"
    gain_df = pd.read_csv(read_raw_dir/"gain_values.csv")
    gain_vals = gain_df['gain']
    final_10_vals = np.array(gain_vals[-10:])
    print(f"Shape of final_20 is {final_10_vals.shape}")    
    if args.Clustering:
        inst_cluster = Plotter.analyse_clustering()
        return final_10_vals.mean(), inst_cluster


    return final_10_vals.mean() #Return the average of the last 10 samples gain treat this as averaging out the noise

def ensemble_scan(args,UniformA,stds: list, inst_w:float,low_align:float,inst_high_align:float,gens_config: SimulationGensConfig,sub_dir:str,
                  time_res:float, total_iterations:int):
    """Runs the ensemble scan over a specific configuration of high align and weights of high align"""


    ensemble_means = [low_align, inst_high_align]
    ensemble_noise_config = SimulationNoiseConfig(UniformA,ensemble_means,stds,weight=inst_w)

    #Generate the initial folder directory to hold all the ensemble information and the plots:
    sub_directory = Path(sub_dir)/f"HA_{inst_high_align}"
    sub_directory.mkdir(parents=True, exist_ok=True)
    replica_ids = np.arange(args.EnsembleNum)


    results = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(run_replica)(
            args, 
            total_iterations,
            time_res,
            gens_config,
            ensemble_noise_config,
            replica_id=i, 
            output_base_dir=sub_directory
        ) for i in replica_ids
    )

    if args.Cluster:
        avg_gain, avg_cluster = zip(*results)
        return avg_gain,avg_cluster

    else:
        return results
    
def total_scan(args,
               UniformA,
               scan_output_directory:str,
               total_iterations:int, 
               time_res:float,
               w_arr:np.ndarray,
               high_A_arr: np.ndarray,
               stds: list,
               low_align:float,
               gens_config:SimulationGensConfig):

    num_high_align = high_A_arr.shape[0]

    if args.AllScatter:
        num_weights = w_arr.shape[0]
        all_scatter_arr = np.empty((num_weights,num_high_align)) #[weights, high align] 
        #only store mean value, if we did the violin plot it would be impossible to interpret

        w_iter = 0

    for w in w_arr:
        inst_np_arr = np.empty((num_high_align,args.EnsembleNum)) #np array to store the results of the ensmble scan
        w_dir = Path(scan_output_directory)/f"W_{w}"

        if not args.Analysis: #If not args.Analysis we are collecting data, need to recapture it
            for iter, high_align in enumerate(high_A_arr):
                if args.Cluster: #ensemble_scan returns two values if the cluster argument is set
                    ensemble_scan_results, clusters =ensemble_scan(args,UniformA,stds,w,low_align,high_align,gens_config,w_dir,time_res,total_iterations) 
                else:
                    ensemble_scan_results = ensemble_scan(args,UniformA,stds,w,low_align,high_align,gens_config,w_dir,time_res,total_iterations)
                inst_np_arr[iter,:] = ensemble_scan_results #append to the list


        #Once the loop has been done we can save data and create the violin plots of the spectrum of ensemble results
        #First save the numpy array to a txt file so it can be accessed later, no reason to use h5py for this, numpy to txt should be fine since small array
        
            np.savetxt(Path(w_dir)/"ensemble_results.txt",inst_np_arr)
        else:
            inst_np_arr = np.loadtxt(Path(w_dir)/"ensemble_results.txt")
        #For the violin plot need to transpose the array
        plot_arr = inst_np_arr.T


        #Creating plots
        fig, ax = plt.subplots(figsize = (10,6))
        ax.violinplot(plot_arr,showmeans=False,showmedians=True)

        # Set x-ticks to correspond to the high_align values
        ax.set_xticks(range(1, num_high_align + 1))
        ax.set_xticklabels([f"{h:.2f}" for h in high_A_arr])

        ax.set_xlabel("High Alignment")
        ax.set_ylabel("Final Gain")
        ax.set_title(f"Violin Plot W = {w}")
        plt.savefig(Path(w_dir)/"violin.png")

        if args.AllScatter: #If plotting all the scatters, we need to go through and find the means for the given w
            for s in range(inst_np_arr.shape[0]):
                all_scatter_arr[w_iter,s] = inst_np_arr[s,:].mean() #Only stores the mean, see above for reasoning
            w_iter +=1

    if args.AllScatter:
        #Need to actually make the scatterplot now
        fig,ax = plt.subplots(figsize = (10,8)) 
        for s in range(num_weights):
            ax.scatter(high_A_arr,all_scatter_arr[s,:], label = f"Weight = {w_arr[s]:.2f}")
        ax.set_xlabel("High Alignment")
        ax.set_ylabel("Ensemble Average Final Gain")
        ax.set_title("Hyperparameter Scan Results")
        ax.legend()
        plt.savefig(Path(scan_output_directory)/"Hyperparam_Scatter.png")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--NumMeanings', type = int, help="Determines the numbers of meanings to be created/used in the case where meanings are not" \
    "hardcoded", default=12)
    parser.add_argument('--NumSignals', type = int, help="Determines the numbers of signals to be created/used in the case where signals are not" \
    "hardcoded", default=12)
    parser.add_argument('--NumAgents', type = int, help="Determines the numbers of communicating/learning agents", default=15)
    parser.add_argument('--EnsembleNum', type = int, help="Determines the number of ensemble instances to generate per configuration", default=8)


    parser.add_argument('--Beta', type = int, help="Determines the value of parameter beta", default=49)
    parser.add_argument('--Alpha', type = float, help="Determines the value of parameter alpha", default=0.1)
    parser.add_argument('--lambda_val', type = float, help="Determines the value of lambda, or the forgetting rate of the models", default=0.01)
    parser.add_argument('--alignment', type = float, help="Determines the alignment between models", default=0.85)
    parser.add_argument('--GenAlign', type = float, help="Determines the alignment between models of different generations", default=0.85)
    parser.add_argument('--iterations', type=float, help = "Determines how many millions of iterations to run the code for", default = 30.0)
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
    parser.add_argument('--Import', action='store_true', help='If set, lets us import agents')

    #Plotter args
    parser.add_argument('--ReadDir', type = str, help = 'Determines the read file location ', 
                            default =str(Path.home()/"Downloads"/"MPhys/Code_Runs"))
    parser.add_argument('--SaveFig', action='store_true', help='If set, saves figures to args.Filename')
    parser.add_argument('--ShowFig', action='store_true', help='If set, shows figures')
    parser.add_argument('--fps', type = int, help = "Determines the frames per second on videos", default =10)
    parser.add_argument('--Gens', action='store_true', help='If set, determines which parameter file reading to do')
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

    parser.add_argument(
        "--Analysis",
        action="store_true",
        help="If set, we don't actually run any simulations, but just do plotting and data analysis",
    )
    parser.add_argument('--AllScatter', action = "store_true", help = "If set,once we do the indiviudal scans/plots, we also a plot a " \
    "scatterplot of all possible combinations to make observing results easier ")

    parser.add_argument('--Cluster', action = "store_true", help = "If set,we also compute the average clustering of the network under the " \
    "specific configuration described ")


    
    args = parser.parse_args()


    num_cores = cpu_count()-1 #Keep 1 core free

    time_res = int(args.TimeRes * 100000)
    total_iterations = int(args.iterations * 1000000)

    input_dir = Path(args.OutputDir)/args.ImportFile


    if args.NonUniform:
        UniformA = False
    else:
        UniformA = True


    output_directory = Path(args.OutputDir) / args.Filename
    output_directory.mkdir(
        parents=True, exist_ok=True
    )  # Creates the folder if it does not already exist

    
    #Sits outside, we are not doing anything with gens on this run
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


    final_gain_avg =[]
    final_gain_uncert = []

    high_align = np.linspace(0.75, .95, num = 5)
    # high_align_val = 0.85
    Low_align = 0.65
    # means = [Low_align,high_align_val]
    stds = [0.1,0.1]

    # W = 0.2

    W_s = np.linspace(0.2,0.6,num = 4)





    start_time = time.time()

    # for inst_w in W_s:
    #     run_sim_for_w(args,inst_w,high_align,output_directory,time_res)

    total_scan(args,UniformA,output_directory,total_iterations,time_res,W_s,high_align,stds,Low_align,genconfig)

    end_time = time.time()
    print(f"Parameter Scan runtime was {end_time-start_time}")
