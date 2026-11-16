import argparse
import ast
import json
import time
from functools import partial
from joblib import Parallel, delayed
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--NumMeanings', type = int, help="Determines the numbers of meanings to be created/used in the case where meanings are not" \
    "hardcoded", default=12)
    parser.add_argument('--NumSignals', type = int, help="Determines the numbers of signals to be created/used in the case where signals are not" \
    "hardcoded", default=12)
    parser.add_argument('--NumAgents', type = int, help="Determines the numbers of communicating/learning agents", default=15)
    parser.add_argument('--Beta', type = int, help="Determines the value of parameter beta", default=49)
    parser.add_argument('--Alpha', type = float, help="Determines the value of parameter alpha", default=0.1)
    parser.add_argument('--lambda_val', type = float, help="Determines the value of lambda, or the forgetting rate of the models", default=0.01)
    parser.add_argument('--alignment', type = float, help="Determines the alignment between models", default=0.85)
    parser.add_argument('--GenAlign', type = float, help="Determines the alignment between models of different generations", default=0.85)
    parser.add_argument('--iterations', type=float, help = "Determines how many millions of iterations to run the code for", default = 150.0)
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


    
    args = parser.parse_args()


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

    high_align = np.linspace(0.70, 1.0, num = 10)
    # high_align_val = 0.85
    Low_align = 0.65
    # means = [Low_align,high_align_val]
    stds = [0.1,0.1]

    # W = 0.2

    W_s = np.linspace(0.2,0.7,num = 10)


    def run_simulation(args,
                       inst_w:float,
                       high_align:np.ndarray,
                       output_dir_base:str,
                       BaseConfig:SimulationBaseConfig,
                       GensConfig:SimulationGensConfig,
                       iterations:int,
                       time_res:int):
        """A function to be passed to joblib which iterates over all the high_align values runs the simulation in parallel."""

    start_time = time.time()
    for i in range(W_s.shape[0]):
        inst_w = W_s[i]
        sub_dir = Path(output_directory)/f"W_{inst_w:.3f}"
        sub_dir.mkdir(
        parents=True, exist_ok=True)

        final_gain_avg =[]
        final_gain_uncert = []


        for j in range(high_align.shape[0]):
            inst_high_align = high_align[j]
            means = [Low_align, inst_high_align]
            inst_filename = f"W_{inst_w:.3f}_low_{Low_align}_high{inst_high_align:.3f}"
            inst_output_directory = Path(sub_dir)/inst_filename
            inst_output_directory.mkdir(parents = True, exist_ok = True)#Creates the folder if it does not already exist

            if not args.Analysis:
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
            final_gain_avg.append(final_20_vals.mean())
            uncert = np.std(final_20_vals)/np.sqrt(args.NumAgents)
            final_gain_uncert.append(uncert)


        np_buffer_avg = np.array(final_gain_avg)
        np_buffer_uncert = np.array(final_gain_uncert)
        fig,ax = plt.subplots(figsize = (10,6))
        ax.scatter(high_align,final_gain_avg, label = "Average")
        ax.errorbar(high_align, np_buffer_avg,yerr=np_buffer_uncert, label = "Standard Error", alpha = 0.4)
        ax.set_xlabel("High Alignment")
        ax.set_ylabel("Average of Metastable Gain")
        ax.set_title(f" High Alignment vs Avg of Metastable Gain at Weight {inst_w}")
        ax.legend()
        plt.savefig(Path(sub_dir)/"Avg_Gain.png")

    end_time = time.time()
    print(f"Parameter Scan runtime was {end_time-start_time}")