import numpy as np
import scipy as sc
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numba
from numba import njit, prange
import argparse
from pathlib import Path
import pandas as pd
import json
import h5py
from functools import partial
import ast


class File_Interpreter:
    def __init__(self, raw_dir, save_fig,show_fig,save_dir,fps, gens= False):
        self.read_dir = Path(raw_dir)
        self.read_raw_dir = self.read_dir/"raw"
        self.save_dir = Path(save_dir)
        self.show = show_fig
        self.save = save_fig
        self.fps = fps


        if gens:
            self.read_param_file_gens()
        else:
            self.read_param_file()
        
        self.gen_colors = ["red","blue","green"]
        #This is hardcoded, so obviously not ideal, but I doubt I will ever need more than 3 gens per plots


    def read_param_file_gens(self):
        """Reads the inputted parameter.csv file and updates/records the relevant ensemble parameter values accordingly"""
        params_df = pd.read_csv(self.read_dir/"simulation_parameters.csv")
        self.num_meanings = params_df['num_meanings'][0]
        # print(self.num_meanings)
        self.num_signals = params_df['num_signals'][0]
        self.num_imported_agents = params_df['num_agents'][0]
        self.imported_gens_lambda = params_df['lambda'][0]
        self.beta = params_df['beta'][0]
        self.gamma = params_df['gamma'][0]
        self.gen_focuses = params_df['generation_focuses'].apply(ast.literal_eval)[0]
        print(f"Generation focuses are {self.gen_focuses}")
        self.num_gens = len(self.gen_focuses)
        
        self.A = params_df['alignment'][0]
        #Need the extra handling to convert the csv's string back into an actual python list
        self.certainty = params_df['certainty'][0]

        self.avgs_map = params_df['avg_map'].apply(ast.literal_eval)[0]
        print(f"Averages map is {self.avgs_map}")



        self.gen_alphas = params_df['gen_alphas'].apply(ast.literal_eval)[0]
        # print(f"gen alphas is {self.gen_alphas}")
        # print(f"Length of gen alphas is {len(self.gen_alphas)} ")
        # print(f"First gen alpha is {self.gen_alphas[0]}")
        # print(f"Second gen alpha is {self.gen_alphas[1]}")

    def read_param_file(self):
        """Reads the inputted parameter.csv file and updates/records the relevant ensemble parameter values accordingly"""
        params_df = pd.read_csv(self.read_dir/"simulation_parameters.csv")
        self.num_meanings = params_df['num_meanings'][0]
        # print(self.num_meanings)
        self.num_signals = params_df['num_signals'][0]
        self.num_imported_agents = params_df['num_agents'][0]
        self.imported_gens_lambda = params_df['lambda'][0]
        self.beta = params_df['beta'][0]
        self.gamma = params_df['gamma'][0]
        self.gen_focuses = params_df['generation_focuses'].apply(ast.literal_eval)[0]
        self.num_gens = len(self.gen_focuses)
        
        self.A = params_df['alignment'][0]
        #Need the extra handling to convert the csv's string back into an actual python list
        self.certainty = params_df['certainty'][0]

    def plot_save_or_show(self, fig, name):
        """A helper function to be called after creating a figure in any of the below functions.  Determines what to do with the created figure"""
        if self.save:
            fig.savefig(self.save_dir / f"{name}.png")
        if self.show:
            plt.show()
        plt.close(fig)

    def anim_save_or_show(self, fig,anim, name):
        """A helper function to be called after creating a figure in any of the below functions.  Determines what to do with the created figure"""
        if self.save:
            anim.save(self.save_dir / f"{name}.gif", writer = 'pillow', fps = self.fps)
        if self.show:
            plt.show()
        plt.close(fig)

    def highlight_focuses(self,ax,highlights = True, gen = None):
        """A helper function to determine whether or not generational focuses should be highlighted
         on heatmap plots """
        if highlights:
            if gen is None:
                for i in range(self.num_gens):
                    inst_gen_focuses = self.gen_focuses[i]
                    for label_id, label in enumerate(ax.get_xticklabels()):
                        if label_id in inst_gen_focuses:
                            label.set_color(self.gen_colors[i])
                            label.set_fontweight("bold")
            else:
                inst_gen_focuses = self.gen_focuses[gen]
                for label_id, label in enumerate(ax.get_xticklabels()):
                    if label_id in inst_gen_focuses:
                        label.set_color(self.gen_colors[gen])
                        label.set_fontweight("bold")

    def plot_dirichlet_dists(self,num_toys):
        len_alpha = len(self.gen_alphas[0])
        meanings = np.arange(len_alpha)
        
        for i in range(self.num_gens):
            
            inst_alpha = self.gen_alphas[i]
            arr = np.full(shape = len_alpha, fill_value=inst_alpha)
            dirichlet_samples = scipy.stats.dirichlet.rvs(arr, size = num_toys)
            print(f"Shape of dirichlet samples is {dirichlet_samples.shape}")
            avg = dirichlet_samples.mean(axis = 0)
            print(f"Dirichlet sample average is {avg}")

            magnified_counts = (100*avg).astype(int)
            print(magnified_counts)

            fig,ax = plt.subplots(figsize =(10,6))
            ax.bar(meanings,magnified_counts,width = 0.8, edgecolor = 'black')
            ax.set_ylabel("100x Average Attention Prob")
            ax.set_xticks(meanings)
            ax.set_xlabel(f"Meaning")
            ax.set_title(f"Average of {num_toys} Dirichlet Draws Multiplied by 100")
            self.plot_save_or_show(fig,name = f"gen_{i}_dirichlet_visual")

    def plot_final_phi(self, highlight_focuses = True):
        with h5py.File(self.read_raw_dir/"array_data.h5", 'r') as f:
            final_phi = f['Phi']['phi_dataset'][-1,:,:]
            print(f"final_phi max and min vals are {np.max(final_phi)} {np.min(final_phi)}")
            fig,ax = plt.subplots(figsize = (10,8))
            sns.heatmap(final_phi,cmap = 'coolwarm')
            ax.set_title("Final Recorded Ensemble Average Phi")
            ax.set_xlabel("Meaning")
            ax.set_ylabel("Signal")
            self.highlight_focuses(ax,highlights=self.highlight_focuses)
            name = f"Final_Phi"
            self.plot_save_or_show(fig,name)
    
    def plot_final_counts(self, highlight_focuses = True):
        with h5py.File(self.read_raw_dir/"array_data.h5", 'r') as f:
            final_count = f['signal_meaning']['signal_meanings_dataset'][-1,:,:]
            print(f"final_count max and min vals are {np.max(final_count)} {np.min(final_count)}")
            fig,ax = plt.subplots(figsize = (10,8))
            sns.heatmap(final_count,cmap = 'coolwarm')
            ax.set_title("Final Recorded Ensemble Average Counts")
            ax.set_xlabel("Meaning")
            ax.set_ylabel("Signal")
            self.highlight_focuses(ax,highlights=self.highlight_focuses)
            name = f"Final_Counts"
            self.plot_save_or_show(fig,name)

    def animate_phi(self):
        with h5py.File(self.read_raw_dir/"array_data.h5", 'r') as f:
            
            
            phi_array = f['Phi']['phi_dataset'][:,:,:]
            
            num_frames = phi_array.shape[0]
            fig,ax = plt.subplots(figsize = (10,8))
            im = ax.imshow(phi_array[0,:,:], cmap = 'coolwarm', animated = True,vmin=phi_array.min(), vmax=phi_array.max())
            ax.set_xlabel("Meanings")
            ax.set_ylabel("Signal")
            fig.colorbar(im, ax = ax)
            ax.set_title(f"Ensemble Phi Array")

            ax.set_xticks(np.arange(self.num_meanings))
            #Need to set the number of x ticks to be exactly the number of possible x vals, otherwise the highlighter won't work 
            self.highlight_focuses(ax,highlights=self.highlight_focuses)

            
            #Partially fill the parameters of the update_array function using functools partials
            update_func = partial(self.update_array_animate,data_array = phi_array, imshow_array = im,
                                  time_mult = 1)
            
            
            anim  = animation.FuncAnimation(fig,update_func,frames = num_frames,blit =True, repeat = False)

            name = "Phi_animation"
            self.anim_save_or_show(fig,anim,name)

    def animate_phi_gens(self):
        with h5py.File(self.read_raw_dir/"array_data.h5", 'r') as f:
            ensemble_array= f['Phi']['phi_gens_dataset']
            print(f"Shape of the ensemble array is {ensemble_array.shape}")
            for i in range(self.num_gens):
                phi_array = f['Phi']['phi_gens_dataset'][:,i,:,:]
                print(f"shape of inst phi array is {phi_array.shape}")
                
                num_frames = phi_array.shape[0]
                fig,ax = plt.subplots(figsize = (10,8))
                im = ax.imshow(phi_array[0,:,:], cmap = 'coolwarm', animated = True,vmin=phi_array.min(), vmax=phi_array.max())
                ax.set_xlabel("Meanings")
                ax.set_ylabel("Signal")
                fig.colorbar(im, ax = ax)
                ax.set_title(f"Phi Array for Gen {i}")

                ax.set_xticks(np.arange(self.num_meanings))
                #Need to set the number of x ticks to be exactly the number of possible x vals, otherwise the highlighter won't work 
                self.highlight_focuses(ax,gen=i)

                
                #Partially fill the parameters of the update_array function using functools partials
                update_func = partial(self.update_array_animate,data_array = phi_array, imshow_array = im,
                                    time_mult = 1)
                
                
                anim  = animation.FuncAnimation(fig,update_func,frames = num_frames,blit =True, repeat = False)

                name = f"Phi_gen_{i}_animation"
                self.anim_save_or_show(fig,anim,name)


    def update_array_animate(self,frame, data_array,imshow_array, time_mult):
        imshow_array.set_array(data_array[frame * time_mult,:,:])
        # imshow_array.set_title(f"Phi array at frame {frame}")
        # title = imshow_array.axes.set_title(f"Phi array at frame {frame}")
        return [imshow_array]

    def animate_counts(self):
        with h5py.File(self.read_raw_dir/"array_data.h5", 'r') as f:
                    
            
            counts_array = f['signal_meaning']['signal_meanings_dataset'][:,:,:]
            print(f"Ensemble counts array is of shape {counts_array.shape}")
            num_frames = counts_array.shape[0]
            fig,ax = plt.subplots(figsize = (10,8))
            im = ax.imshow(counts_array[0,:,:], cmap = 'coolwarm', animated = True,vmin=counts_array.min(), vmax=counts_array.max())
            ax.set_xlabel("Meanings")
            ax.set_ylabel("Signal")
            fig.colorbar(im, ax = ax)
            ax.set_title(f"Ensemble Counts Array")
            ax.set_xticks(np.arange(self.num_meanings))
            self.highlight_focuses(ax,highlights=self.highlight_focuses)
 
            #Partially fill the parameters of the update_array function using functools partials
            update_func = partial(self.update_array_animate,data_array = counts_array, imshow_array = im,
                                    time_mult = 1)
            
            
            anim  = animation.FuncAnimation(fig,update_func,frames = num_frames,blit =True, repeat = False)

            name = "Counts_animation"
            self.anim_save_or_show(fig,anim,name)



if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--OutputDir', type = str, help = 'Determines the file output of saved plots, data, etc', 
                            default =str(Path.home()/"Downloads"/"MPhys/Code_Runs"))
    parser.add_argument('--ReadDir', type = str, help = 'Determines the read file location ', 
                            default =str(Path.home()/"Downloads"/"MPhys/Code_Runs"))
    parser.add_argument('--Filename', type = str, help='Determines the file to save data to', default= 'Unsorted')
    parser.add_argument('--SaveFig', action='store_true', help='If set, saves figures to args.Filename')
    parser.add_argument('--ShowFig', action='store_true', help='If set, shows figures')
    parser.add_argument('--fps', type = int, help = "Determines the frames per second on videos", default =10)
    parser.add_argument('--Gens', action='store_true', help='If set, determines which parameter file reading to do')
    args = parser.parse_args()

    
    output_directory = Path(args.OutputDir)/args.Filename
    output_directory.mkdir(parents = True, exist_ok = True)#Creates the folder if it does not already exist
    read_directory = output_directory

    Plotter = File_Interpreter(read_directory,args.SaveFig,args.ShowFig,output_directory, args.fps, args.Gens)

    if args.Gens:
        Plotter.plot_final_phi()
        Plotter.plot_final_counts()
        Plotter.animate_phi_gens()
        Plotter.animate_counts()
        Plotter.animate_phi()
        Plotter.plot_dirichlet_dists(num_toys=100)
    else:
        Plotter.plot_final_phi()
        Plotter.plot_final_counts()
        Plotter.animate_counts()
        Plotter.animate_phi()
