import argparse
import ast
from functools import partial
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib import animation


class File_Interpreter:
    def __init__(self, raw_dir, save_fig,show_fig,save_dir,fps, gens= False,NonUniform = False):
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

        if NonUniform:
            self.read_param_fileNUA()


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
        self.num_gens = params_df['number_of_gens'][0]+1
        print(f"Generation focuses are {self.gen_focuses}")
        
        
        self.A = params_df['alignment'][0]
        #Need the extra handling to convert the csv's string back into an actual python list

        self.avgs_map = params_df['avg_map'].apply(ast.literal_eval)
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

    def read_param_fileNUA(self):
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
        
        self.A_vec = params_df["NUA_AVec"].apply(ast.literal_eval)[0]
        print(f"Debug: read A_vec is {self.A_vec}")
        self.nua_means = params_df["NUA_means"].apply(ast.literal_eval)[0]
        self.nua_stds = params_df["NUA_stds"].apply(ast.literal_eval)[0]
        self.nua_weights = params_df["NUA_weights"].apply(ast.literal_eval)[0]
        #Need the extra handling to convert the csv's string back into an actual python list


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

    def plot_NUA_dists(self,samples):
        """Creates plots of the bimodal NUA distributions for visual inspection"""
        print(f"Debug: means {self.nua_means}, stds {self.nua_stds}, weights {self.nua_weights}")
        num_choices = len(self.nua_means)
        components = np.random.choice(num_choices, size = samples, p =self.nua_weights)
        samples = np.random.normal(np.take(self.nua_means,components), np.take(self.nua_stds,components)) #vectorises the pull so faster for large N
        fig, ax = plt.subplots(figsize = (10,6))
        ax.hist(samples, bins = 20)
        ax.set_xlabel("Alignment")
        ax.set_ylabel("Counts")
        ax.set_title(f"Histogram of {len(samples)} draws from bimodal distribution")
        title= "NUA"
        self.plot_save_or_show(fig,title)

        

    def plot_dirichlet_dists(self,num_samples):
        len_alpha = len(self.gen_alphas[0])
        meanings = np.arange(len_alpha)
        
        for i in range(self.num_gens):
            
            inst_alpha = self.gen_alphas[i]
            arr = np.full(shape = len_alpha, fill_value=inst_alpha)
            dirichlet_samples = scipy.stats.dirichlet.rvs(arr, size = num_samples)

            magnified_samples = dirichlet_samples*100

            print(f"Shape of dirichlet samples is {dirichlet_samples.shape}")

            fig, axes = plt.subplots(nrows = 2, ncols =4, figsize =(16,8), layout = "constrained")
            axes = axes.flatten()
            for k, ax in enumerate(axes):
                ax.bar(meanings,magnified_samples[k,:],width = 0.8, edgecolor = 'black')
                ax.set_xticks(meanings) 
                ax.set_xlabel("Meanings")
                ax.set_ylabel("100x Attention Draw")

            fig.suptitle(f"Magnified Dirichlet Draws for Generation {i}")

            self.plot_save_or_show(fig,name = f"gen_{i}_dirichlet_visual")

        """Now doing the intergen attentional focuses"""
        
        intergen_alphas = self.gen_alphas[-1]
        arr = np.full(shape = len_alpha, fill_value=intergen_alphas)
        dirichlet_samples = scipy.stats.dirichlet.rvs(arr, size = num_samples)

        magnified_samples = dirichlet_samples*100

        print(f"Shape of dirichlet samples is {dirichlet_samples.shape}")

        fig, axes = plt.subplots(nrows = 2, ncols =4, figsize =(16,8), layout = "constrained")
        axes = axes.flatten()
        for k, ax in enumerate(axes):
            ax.bar(meanings,magnified_samples[k,:],width = 0.8, edgecolor = 'black')
            ax.set_xticks(meanings) 
            ax.set_xlabel("Meanings")
            ax.set_ylabel("100x Attention Draw")
        fig.suptitle("Magnified Dirichlet Draws for Intergen Attentional Focus")
        self.plot_save_or_show(fig,name = "intergen_dirichlet_visual")

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
            name = "Final_Phi"
            self.plot_save_or_show(fig,name)
        f.close()

    def plot_final_phi_gens(self, highlight_focuses = True):
        """Currently hardcoded to only work with two generations, for further generations, more flexible methodology is needed"""
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize = (18,9))
        with h5py.File(self.read_raw_dir/"array_data.h5", 'r') as f:
            ensemble_array= f['Phi']['phi_gens_dataset']
            print(f"Shape of the ensemble array is {ensemble_array.shape}")

            phi_array = f['Phi']['phi_gens_dataset'][:,0,:,:]
            final_count = phi_array[-1,:,:]
            sns.heatmap(final_count,ax = ax1,cmap = 'coolwarm',vmax = 1.0)
            ax1.set_xlabel("Meanings")
            ax1.set_ylabel("Signals")
            ax1.set_title("Generation 0")    
            self.highlight_focuses(ax1,gen = 0)

            final_phi_2_array =f['Phi']['phi_gens_dataset'][-1,1,:,:] 
            sns.heatmap(final_phi_2_array,ax = ax2,cmap = 'coolwarm',vmax =1.0)
            ax2.set_xlabel("Meanings")
            ax2.set_ylabel("Signals")
            ax2.set_title("Generation 1")    
            self.highlight_focuses(ax2,gen = 1)

            name = "Final_Phi_Both_Gens"
            self.plot_save_or_show(fig,name)


    def plot_final_counts(self, highlight_focuses = True):
        with h5py.File(self.read_raw_dir/"array_data.h5", 'r') as f:
            final_count = f['counts']['counts_dataset'][-1,:,:]
            print(f"final_count max and min vals are {np.max(final_count)} {np.min(final_count)}")
            fig,ax = plt.subplots(figsize = (10,8))
            sns.heatmap(final_count,cmap = 'coolwarm')
            ax.set_title("Final Recorded Ensemble Average Counts")
            ax.set_xlabel("Meaning")
            ax.set_ylabel("Signal")
            self.highlight_focuses(ax,highlights=self.highlight_focuses)
            name = "Final_Counts"
            self.plot_save_or_show(fig,name)

    def plot_gain(self):
        """Plotter function to plot the gain functions"""
        gain_df = pd.read_csv(self.read_raw_dir/"gain_values.csv")
        gain_vals = gain_df['gain']
        print(f"Debug: gain vals have length {len(gain_vals)}")

        
        epochs = gain_df['epoch']
        print(f"Debug: Gain epochs has leng {len(epochs)}")
        fig, ax = plt.subplots(figsize = (10,8))
        ax.plot(epochs,gain_vals)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Communication Gain")
        ax.set_title("Communication Gain vs Iterations")
        name = "Gain"
        self.plot_save_or_show(fig,name)
        ax.set_yscale('log')
        name = "Log_Gain"
        self.plot_save_or_show(fig,name)

    def plot_gain_gens(self):
        """Simulation runs with generations will save the gain values to a 
        dict as opposed to a list, as such a separate function is necessary"""
        gain_df = pd.read_csv(self.read_raw_dir/"gen_gain_values.csv")
        ensemble_gain = gain_df['ensemble_gain']
        gen_0_gain = gain_df['gen_0_gain']
        gen_1_gain = gain_df['gen_1_gain']
        gain_epochs = gain_df['epoch']

        fig, ax = plt.subplots(figsize = (10,8))
        ax.plot(gain_epochs,ensemble_gain, label = "Ensemble")
        ax.plot(gain_epochs,gen_0_gain, label = "Gen 0")
        ax.plot(gain_epochs,gen_1_gain, label = "Gen 1")
        ax.legend()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Communication Gain")
        ax.set_title("Communication Gain vs Iterations")
        name = "Gen_Gain"
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
            ax.set_title("Ensemble Phi Array")

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
                    
            
            counts_array = f['counts']['counts_dataset'][:,:,:]
            print(f"Ensemble counts array is of shape {counts_array.shape}")
            num_frames = counts_array.shape[0]
            fig,ax = plt.subplots(figsize = (10,8))
            im = ax.imshow(counts_array[0,:,:], cmap = 'coolwarm', animated = True,vmin=counts_array.min(), vmax=counts_array.max())
            ax.set_xlabel("Meanings")
            ax.set_ylabel("Signal")
            fig.colorbar(im, ax = ax)
            ax.set_title("Ensemble Counts Array")
            ax.set_xticks(np.arange(self.num_meanings))
            self.highlight_focuses(ax,highlights=self.highlight_focuses)
 
            #Partially fill the parameters of the update_array function using functools partials
            update_func = partial(self.update_array_animate,data_array = counts_array, imshow_array = im,
                                    time_mult = 1)
            
            
            anim  = animation.FuncAnimation(fig,update_func,frames = num_frames,blit =True, repeat = False)

            name = "Counts_animation"
            self.anim_save_or_show(fig,anim,name)

    def generate_nx_plot(self):
        """Generates a plot of the network using networkx, where connections between graph nodes 
        represent the strength (magnitude) of the alignment in the given interaction.  The procedure 
        for generating the alignment matrix allows values along diagonal (i.e., a self interaction
        which is not possible), thus we need to manually remove these values """

        alignment_matrix = np.load(self.read_dir/"align_matrix.npy")
        np.fill_diagonal(alignment_matrix,0.0) #all main diagonal elements are 0 so as to neglect self interaction

        scaled_alignment  = alignment_matrix #scale values up so difference on plots actually shows up
        # G=nx.DiGraph()
        G = nx.from_numpy_array(scaled_alignment)

        #If want non-symmetric Alignment matrix use the following
        # G = nx.from_numpy_array(scaled_alignment, create_using=nx.DiGraph)

        alignments_flattened  = [scaled_alignment[i,j] for i,j in G.edges()]

        pos = nx.spring_layout(G, weight="weight")
        

        fig, ax = plt.subplots(figsize = (10,6))
        nx.draw_networkx_nodes(G,pos = pos, node_color = 'lightblue', ax=ax)
        nx.draw_networkx_labels(G, pos = pos, ax=ax)
        nx.draw_networkx_edges(G,pos = pos,width = alignments_flattened,edge_color=alignments_flattened, 
                               edge_cmap=plt.cm.coolwarm,ax=ax,arrows=True, connectionstyle="arc3,rad=0.15") 
        #connection style curves lines a bit--> easier to spot alignment in each direction
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(0, 1))
        fig.colorbar(sm, ax=ax, label="Alignment")
        
        self.plot_save_or_show(fig,name = "Network")

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
    parser.add_argument('--NonUniform', action='store_true', help='If set, determines which parameter file reading to do')
    parser.add_argument('--Samples', type = int, help = "Determines the number of samples to draw for the bimodal distribution", default =1000)


    args = parser.parse_args()

    
    output_directory = Path(args.OutputDir)/args.Filename
    output_directory.mkdir(parents = True, exist_ok = True)#Creates the folder if it does not already exist
    read_directory = output_directory

    Plotter = File_Interpreter(read_directory,args.SaveFig,args.ShowFig,output_directory, args.fps, args.Gens,args.NonUniform)

    # if args.Gens:
    #     Plotter.plot_final_phi()
    #     Plotter.plot_final_counts()
    #     Plotter.animate_phi_gens()
    #     Plotter.animate_counts()    
    #     Plotter.animate_phi()
    #     Plotter.plot_dirichlet_dists(num_samples=8)
        
    #     Plotter.plot_gain_gens()
    #     # Plotter.plot_gain()
    #     Plotter.plot_final_phi_gens()
    #     if args.NonUniform:
    #         Plotter.plot_NUA_dists(args.Samples)
    # else:
    #     Plotter.plot_final_phi()
    #     Plotter.plot_final_counts()
    #     Plotter.animate_counts()
    #     Plotter.animate_phi()
    #     Plotter.plot_gain()
    #     if args.NonUniform:
    #         Plotter.plot_NUA_dists(args.Samples)

    Plotter.generate_nx_plot()
