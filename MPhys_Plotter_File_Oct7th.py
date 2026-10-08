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

class File_Interpreter:
    def __init__(self, raw_dir, save_fig,show_fig,save_dir):
        self.read_dir = Path(raw_dir)
        self.read_raw_dir = self.read_dir/"raw"
        self.save_dir = Path(save_dir)
        self.show = show_fig
        self.save = save_fig



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
            anim.save(self.save_dir / f"{name}.gif", writer = 'pillow', fps = 20)
        if self.show:
            plt.show()
        plt.close(fig)



    def plot_final_phi(self):
        with h5py.File(self.read_raw_dir/"array_data.h5", 'r') as f:
            final_phi = f['Phi']['phi_dataset'][-1,:,:]
            fig,ax = plt.subplots(figsize = (10,8))
            sns.heatmap(final_phi,cmap = 'coolwarm')
            ax.set_title("Final Recorded Ensemble Average Phi")
            ax.set_xlabel("Meaning")
            ax.set_ylabel("Signal")
            name = f"Final_Phi"
            self.plot_save_or_show(fig,name)
    
    def plot_final_counts(self):
        with h5py.File(self.read_raw_dir/"array_data.h5", 'r') as f:
            final_phi = f['signal_meaning']['signal_meanings_dataset'][-1,:,:]
            fig,ax = plt.subplots(figsize = (10,8))
            sns.heatmap(final_phi,cmap = 'coolwarm')
            ax.set_title("Final Recorded Ensemble Average Counts")
            ax.set_xlabel("Meaning")
            ax.set_ylabel("Signal")
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
            
            #Partially fill the parameters of the update_array function using functools partials
            update_func = partial(self.update_array_animate,data_array = phi_array, imshow_array = im,
                                  time_mult = 1)
            
            
            anim  = animation.FuncAnimation(fig,update_func,frames = num_frames,blit =True, repeat = False)

            name = "Phi_animation"
            self.anim_save_or_show(fig,anim,name)


    def update_array_animate(self,frame, data_array,imshow_array, time_mult):
        imshow_array.set_array(data_array[frame * time_mult,:,:])
        # imshow_array.set_title(f"Phi array at frame {frame}")
        # title = imshow_array.axes.set_title(f"Phi array at frame {frame}")
        return [imshow_array]

    def animate_counts(self):
        with h5py.File(self.read_raw_dir/"array_data.h5", 'r') as f:
                    
            
            counts_array = f['signal_meaning']['signal_meanings_dataset'][:,:,:]
            
            num_frames = counts_array.shape[0]
            fig,ax = plt.subplots(figsize = (10,8))
            im = ax.imshow(counts_array[0,:,:], cmap = 'coolwarm', animated = True,vmin=counts_array.min(), vmax=counts_array.max())
            ax.set_xlabel("Meanings")
            ax.set_ylabel("Signal")
            fig.colorbar(im, ax = ax)
            ax.set_title(f"Ensemble Counts Array")
            
            #Partially fill the parameters of the update_array function using functools partials
            update_func = partial(self.update_array_animate,data_array = counts_array, imshow_array = im,
                                    time_mult = 1)
            
            
            anim  = animation.FuncAnimation(fig,update_func,frames = num_frames,blit =True, repeat = False)

            name = "Counts_animation"
            self.anim_save_or_show(fig,anim,name)



    def animate_array(self,array, frames):
        pass



if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--OutputDir', type = str, help = 'Determines the file output of saved plots, data, etc', 
                            default =str(Path.home()/"Downloads"/"MPhys/Code_Runs"))
    parser.add_argument('--ReadDir', type = str, help = 'Determines the read file location ', 
                            default =str(Path.home()/"Downloads"/"MPhys/Code_Runs"))
    parser.add_argument('--Filename', type = str, help='Determines the file to save data to', default= 'Unsorted')
    parser.add_argument('--SaveFig', action='store_true', help='If set, saves figures to args.Filename')
    parser.add_argument('--ShowFig', action='store_true', help='If set, shows figures')
    args = parser.parse_args()

    
    output_directory = Path(args.OutputDir)/args.Filename
    output_directory.mkdir(parents = True, exist_ok = True)#Creates the folder if it does not already exist
    read_directory = output_directory

    Plotter = File_Interpreter(read_directory,args.SaveFig,args.ShowFig,output_directory)
    Plotter.plot_final_phi()
    Plotter.plot_final_counts()
    Plotter.animate_phi()
    Plotter.animate_counts()