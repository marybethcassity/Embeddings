import itertools
import numpy as np
import pandas as pd
import math 
from tqdm import tqdm
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from psutil import virtual_memory
import umap
import hdbscan

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

import streamlit as st

from bsoid_utils import * 

import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html

def main(csv_folders, cluster_range, fps, UMAP_PARAMS, HDBSCAN_PARAMS, method):
    plots = []
    subfolders = [f.path for f in os.scandir(csv_folders) if f.is_dir()]
    csv_files1 = [f for f in os.listdir(subfolders[0]) if f.endswith('.csv')]
    csv_files2 = [f for f in os.listdir(subfolders[1]) if f.endswith('.csv')]
    n_files1 = len(csv_files1)
    n_files2 = len(csv_files2)

    #fig, axs = plt.subplots(n_files1 + n_files2, 2, figsize=(16, 9 * (n_files1 + n_files2, projection='3d'))) 

    for idx, file in enumerate(csv_files1):
        
        pose_chosen = []

        file_j_df = pd.read_csv(os.path.join(subfolders[0],file), low_memory=False)
       
        file_j_df_array = np.array(file_j_df)
        p = st.multiselect('Identified __pose__ to include:', [*file_j_df_array[0, 1:-1:3]], [*file_j_df_array[0, 1:-1:3]])
        for a in p:
            index = [i for i, s in enumerate(file_j_df_array[0, 1:]) if a in s]
            if not index in pose_chosen:
                pose_chosen += index
        pose_chosen.sort()

        file_j_processed, p_sub_threshold = adp_filt(file_j_df, pose_chosen)
        file_j_processed = file_j_processed.reshape((1, file_j_processed.shape[0], file_j_processed.shape[1]))

        scaled_features, features = compute(file_j_processed, fps)

        train_size = subsample(file_j_processed, fps)

        sampled_embeddings = learn_embeddings(scaled_features, features, UMAP_PARAMS, train_size)

        assignments = hierarchy(cluster_range, sampled_embeddings, HDBSCAN_PARAMS)
        
        if method == "plotly":
            plot = create_plotly(sampled_embeddings, assignments, file)
        if method == "matplotlib":
            plot = plot_classes(sampled_embeddings, assignments, file)
        
        # axs[idx, 0].set_title(f'{file}')
        # axs[idx, 0].set_axis_off()
        # axs[idx, 0].plot(plot, projection='3d')

    for idx, file in enumerate(csv_files2):
        pose_chosen = []

        file_j_df = pd.read_csv(os.path.join(subfolders[1],file), low_memory=False)
       
        file_j_df_array = np.array(file_j_df)
        p = st.multiselect('Identified __pose__ to include:', [*file_j_df_array[0, 1:-1:3]], [*file_j_df_array[0, 1:-1:3]])
        for a in p:
            index = [i for i, s in enumerate(file_j_df_array[0, 1:]) if a in s]
            if not index in pose_chosen:
                pose_chosen += index
        pose_chosen.sort()

        file_j_processed, p_sub_threshold = adp_filt(file_j_df, pose_chosen)
        file_j_processed = file_j_processed.reshape((1, file_j_processed.shape[0], file_j_processed.shape[1]))

        scaled_features, features = compute(file_j_processed, fps)

        train_size = subsample(file_j_processed, fps)

        sampled_embeddings = learn_embeddings(scaled_features, features, UMAP_PARAMS, train_size)

        assignments = hierarchy(cluster_range, sampled_embeddings, HDBSCAN_PARAMS)

        if method == "plotly":
            plot = create_plotly(sampled_embeddings, assignments, file)
        if method == "matplotlib":
            plot = plot_classes(sampled_embeddings, assignments, file)
        
        # axs[idx, 1].set_title(f'{file}')
        # axs[idx, 1].set_axis_off()
        # axs[idx, 1].plot(plot, projection='3d')

    #plt.tight_layout()
    #plt.show()

if __name__ == "__main__":
    main()