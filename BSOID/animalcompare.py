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

def main(csv_folder, cluster_range, fps, UMAP_PARAMS, HDBSCAN_PARAMS):
    plots = []
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    n_files = len(csv_files)

    for idx, file in enumerate(csv_files):
        
        pose_chosen = []

        file_j_df = pd.read_csv(os.path.join(csv_folder,file), low_memory=False)
       
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

        plot = plot_classes(sampled_embeddings, assignments, file)
        
        plots.append(plot)

    return plots
if __name__ == "__main__":
    main()