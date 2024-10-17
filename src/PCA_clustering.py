# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:17:12 2024

@author: felix
"""

import os
from sklearn.preprocessing import MultiLabelBinarizer
import umap
import hdbscan
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import plotly.express as px
import pandas as pd
from pathlib import Path
import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")
os.environ['OMP_NUM_THREADS'] = '1'

def extract_all_cardnames(expo_path):
    """
    Extracts all card names from the specified path into a DataFrame,
    where each row contains the cube name, a list of card names, 
    and the latest timestamp.

    Args:
        expo_path (Path): The path to the directory containing the cardnames files.

    Returns:
        cubename_df (DataFrame): A DataFrame containing cube names, 
                                  lists of card names, and latest timestamps.
        all_cardnames (set): A set of all unique card names.
    """
    cubename_data = []
    all_cardnames = set()

    # Loop through all *cardnames.txt files
    for filename in os.listdir(expo_path):
        if filename.startswith("cardlist_") and filename.endswith(".txt"):
            file_path = expo_path / filename 
            print(f"Processing file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # Skip the header line
                next(f)
                
                # Read card names, stripping whitespace
                cardnames = [line.strip() for line in f if line.strip()]
                
                # Extract cubename and latest timestamp
                cubename = filename.split("_")[2].replace(".txt", "")
                
                # Extract the timestamp part from the filename
                formatted_date = filename.split("_")[1]
                
                # Append a dictionary with the cubename, cards, and latestTmsp to the list
                cubename_data.append({
                    "cubename": cubename,
                    "cards": cardnames,
                    "latestTmsp": formatted_date
                })
                
                # Update all_cardnames set
                all_cardnames.update(cardnames)

    # Create a DataFrame from the collected data
    cubename_df = pd.DataFrame(cubename_data)

    return cubename_df, all_cardnames

def create_binary_matrix(cubename_df, all_cardnames):
    """
    Creates a binary matrix from the cubename DataFrame based on card presence.

    Args:
        cubename_df (DataFrame): A DataFrame containing cubenames and their associated card lists.
        all_cardnames (set): A set of all unique card names.

    Returns:
        binary_matrix (ndarray): A binary matrix indicating the presence of cards.
        cube_names (DataFrame): A DataFrame containing the cube names.
    """
    # Initialize MultiLabelBinarizer to handle binary presence of cards
    mlb = MultiLabelBinarizer(classes=list(all_cardnames))
    
    # Convert the cardname lists into a binary matrix
    binary_matrix = mlb.fit_transform(cubename_df['cards'])
    
    return binary_matrix


def find_best_silhouette_score(PCA_results):
    # Initialize a dictionary to store the best results for each method
    best_PCA_results = {
        'kmeans': None,
        'agglomerative': None,
        'hdbscan': None
    }
    # Iterate over clustering methods with fixed number of clusters
    for method in ['kmeans', 'agglomerative']:
        for result in PCA_results[method]:
            n_neighbors, umap_embedding, min_dist, cluster_labels, silhouette_score_val, n_clusters = result
            
            # Check if this is the best result for the current method
            if best_PCA_results[method] is None or silhouette_score_val > best_PCA_results[method][3]:
                best_PCA_results[method] = (n_neighbors, min_dist, method, silhouette_score_val, n_clusters)

    # For HDBSCAN, no fixed number of clusters, handle separately
    for result in PCA_results['hdbscan']:
        n_neighbors, umap_embedding, min_dist, cluster_labels, silhouette_score_val, n_clusters = result

        # Check if this is the best result for HDBSCAN
        if best_PCA_results['hdbscan'] is None or silhouette_score_val > best_PCA_results['hdbscan'][3]:
            best_PCA_results['hdbscan'] = (n_neighbors, min_dist, 'hdbscan', silhouette_score_val, n_clusters)

    return best_PCA_results


def umap_clustering_interactive(binary_matrix, cubename_df, all_cardnames, PCA_results, n_neighbors=5, min_dist=0.1, n_clusters = 3, clustering=True, clustering_method='hdbscan', plot = True, export_path = None):
    """
    This function runs UMAP on a binary matrix of card names, optionally applies clustering, 
    and generates an interactive Plotly plot with the results. The plot can be saved in both 
    HTML and JPG formats.

    Args:
        binary_matrix (ndarray): A binary matrix indicating the presence of cards in each cube.
        cubename_df (DataFrame): A DataFrame containing cube names and associated card lists.
        all_cardnames (set): A set of all unique card names.
        best_result (dict): A dictionary to store the best results for different clustering methods.
        n_neighbors (int, optional): The number of nearest neighbors to consider for UMAP. Defaults to 5.
        min_dist (float, optional): The minimum distance parameter for UMAP. Defaults to 0.1.
        n_clusters (int, optional): The number of clusters for 'kmeans' and 'agglomerative' methods. Defaults to 3.
        clustering (bool, optional): Whether to apply clustering after UMAP projection. Defaults to True.
        clustering_method (str, optional): The clustering method to use ('hdbscan', 'kmeans', 'agglomerative'). Defaults to 'hdbscan'.
        plot (bool, optional): Whether to generate the Plotly plot. Defaults to True.
        export_path (Path, optional): Directory to export the generated plot. Defaults to None.

    Returns:
        best_result (dict): Updated best_result dictionary with clustering outcomes and silhouette scores.

    Saves:
        - HTML file of the plot at export_path if provided.
    """    
    # Run UMAP
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    umap_embedding = umap_reducer.fit_transform(binary_matrix)
    if export_path is None:
        export_path = Path("..//Output/All Results")  
    # Add clustering if needed
    if clustering:
        if clustering_method == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, prediction_data=True)
            cluster_labels = clusterer.fit_predict(umap_embedding)
            if len(set(cluster_labels)) > 1:  # Only compute silhouette if we have more than 1 cluster
                silhouette = silhouette_score(umap_embedding, cluster_labels)
                PCA_results['hdbscan'].append((n_neighbors, umap_embedding, min_dist, cluster_labels,silhouette ,n_clusters))
        
        elif clustering_method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(umap_embedding)
            silhouette = silhouette_score(umap_embedding, cluster_labels)
            PCA_results['kmeans'].append((n_neighbors, umap_embedding, min_dist, cluster_labels, silhouette,n_clusters))
            
        elif clustering_method == 'agglomerative':
            agg_clust = AgglomerativeClustering(n_clusters=n_clusters,)
            cluster_labels = agg_clust.fit_predict(umap_embedding)
            silhouette = silhouette_score(umap_embedding, cluster_labels)
            PCA_results['agglomerative'].append((n_neighbors, umap_embedding, min_dist, cluster_labels, silhouette, n_clusters))

        else:
            raise ValueError("Invalid clustering method specified.")
    else:
        cluster_labels = np.zeros(umap_embedding.shape[0])  # No clustering
        PCA_results['none'].append((n_neighbors, umap_embedding, min_dist))

    if plot:
        df = pd.DataFrame({
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'Cube Name': cubename_df["cubename"],
        'Cluster': cluster_labels.astype(str),
        'Changedate': cubename_df["latestTmsp"]
        
        })
    
        fig = px.scatter(
            df, 
            x='UMAP1', 
            y='UMAP2', 
            color='Cluster', 
            hover_name=cubename_df['cubename'],
            hover_data={'Cluster': True, 'Changedate': True, 'UMAP1': False, 'UMAP2': False},
            title=f'UMAP Projection with {n_neighbors} nearest neighbors, a minimal distance of {min_dist} and {clustering_method}-clustering.'
        )
        formatted_silhouette = f'{silhouette:.2f}'

        # Add subtitle using fig.update_layout()
        fig.update_layout(
            title={
                'text': f'UMAP Projection with {n_neighbors} nearest neighbors, a minimal distance of {min_dist} and {clustering_method}-clustering.<br><sup>Silhouette score = {formatted_silhouette}</sup>',
                'y': 0.93,  # Adjust vertical position
                'x': 0.5,   # Center the title
                'xanchor': 'center',
                'yanchor': 'bottom'
            }
        )
    
         # Add annotations with cube names initially hidden
        annotations = []
        for i, row in df.iterrows():
            annotations.append(
                dict(
                    x=row['UMAP1'], 
                    y=row['UMAP2'] + 0.05, 
                    xref='x',  # Reference to x-axis in plot
                    yref='y',  # Reference to y-axis in plot
                    text=row['Cube Name'],
                    showarrow=False,
                    font=dict(size=10),
                    opacity=0  # Initially invisible
                )
            )
    
        # Add the annotations to the figure
        fig.update_layout(annotations=annotations)
    
        # Add dropdown button to toggle cube name visibility
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label="Show Cube Names",
                            method="relayout",
                            args=[{"annotations": [dict(annotation, opacity=1) for annotation in annotations]}]
                        ),
                        dict(
                            label="Hide Cube Names",
                            method="relayout",
                            args=[{"annotations": [dict(annotation, opacity=0) for annotation in annotations]}]
                        )
                    ],
                    direction="up",
                    showactive=True,
                    x=0.5,
                    xanchor="center",
                    y=-0.25,
                    yanchor="bottom"
                )
            ]
        )
    
        ## cdn if the .html files are too big. cdn reaquires internet connection and
        ## redownloads the important plotly files from Content Delivery Network
        # # Output html that you can copy paste
        # fig.to_html(full_html=False, include_plotlyjs='cdn')
        # # Saves a html doc that you can copy paste
        # fig.write_html("output.html", full_html=False, include_plotlyjs='cdn')
                       
        # Save the plot as an HTML file
        filename = f"umap_{n_neighbors}n_neighbours_mindist{min_dist}_{clustering_method}"
        if clustering_method == "agglomerative" or clustering_method == "kmeans":
            filename = filename + f"_{n_clusters}clusters"
        filename = filename + ".html"
        
        if export_path:
           export_path = Path(export_path)  # Ensure it is a Path object
           export_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
           filename = export_path / filename

        fig.write_html(filename)
        fig.show()
 
    return PCA_results


def find_best_PCA_parameters(input_path = None, clustering_methods = None, 
                             min_dists = None, n_neighbors = None, 
                             n_clusters= None, clustering = True,plot = False, export_path = None, verbose = 0):
    
    if input_path == None:
        input_path =  Path("..//Input/Cubedata")  
        # Path where the *_cardnames.txt files are stored
        
    if clustering_methods == None:
        clustering_method = ["hdbscan","kmeans","agglomerative"]
        
    if min_dists == None:
        min_dists = [0.02,0.025,0.03,0.04,0.045,0.05,0.06]
        
    if n_neighbors == None:
        n_neighbors = [3,4,5,6]
        
    if n_clusters == None:
       n_cluster = [2, 3, 4, 5]
       
    if export_path is None:
        export_path =  Path("..//Output/All Results")  

    
    PCA_results = {'hdbscan': [], 'kmeans': [], 'agglomerative': [], 'none': []}
    # Extract all cardnames and create a binary matrix
    cubename_df, all_cardnames = extract_all_cardnames(input_path)
    binary_matrix = create_binary_matrix(cubename_df, all_cardnames)
    
    
    ## Find the best parameters       
    for clustering_method in clustering_methods:                 
        for min_dist in min_dists:
            for n_neighbor in n_neighbors:
               # If clustering method is 'kmeans' or 'agglomerative', loop through n_clusters
                if clustering_method in ["kmeans", "agglomerative"]:
                    for n_cluster in n_clusters:
                        PCA_results = umap_clustering_interactive(
                                binary_matrix, cubename_df, all_cardnames, PCA_results,
                                n_neighbors=n_neighbor, min_dist=min_dist, n_clusters=n_cluster,
                                clustering=clustering, clustering_method=clustering_method, plot=plot, export_path = export_path)
                        if verbose >0:
                            print(f"{clustering_method} with {n_cluster} clusters, {n_neighbor} n_neighbors in a minimal distance of {min_dist}")
        
                else:
                    # For 'hdbscan', simply call the function without n_clusters
                    PCA_results = umap_clustering_interactive(
                            binary_matrix, cubename_df, all_cardnames, PCA_results,
                            n_neighbors=n_neighbor, min_dist=min_dist,
                            n_clusters=None, clustering=clustering,
                            clustering_method=clustering_method, plot=plot)
                if verbose >0:
                    print(f"{clustering_method} with {n_neighbor} n_neighbors in a minimal distance of {min_dist}")
    
    best_PCA_results = find_best_silhouette_score(PCA_results)
    
    return [PCA_results, best_PCA_results,binary_matrix, cubename_df, all_cardnames]

def print_best_PCA_results(PCA_results = None, best_PCA_results = None, binary_matrix=None, cubename_df = None, all_cardnames = None, export_path = None ):
    
    if PCA_results is None:
        PCA_results =  {'hdbscan': [], 'kmeans': [], 'agglomerative': [], 'none': []}
        
    if binary_matrix is None:
        cubename_df, all_cardnames = extract_all_cardnames(input_path)
        binary_matrix = create_binary_matrix(cubename_df, all_cardnames)
    
    if best_PCA_results is None:
        print("Error, no PCA results given printing default results")
        # Given tuples for best results from previous scan
        agglomerative_result = (5, 0.05, 'agglomerative', 0.87544876, 3)
        hdbscan_result = (3, 0.02, 'hdbscan', 0.92644346, 3)
        kmeans_result = (5, 0.05, 'kmeans', 0.87544876, 3)
        
        # Creating the dictionary
        best_PCA_results = {
            'agglomerative': agglomerative_result,
            'hdbscan': hdbscan_result,
            'kmeans': kmeans_result
        }
   
    if export_path is None:
        export_path = Path("..//Output/Best Results")
    
    if best_PCA_results:
        for method in  ["hdbscan","kmeans","agglomerative"]:
            
            # Unpack the best result
            n_neighbors, min_dist, clustering_method, silhouette_score_val, n_clusters = best_PCA_results[method]
            print( n_neighbors, min_dist, clustering_method, silhouette_score_val, n_clusters)
            print(f"\nBest result found: n_neighbors={n_neighbors},  min_dist={min_dist}, method={clustering_method}, silhouette_score={silhouette_score_val}")
            # umap_clustering_interactive(binary_matrix, cubename_df, all_cardnames, best_results, n_neighbors=5, min_dist=0.1, n_clusters = 3, clustering=True, clustering_method='hdbscan', plot = True, export_path = None):
    
            # Call the interactive UMAP clustering with the best parameters
            umap_clustering_interactive(binary_matrix, cubename_df, 
                                        all_cardnames, PCA_results, 
                                        n_neighbors=n_neighbors, 
                                        min_dist=min_dist, clustering=True, 
                                        clustering_method=clustering_method,
                                        plot = True, export_path=export_path)
    else:
        print("No valid clustering results were found.")
        
    
    
if __name__ == "__main__":

    input_path = Path("..//Input/Cubedata")  # Path where the *_cardnames.txt files are stored
    PCA_results = {'hdbscan': [], 'kmeans': [], 'agglomerative': [], 'none': []}

    # Extract all cardnames and create a binary matrix
    cubename_df, all_cardnames = extract_all_cardnames(input_path)
    binary_matrix = create_binary_matrix(cubename_df, all_cardnames)
    

    # # Example usage
    n_neighbors = 4
    min_dist = 0.05
    clustering_method = "hdbscan"
    n_clusters = 3 # for "kmeans" and "agglomerative"
    clustering=True
    plot = True
    export_path = Path("..//Output/All Results")
    PCA_results = umap_clustering_interactive(
        binary_matrix, cubename_df, all_cardnames, PCA_results, 
        n_neighbors=n_neighbors, min_dist=min_dist, n_clusters = n_clusters, clustering=clustering, 
        clustering_method=clustering_method, plot = plot, export_path=export_path)

    plot = False

    ### Example usage 
    input_path =  Path("..//Input/Cubedata") 

    [PCA_results, best_PCA_results,binary_matrix, 
     cubename_df, all_cardnames] = find_best_PCA_parameters(input_path,                               
                    clustering_methods = ["hdbscan","kmeans","agglomerative"], 
                           min_dists = [0.035,0.4,0.5],
                           n_neighbors = [2,4,6], 
                           n_clusters= [3,4,5,6],
                           plot = True,
                           verbose = 1)

    print_best_PCA_results(PCA_results, best_PCA_results, 
                           binary_matrix,
                           cubename_df,
                           all_cardnames )

        
   