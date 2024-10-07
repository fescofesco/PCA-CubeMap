# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:38:20 2024

@author: felix

Structure of the project

PCA CubeMap
│
├── src\
│   └── scryfallmanager.py
│
└── Input\
│   └── Cubedata\
│       ├── fetch_cubedata.txt
│       └── Cubedata\
│           ├── cubename.csv
│           └── cubename_cardnames.txt        
│
└── Output\

"""

from pathlib import Path
from import_cube_from_url import fetch_cube_data, read_urls
from PCA_clustering import find_best_PCA_parameters, print_best_PCA_results

        
# Define the relative path to the input file
input_file_path = Path("../Input/fetch_cubedata.txt")
urls = read_urls(input_file_path)
export_path = Path("..//Input/Cubedata")
fetch_cube_data(urls,export_path)
input_path =  Path("..//Input/Cubedata")  
export_path = Path("..//Output/Best Results")  
[PCA_results, best_PCA_results,
 binary_matrix, cubename_df, all_cardnames] = find_best_PCA_parameters(
     input_path,                            
    clustering_methods = ["hdbscan","kmeans","agglomerative"], 
    min_dists = [0.03, 0.035,0.4,0.5], 
    n_neighbors = [2,4,6], 
    n_clusters= [3,4,5,6],
    clustering = True,
    plot = True,
      verbose = 1)


print_best_PCA_results(PCA_results, best_PCA_results, 
                       binary_matrix,
                       cubename_df,
                       all_cardnames,
                       export_path)
    

print("Warum wird falsche PCA best Results geprinted")   
def main():
    # Define the relative path to the input file
    input_file_path = Path("../Input/fetch_cubedata.txt")
    urls = read_urls(input_file_path)
    export_path = Path("..//Input/Cubedata")
    fetch_cube_data(urls,export_path)
    input_path =  Path("..//Input/Cubedata")  
    export_path = Path("..//Output/Best Results")  
    [PCA_results, best_PCA_results,
     binary_matrix, cubename_df, all_cardnames] = find_best_PCA_parameters(
         input_path,                            
        clustering_methods = ["hdbscan","kmeans","agglomerative"], 
        min_dists = [0.03, 0.035,0.4,0.5], 
        n_neighbors = [2,4,6], 
        n_clusters= [3,4,5,6],
        clustering = True,
        plot = True,
          verbose = 1)


    print_best_PCA_results(PCA_results, best_PCA_results, 
                           binary_matrix,
                           cubename_df,
                           all_cardnames,
                           export_path)
    

if __name__ == "__main__1":
    main()
