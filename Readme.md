# MtG PCA Cubemap

This program demonstrates the power of dimensional reduction / Principal component analysis [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)  by clustering different sets of 360 MtG-cards (later referred as "cubes") on a 2-dimensional chart using unifold-manually
Uniform Manifold Approximation and Projection for Dimension Reduction [UMAP](https://umap-learn.readthedocs.io/en/latest/). Every card present in the cubes corresponds to another dimension (present or not present). 
Clustering is performed and a output file is generated. 

A sample file can be seen here:

![fileinfo](./Readme_JPGs/SampleCubemap.jpg)
The interactive version of this file can be accessed ![here](./Readme_JPGs/umap_3n_neighbours_mindist0.02_hdbscan.html")

<span style="color:red">Add link to github repo</span>.

1. **HDB Scan** [wiki](https://de.wikipedia.org/wiki/DBSCAN#cite_note-5) and see the [github-repo](https://github.com/scikit-learn-contrib/hdbscan)
2. 
3. , 
will produce a cubemap for [MtG-cards](https://magic.wizards.com/en) using 
princicple component analysis 
like described in the article from [luckypaper](https://luckypaper.co/articles/mapping-the-magic-landscape/) starting from .csv files obtained from your cube from [cubecobra](cubecobra.com).

The clustering can be evaluated using the [Silhouette score](https://en.wikipedia.org/wiki/Silhouette_(clustering)).





## Setup

### Setup the cubes you want to compare

First, you need to download the cube contents as a .csv to the direction
Input csv.

* This is doen by writing the links to the cubes into the ".\PCA CubeMap\Inputs\fetch_cubedata.txt" file.
[Input / fetch_cubedata.txt](Input/fetch_cubedata.txt)


* Or do it manually with the export list command from cubecobra "C:\Users\felix\Documents\PCA CubeMap\Input_CSV"
 Therefore, click on the cube's you want to compare and export the list as *Comma-separated (.csv)*
![Download .csv cubefiles via export as list](./Readme_JPGs/Export_csv.jpg)

Then start the main.py function in src / main.py