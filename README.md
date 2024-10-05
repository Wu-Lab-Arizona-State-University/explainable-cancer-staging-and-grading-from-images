# Inferring explainable algorithmic approaches to identifying cancer stages, cancer grades and regions of interest automatically from medical images
Suryadipto Sarkar, A S Aravinthakshan, Teresa Wu, Alvin C. Silva




# About

This is a repository that contains information on how to reproduce results corresponding to the *cutaneous T cell lymphoma (CTCL)* case study reported in [Spatial cell graph analysis reveals skin tissue organization characteristic for cutaneous T cell lymphoma](https://paper-doi-when-available).

# Summary of methods (rough documentation-to be removed)

## **Part-1**: Gradient-driven stochastic random walk identifies urologic cancer stages from CT and MRI images (access [here](https://drive.google.com/file/d/1EUF3mP1GRZoJq1YrjgqRfqZzonPP_h_o/view))


## **Part-2**: Automated cancer staging and grading with multimodal (imaging + non-imaging) data? Can we predict stage from aggressiveness? (access [here](https://drive.google.com/file/d/1cabsMYn3Nx24RTV4fxSnoucd9XwMRDBk/view)) (access shorter version [here](https://drive.google.com/file/d/12noc7UbtaH9IIwRjv4OWsyFSo1K9oh_C/view) for potential students--please ignore)


## **Part-3**: Heterogeneity-based approach

- SHouT(entropy, egophily, homophily) (access [here](https://www.biorxiv.org/content/10.1101/2024.05.17.594629v1.abstract))

- Leibovici entropy, Altieri entropy

- Spatial entropy (Moran's I, Geary's C)

## **Part-4**: Classifying cancer ROIs by combining imaging-based heterogeneity in tumor microenvironments, in combination with molecular data (for example, gene expression data)

- Can this help explain underlying molecular mechanisms (eg. gene and/ or protein expressions), and how they manifest in scans?

- Very loosely related to this work (ECCB2024 poster presentation):

![ECCB2024-heterogeneity-poster](/ECCB2024-heterogeneity-poster.jpeg)

<!------------------>

<!------------------>

# Data

## Overview

As described in our [preprint](https://doi.org/10.1101/2024.05.17.594629), the data used for our analyses comprised a total of 69 skin tissue samples (21 CTCL, 23 AD, 25 PSO), obtained from 27 treated patients (8 CTCL, 7 AD, 12 PSO). Each sample contained at least 35 images protein channels, each of resolution 2018 X 2018.

## Availability

Complete data available publicly over [Zenodo](https://doi.org/10.5281/zenodo.11125482). For the purpose of conserving memory, each of the images have been down-scaled to 512 X 512 pixels from their original resolution of 2018 X 2018 pixels.

## Description

The [repository](https://doi.org/10.5281/zenodo.11125482) follows the following data structure:

1. File **data\_description.xlsx**: Provides information pertaining to *PatientID*, *SampleNumber* and *Condition*; this file has been color coded per condition for improved readability.

2. File **additional\_metadata.xlsx**: Provides additional information pertaining to the samples specified in **data\_description.xlsx**.

3. Directories **/AD**, **/PSO** and **/CTCL**: Folders containing the respective samples specified in **data\_description.xlsx** and **additional\_metadata.xlsx**. The sub directory organizations are self explanatory---they are of the form: **/Condition/PatientID\_<patientID>/SampleNumber\_<sampleNumber>/**. Each sample contains at least 35 .tif images, each corresponding to a (protein channel, dye) combination.

<!------------------>

# Installation

Install conda environment as follows (there also exists a requirements.txt)
```bash
conda create --name ctcl_case_study
conda activate ctcl_case_study
pip install scipy==1.10.1 numpy==1.23.5 squidpy==1.3.0 pandas==1.5.3 scikit-learn==1.2.2
```
*Note:* Additionally, modules *math* and *statistics* were used, however no installation is required as they are provided with Python by default.

<!------------------>

# HPA-based cell type assignment

## Steps involved:

**Algorithm:**

Step-1: Fit GMM model.

Step-2: Find list of good split genes.

Step-3: Calculate spread per celltype per gene in set {good\_split\_genes}.

Step-4: Arrange all genes in set {good\_split\_genes} in descending order, per celltype.

Step-5: Pick {gene-g, celltype-C} that maximizes spread.

Step-6: Assign g+: C.

Step-7: Repeat step-1.

**Once the algorithm is complete, do the following:**

i. Assign unassigned cells to 'Unknown' type.

ii. Map back assigned celltypes to the original cells.

iii. Save sample-wise cell type assignment results to a csv file and as an anndata.h5ad file.

*Note:* For a detailed explanation of the cell type assignment algorithm, please refer to the [paper](https://paper-doi-when-available).

## Running the code:

Navigate  to */scripts/hpa\_based\_cell\_type\_assignment/* and run **cell\_type\_assignment.py**.

## Output:

Sample-wise cell type assignment results saved as **/scripts/hpa\_based\_cell\_type\_assignment/result/sample-wise celltypes (HPA-based clustering).csv** and **/scripts/hpa\_based\_cell\_type\_assignment/result/celltype\_assigned\_anndata.h5ad**.

Additionally, if you want to save the results as separate .h5ad files per sample, please uncomment and run the last section of **cell\_type\_assignment.py** titled *Generate separate .h5ad files* (lines 77-86). This will result in separate .h5ad files saved as **/scripts/hpa\_based\_cell\_type\_assignment/result/<sample_id>.h5ad**. However, since these files have already been provided under the **/data** directory, upon which all of our subsequent SHouT heterogeneity score-based analyses have been performed, this code has been commented out by default in order to avoid generation of redundant .h5ad files.

*Note:* It must be noted that two of the samples, namely *291* and *294*, have been removed from the **/data** directory: that is simply because samples *291* and *294* contain too few segmented cells to run the SHouT heterogeneity scores on.

<!------------------>

# Generating and saving SHouT heterogeneity scores

## Computing the actual SHouT scores

Navigate  to */scripts/shout\_score\_generation/* and run **compute\_shout\_scores.py**.

New AnnData objects with SHouT scores saved in */results* folder as **.h5ad** files, with the same name as the original sample number.

## Saving SHouT scores as .csv files

Navigate  to */scripts/shout\_score\_generation/* and run **results\_to\_csv.py**.

Results are saved in */results/sample\_results.csv* for global (sample-wise) heterogeneity scores, and */results/cell\_results.csv* for local (individual cell-wise) heterogeneity scores.

## Statistical testing (Mann-Whitney U-test)

Navigate  to */scripts/shout\_score\_generation/* and run **statistical\_tests.py**.

Results are saved in */results/p\_values\_global.csv* for local (individual cell-wise) and global (sample-wise) heterogeneity scores with **r={1,2,3,4,5}** between all pairs of conditions; and */results/p\_values\_cell\_type.csv* for local (individual cell-wise) heterogeneity scores with **r={1,2,3,4,5}** and per celltype, between all pairs of conditions.

<!------------------>

# Robustness testing

## Shuffled labels

### Statistical testing (Mann-Whitney U-test)

Navigate  to */scripts/shout\_score\_generation/* and run **save\_pvalue\_statistics\_shuffled\_labels.py**.

Results are saved in */results/pvalue\_statistics\_shuffled\_labels.csv* for local (individual cell-wise) heterogeneity scores with **r={1,2,3,4,5}** between all pairs of conditions after having randomized the condition labels.


## Subsampled patients

### Statistical testing (Mann-Whitney U-test)

Navigate  to */scripts/shout\_score\_generation/* and run **statistical_tests_subsampled_patientwise.py**.

Results for local (individual cell-wise) heterogeneity scores are saved in */results/p_values_cell_type_subsampled_patientwise.csv* with **r={1,2,3,4,5}** between all pairs of conditions after having randomly subsampled 15 patients per condition at a time, while maintaining the actual condition labels, and repeating this subsampling process for 100 iterations.

Results for global (sample-wise) heterogeneity scores are saved in */results/p_values_global_subsampled_patientwise.csv* with **r={1,2,3,4,5}** between all pairs of conditions after having randomly subsampled 15 patients per condition at a time, while maintaining the actual condition labels, and repeating this subsampling process for 100 iterations.

<!------------------>

# Scalability testing

## Runtimes with varying radii

In order to record the runtimes of executing SHouT heterogeneity scores on all 69 samples, with radii r = {1, 5, 10, 20, 40, 80, 100}, navigate to */scripts/plots\_for\_paper/fig6\_scalability\_plot/* and run **per\_sample\_runtimes\_with\_varying\_radius.py**. The runtimes per sample, for each of r = {1, 5, 10, 20, 40, 80, 100}, is saved as */results/data\_all\_cells.csv*

<!------------------>

# Reproducing figures

## Reproducing results shown in Fig 2

Navigate  to */scripts/plots\_for\_paper/fig2\_database/* and run **fig2\_database.py**. Plot saved as *fig2\_database.pdf*.

## Reproducing results shown in Fig 3

Navigate  to */scripts/plots\_for\_paper/fig3\_subplot\_mosaic/* and run **fig3\_subplot\_mosaic.py**. Plot saved as *fig3\_subplot\_mosaic.pdf*.


## Reproducing results shown in Fig 4

Navigate  to */scripts/plots\_for\_paper/fig4\_shuffled\_labels/* and run **fig4\_shuffled\_labels.py**. Plot saved as *fig4\_shuffled\_labels.pdf*.

## Reproducing results shown in Fig 5

Navigate  to */scripts/plots\_for\_paper/fig5\_subsampled\_patients/* and run **fig5\_subsampled\_patients.py**. Plot saved as *fig5\_subsampled\_patients.pdf*.

## Reproducing results shown in Fig 6

Navigate  to */scripts/plots\_for\_paper/fig6\_scalability\_plot/* and run **fig6\_scalability\_plot.py**. Plot saved as *fig6\_scalability\_plot.pdf*.


## Reproducing results shown in Supplementary Fig 2

Navigate  to */scripts/plots\_for\_paper/supfig2\_cell\_abundance\_analyses/* and run **supfig2\_cell\_abundance\_analyses.py**. Plot saved as *supfig2\_cell\_abundance\_analyses.pdf*.

## Reproducing results shown in Supplementary Fig 3 - 5

Navigate  to */scripts/plots\_for\_paper/supfigs3,4,5\_SHouT\_scores\_radius\_5/* and run **generate\_supfig\_1\_2\_and\_3\_violinplots.py**. Plots saved as *SHouT\_score\_1\_violinplots.pdf*, *SHouT\_score\_2\_violinplots.pdf* and *SHouT\_score\_3\_violinplots.pdf* respectively.


## Reproducing results shown in Supplementary Fig 6

Navigate  to */scripts/plots\_for\_paper/supfig6\_SHouT\_scores\_all\_radii/* and run **supfig6\_SHouT\_scores\_all\_radii.py**. Plot saved as *supfig6\_SHouT\_scores\_all\_radii.pdf*.


## Reproducing results shown in Supplementary Fig 7

Navigate  to */scripts/plots\_for\_paper/supfig7\_centrality\_scores/* and run **generate\_supfig7\_centrality\_scores.py**. Plot saved as *supfig7\_centrality\_scores.pdf*.

<!------------------>

# Citing the work

Please cite our work as follows:
- Sarkar, S., Möller, A., Hartebrodt, A., Erdmann, M., Ostalecki, C., Baur, A. & Blumenthal, D. B. Spatial cell graph analysis reveals skin tissue organization characteristic for cutaneous T cell lymphoma. bioRxiv 2024.05.17.594629 (2024). doi:10.1101/2024.05.17.594629
