# Inferring explainable algorithmic approaches to identifying cancer stages, cancer grades and regions of interest automatically from medical images
Suryadipto Sarkar, A S Aravinthakshan, Teresa Wu, Alvin C. Silva


<!------------------>

# About

This is a repository that contains information on how to reproduce results corresponding to the *cutaneous T cell lymphoma (CTCL)* case study reported in [Spatial cell graph analysis reveals skin tissue organization characteristic for cutaneous T cell lymphoma](https://paper-doi-when-available).

# Summary of methods


## **Part-1**: Gradient-driven stochastic random walk identifies urologic cancer stages from CT and MRI images

### Python packages:
- Network-GradR-Walk (access [here](url-unavailable))
- Spatial-GradR-Walk (access [here](url-unavailable))

### Background:
- Access [here](https://drive.google.com/file/d/1EUF3mP1GRZoJq1YrjgqRfqZzonPP_h_o/view)


## **Part-2**: Heterogeneity-based approach

### Python packages:

- Network-Heterogeneity

- Spatial-Heterogeneity

### Background:

- SHouT(entropy, egophily, homophily) (access [here](https://www.biorxiv.org/content/10.1101/2024.05.17.594629v1.abstract))

- Leibovici entropy, Altieri entropy

- Spatial entropy (Moran's I, Geary's C)


## **Part-3**: Classifying cancer ROIs by combining imaging-based heterogeneity in tumor microenvironments, in combination with molecular data (for example, gene expression data)

- Can this help explain underlying molecular mechanisms (eg. gene and/ or protein expressions), and how they manifest in scans?

- Very loosely related to this work (ECCB2024 poster presentation):

![ECCB2024-heterogeneity-poster](/ECCB2024-heterogeneity-poster.jpeg)

<!------------------>

# Data

## Overview

---

## Description

---

## Availability

Data will be made available under reasonable request to the corresponding author, [Suryadipto Sarkar](suryadipto.sarkar@fau.de) (more contact details below).

<!------------------>

# Installation

Install conda environment as follows (there also exists a requirements.txt)
```bash
conda create --name imaging_heterogeneity_study
conda activate imaging_heterogeneity_study
pip install scipy==1.10.1 numpy==1.23.5 squidpy==1.3.0 pandas==1.5.3 scikit-learn==1.2.2
```
*Note:* Additionally, modules *math* and *statistics* were used, however no installation is required as they are provided with Python by default.


# Robustness testing

Pending

<!------------------>

# Scalability testing

Pending

<!------------------>

# Reproducing figures

Pending

<!------------------>

# Citing the work

## MLA

Will be made available upon publication.

## APA

Will be made available upon publication.

## BibTex

Will be made available upon publication.

<!------------------>

# Contact

&#x2709;&nbsp;&nbsp;suryadipto.sarkar@fau.de<br/>
&#x2709;&nbsp;&nbsp;ssarka34@asu.edu<br/>
&#x2709;&nbsp;&nbsp;ssarkarmanipal@gmail.com

<!------------------>

# Impressum

Suryadipto Sarkar ("Surya"), MS<br/><br/>
PhD Candidate<br/>
Biomedical Network Science Lab<br/>
Department of Artificial Intelligence in Biomedical Engineering (AIBE)<br/>
Friedrich-Alexander University Erlangen-Nürnberg (FAU)<br/>
Werner von Siemens Strasse<br/>
91052 Erlangen<br/><br/>
MS in CEN from Arizona State University, AZ, USA.<br/>
B.Tech in ECE from MIT Manipal, KA, India.
