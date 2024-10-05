# Inferring explainable algorithmic approaches to identifying cancer stages, cancer grades and regions of interest automatically from medical images
Suryadipto Sarkar, A S Aravinthakshan, Teresa Wu, Alvin C. Silva


<!------------------>

# About

This is a repository that contains information on how to reproduce results corresponding to the *cutaneous T cell lymphoma (CTCL)* case study reported in [Spatial cell graph analysis reveals skin tissue organization characteristic for cutaneous T cell lymphoma](https://paper-doi-when-available).

# Summary of methods (rough documentation-to be removed)

## **Part-1**: Gradient-driven stochastic random walk identifies urologic cancer stages from CT and MRI images

### Python package:
- GradR-Walk (access [here](url-unavailable))

### Background:
- Access [here](https://drive.google.com/file/d/1EUF3mP1GRZoJq1YrjgqRfqZzonPP_h_o/view)


## **Part-2**: Automated cancer staging and grading with multimodal (imaging + non-imaging) data? Can we predict stage from aggressiveness?

### Python package:
- Stage2Grade (access [here](url-unavailable))

### Background:
- Access [here](https://drive.google.com/file/d/1cabsMYn3Nx24RTV4fxSnoucd9XwMRDBk/view) (access shorter version [here](https://drive.google.com/file/d/12noc7UbtaH9IIwRjv4OWsyFSo1K9oh_C/view) for potential students&mdash;please ignore)

## **Part-3**: Heterogeneity-based approach

- SHouT(entropy, egophily, homophily) (access [here](https://www.biorxiv.org/content/10.1101/2024.05.17.594629v1.abstract))

- Leibovici entropy, Altieri entropy

- Spatial entropy (Moran's I, Geary's C)

## **Part-4**: Classifying cancer ROIs by combining imaging-based heterogeneity in tumor microenvironments, in combination with molecular data (for example, gene expression data)

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

<!------------------>

# Abc

## Steps involved:

**Algorithm:**

Step-1: ...

Step-2: ...

Step-3: ...

Step-4: ...

Step-5: ...

Step-6: ...

Step-7: Repeat step-1.

**Once the algorithm is complete, do the following:**

i. ...

ii. ...

iii. ...

*Note:* For a detailed explanation of the Abc algorithm, please refer to the [paper](url-unavailable).

## Running the code:

Navigate  to */scripts/.../* and run **xyz.py**.

## Output:

Sample-wise cell type assignment results saved as **/scripts/.../results/xyz.csv** and **/scripts/.../results/xyz.h5ad**.

Additionally, if you want to save the results as separate .h5ad files per sample, please uncomment and run the last section of **cxyz.py** titled *Generate separate .h5ad files* (lines 77-86). This will result in separate .h5ad files saved as ...

*Note:* It must be noted that ...

<!------------------>

# Generating and saving Abc scores

## xyz

Navigate  to */scripts/.../* and run **xyz.py**.

New AnnData objects with SHouT scores saved in */results* folder as **.h5ad** files, with the same name as the original sample number.

## xyz

...

## xyz

...

<!------------------>

# Robustness testing

## Shuffled labels

### Statistical testing (Mann-Whitney U-test)

...

## Subsampled patients

### Statistical testing (Mann-Whitney U-test)

...

<!------------------>

# Scalability testing

## Runtimes with varying radii

...

<!------------------>

# Reproducing figures

## Reproducing results shown in Fig 2

...

## Reproducing results shown in Fig 3

...


## Reproducing results shown in Fig 4

...

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
