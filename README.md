# PathIntegrate_scripts

This repository contains benchmarking and application scripts for the PathIntegrate manuscript. Any analyses using the COVID-19 data by Su et al. 2020 ([DOI](https://doi.org/10.1016/j.cell.2020.10.037)) can be replicated using code and data within this repository. Any analyses requiring the COPDgene multi-omics data require access to the COPDgene consortium data* (see data availability section in the manuscript).

**This is not the repository for the PathIntegrate package. The PathIntegrate package can be found at [PathIntegrate](https://github.com/cwieder/PathIntegrate)**

>*The COPDgene multi-omics data can be found at the following sources: Clinical Data and SOMAScan data are available through COPDGene (https://www.ncbi.nlm.nih.gov/gap/, ID: phs000179.v6.p2). RNA-Seq data is available through dbGaP (https://www.ncbi.nlm.nih.gov/gap/, ID: phs000765.v3.p2). Metabolon data is available at Metabolomics Workbench (https://www.metabolomicsworkbench.org/ ID: PR000907). 

## Prerequisites

- All scripts are run using Python 10
- Dependencies are listed in the `requirements.txt` file
- Benchmarking scripts are developed for running on a PBS HPC cluster
- Application scripts are run within Jupyter notebooks

## Installation

- Clone this respoitory to access the scripts and data

## Data and pathways

- COVID_data: contains the COVID-19 metabolomics and proteomics data from Su et al. 2020
- Pathway_databases: contains the KEGG and Reactome pathway databases for metabolites and proteins. Gene pathways should be downloaded from https://reactome.org/download-data or using the sspa package. 

## Benchmarking scripts

- Sim0: scripts for univariate simulations comparing sensitivity of detection of pathway vs. molecular-level signals 
- Sim1: scripts for comparing PathIntegrate predictive performance to molecular level models and DIABLO
- Sim2: script for benchmarking detection of target enriched pathway using
- Sim3: script for benchmarking predictive performance at the molecular vs pathway level
- Sim4: script for benchmarking predictive performance with varying sub-sample sizes
- MBPLS_Permutation_Testing.py - script for performing permutation testing on the MBPLS model


## Contact

Email cw2019@ic.ac.uk for troubleshooting or questions about the scripts.