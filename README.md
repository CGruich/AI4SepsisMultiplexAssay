![](docs/assets/multiplex_workflow_overview.png)

**High-throughput, Multiplexed Assay Platform for Integrated Analysis of Sepsis**
___
Sepsis is a complex, heterogeneous syndrome with significant variability in host responses and causal pathogens across patients. The condition is highly dynamic, with patient prognosis often deteriorating within hours without timely intervention. 

A major barrier to sepsis care and research is the lack of point-of-care (POC) sensors capable of rapidly assessing sepsis pathophysiology, progression, and endotypes. 

To address this gap, we developed MIDAS (Multiplexed Intelligent Diffraction Analysis System), a new assay platform that enables rapid, multi-dimensional analyses of the host–pathogen interface (bacterial RNAs and plasma proteins) in a single system. 

MIDAS synergistically integrates (i) a digital holographic image cytometer for high-throughput imaging, (ii) shape-encoded hydrogel sensor arrays for high multiplexing, and (iii) deep learning algorithms for fast, automated image analysis, all optimized for POC use in low resource settings. 

In this proof-of-concept study, MIDAS demonstrated high specificity and sensitivity for multiple protein markers, with detection limits as low as ~1 pg/mL, outperforming conventional ELISA.

Its modular design allows for easy adaptation and expansion of biomarker panels, making it versatile for broader clinical applications.

**This repository contains the project effort and underlying codebase for the AI-component of MIDAS.**
___

# Table of Contents
* [Datasets/Results](docs/data.md) - Datasets used in the study as well as the final training results.
* [Notebooks](docs/notebooks.md) - Notebooks used in the study to train and evaluate the models, including the Bayesian hyperparameter search for the models.
* [Library](docs/library.md) - A brief description of the underlying code library .py files used to facilitate training in the Jupyter notebooks.
___

# Environment Installation
Instructions are given below to reproducibly make the Anaconda environment used to run the notebooks/underlying library of this work:

FILL
___

# Citation
If you found this work to be helpful, please cite our paper: `FILL`

or our code:
___

# Acknowledgements
<p align="top">
    <img src="docs/assets/nsf_logo.png" width="8%">
    &nbsp; &nbsp;
</p>

<sub>
*This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE 1841052.
</sub>

<sub>
Any opinion, findings, and conclusions or recommendations expressed in this material are those of the authors(s) and do not necessarily reflect the views of the National Science Foundation.*
</sub>
