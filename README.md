# Improving Topologically-Regularized Multiple Instance Learning on Single Cell Images

This is the repository for Milad Bassil's Master's thesis conducted at [Technical University of Munich (TUM)](https://www.tum.de), under the supervision of **Salome Kazeminia** and the advisorship of PIs **Carsten Marr** and **Bastian Rieck** of the groups MarrLab https://github.com/marrlab and AIDOS https://github.com/aidos-lab as part of [Helmholtz Munich](https://www.helmholtz-munich.de/). 

The thesis, titled *Improving Topologically-Regularized Multiple Instance Learning on Single Cell Images*, details the work done in this project, which can be found here: [Insert link later].

## Abstract
In biomedical data analysis, label scarcity is a common issue, and Multiple Instance Learning (MIL) has
emerged as a promising approach to handle limited annotation or coarse labeling challenges. De-
spite the strengths of MIL, it still faces issues such as data-intensive requirements, training instability, and
susceptibility to overfitting in cases with limited data.
Topological regularization is a potential solution that guides training towards more generalizable and
robust feature extraction. Capturing topological information of data spaces requires them to be metric
spaces, meaning that a distance function must be defined over them. Therefore, a robust distance func-
tion capable of producing informative distances, unaffected by semantically invariant transformations, is
crucial for performance. In most relevant literature, a Minkowski distance function is defined over
the space being analyzed. However, when studying the topology of image spaces, a Minkowski distance
function may not suffice. To improve topological regularization on image input spaces, we explore var-
ious distance functions that may yield more representative inter-bag distances, leading to a more robust
regularization of input and latent spaces.
The effectiveness of these distance functions is tested on several synthetic and real-world datasets. Our
real-world application focuses on single blood cell images, evaluating the impact of these functions on
a topologically regularized model for Leukemia subtype identification from blood smear images, called
SCEMILA.


## Repository Overview
This repository contains the code, data, and supplementary materials used in the project. It includes:
- Implementation of the proposed methods and distance functions.
- Scripts for testing and benchmarking distance functions.
- Analysis and visualization tools for evaluating single-cell image embeddings.
- MIL models for leukemia subtype detection, with and without topological regularizations.
## Usage
*Provide instructions for setting up and using the code here. For example:*
1. Clone the repository:
   ```bash
   git clone <repository-link>
2. Create config file to run project:
   in the folder \config one can find many examples of the different ways to run the project.
3. Copy required data in the data folder, as the data is not included in the repository.
