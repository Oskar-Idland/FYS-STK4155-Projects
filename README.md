# Projects in FYS-STK4155
- [Projects in FYS-STK4155](#projects-in-fys-stk4155)
  - [Repository Structure](#repository-structure)
  - [Project 1: Machine Learning on Dark Matter Distribution](#project-1-machine-learning-on-dark-matter-distribution)
    - [Project Structure](#project-structure)
      - [`Project_1/`: Root folder for project 1.](#project_1-root-folder-for-project-1)
      - [`Project_1/data/`: Dark matter simulation data stored in text file.](#project_1data-dark-matter-simulation-data-stored-in-text-file)
      - [`Project_1/figs/`: Figures generated from the code. Saved as pdf files.](#project_1figs-figures-generated-from-the-code-saved-as-pdf-files)
      - [`Project_1/src/`: Python source code](#project_1src-python-source-code)
      - [`Project_1/tex/`: $\LaTeX$ files](#project_1tex-latex-files)
  - [Project 2: Neural Networks for Breast Cancer Classification](#project-2-neural-networks-for-breast-cancer-classification)
    - [Project Structure](#project-structure-1)
      - [`Project_2/figs`: Figures generated from the code. Saved as pdf files.](#project_2figs-figures-generated-from-the-code-saved-as-pdf-files)
      - [`Project_2/src`: Python source code. This is divided into the following files, categorized by their purpose.](#project_2src-python-source-code-this-is-divided-into-the-following-files-categorized-by-their-purpose)
        - [Regression Models](#regression-models)
        - [Neural Networks](#neural-networks)
        - [Classification Models](#classification-models)
        - [Misc](#misc)
      - [`Project_2/tex`: $\LaTeX$ files](#project_2tex-latex-files)
      - [`Project_2`: Root folder for project 2.](#project_2-root-folder-for-project-2)
- [Authors](#authors)

## Repository Structure 
All projects are stored in the root directory of the repository. Each project has its own folder and internal structure as described below. 
- [Project 1: Machine Learning on Dark Matter Distribution](#project-1-machine-learning-on-dark-matter-distribution)
- [Project 2: Neural Networks for Breast Cancer Classification](#project-2-neural-networks-for-breast-cancer-classification)

## Project 1: Machine Learning on Dark Matter Distribution
Full report can be found [here](Project_1/tex/out/Project_1.pdf)
### Project Structure  
#### `Project_1/`: Root folder for project 1.

#### `Project_1/data/`: Dark matter simulation data stored in text file.
- `darkmatter.txt`: 

#### `Project_1/figs/`: Figures generated from the code. Saved as pdf files. 

#### `Project_1/src/`: Python source code 
- `data_analysis.ipynb`: Analysis of dark matter data found in `data` folder. 
- `franke.ipynb`: Analysis of the Franke Function.
- `functions.py`: All functions definitions. This is includes a main function with tests, verifying that the functions are working as intended.
#### `Project_1/tex/`: $\LaTeX$ files
- `Project_1.tex`: The full report of the project. 


## Project 2: Neural Networks for Breast Cancer Classification
Full report can be found [here](Project_2/tex/out/main.pdf)
### Project Structure
#### `Project_2/figs`: Figures generated from the code. Saved as pdf files.

#### `Project_2/src`: Python source code. This is divided into the following files, categorized by their purpose. 
##### Regression Models
`RegressionModels.py`: Contains the class for our regression models. 
`regression_anal.ipynb`:  Contains the analysis of the
Franke function data, which we used to validate our implementation.

##### Neural Networks 
- `FFNN.py`: Contains the classes for the neural networks.
- `nn_Franke.ipynb`: Contains the analysis of the Franke
function to validate the implementation of neural net-
works.
- `nn_breast_cancer.ipynb`: Contains the analysis of the
breast cancer data using neural networks. This is where
we train a neural network model and tune the hyperparameters for a standard classification problem.
- `activation_funcs.py`: Contains the activation functions used in the neural networks.

cost_funcs.py: Contains the cost functions used in the
neural networks.
- `cost_funcs.py`: Contains the cost functions used in the
neural networks.

##### Classification Models
- `logistic_regression.ipynb`: Contains the analysis of the breast cancer data using logistic regression.

##### Misc
`utils.py`: Contains a collections of utility functions used in the project. Mostly plotting or data handling functions and other repetitive tasks. 

#### `Project_2/tex`: $\LaTeX$ files
- `main.tex`: The full report of the project.
- `out/main.pdf`: The compiled report. 

#### `Project_2`: Root folder for project 2.
- `requirements.txt`: Required packages for the project, and their versions. 

## Project 3: Surveying Convolution Neural Network Architectures for small datasets
Full report can be found [here](Project_3/tex/out/main.pdf)
### Project Structure
#### `Project_3/data`: Data used in the project. 
- `data/original`: Original breast cancer data.
- `data/working`: Augmented version of original data used to generate more data points. 

#### `Project_3/figs`: Figures generated from the code. Saved as pdf files.

#### `Project_3/src`: Python source code. This is divided into the following files. 
- `cnn.py`: Contains the classes for the convolution neural networks.
- `cnn_analysis.ipynb`: Contains the analysis of the dataset using convolution neural networks.

#### `Project_3/tex`: $\LaTeX$ files
- `main.tex`: The full report of the project.
- `out/main.pdf`: The compiled report.

#### `Project_3`: Root folder for project 3.
- `requirements.txt`: Required packages for the project, and their versions.

---
# Authors 
- [Erik Røset](@erikroset)
- [Oskar Idland](@Oskar-Idland)
- [Krithika Gunasegaran](@krithikaguna) (Project 3)
- [Arangan Subramaniam](@Arra1807) (Project 3)
- [~~Håvard Skåli~~](@paljettrosa) (Project 1)