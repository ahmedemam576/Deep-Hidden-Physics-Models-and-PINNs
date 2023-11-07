# Physics-Informed Neural Networks (PINN) for 2D Acoustic Wave Simulation

## Overview
We propose an approach to predict the solution for the 2D Acoustic Wave Equation and the full waveform inversion using the Physics Informed Neural Networks (PINNs) and The Hidden Physics Models (HPM).
In this master thesis, computational experiments are designed to test the ability of PINNs and HPM to handle different training datasets and investigate the approach's performance in reconstructing a heterogeneous velocity model. The process of synthetic data generation is out of the scope of this research. Nevertheless, we discuss it at a high level.
PINNs and HPM meshless nature allow smooth implementation for the wave equation's solver and different boundary conditions, such as absorbing boundary conditions, which forms a computational challenge for common classical wave equation solvers. Moreover, the proposed algorithm can easily encode prior knowledge of the geological structure.
It is found that the proposed approach shows acceptable results for forward modeling and full-waveform inversions, even though the classical methods, such as finite difference and spectral element, are more accurate. Moreover, our results show that HPM and PINNs can detect the main structures in the velocity models in complex geological structures. Using PINNs for geophysical inversion shows promising potentials for seismic inversions and joint inversions using different geophysical datasets, such as magnetic and gravitational datasets, because of the PINN's robustness and ability to scale.

## Dependencies
Make sure you have the following Python libraries installed before running the code:
- `numpy`
- `scipy`
- `pyDOE`
- `torch`
- `matplotlib`
- `pandas`

You can install these dependencies using the following command:
```bash
pip install numpy scipy pyDOE torch matplotlib pandas
```

## Code Structure
The code is organized into several sections, and here's a brief overview of each section:

1. Importing Dependencies: Import necessary libraries and modules.
2. Data Preparation: Load and preprocess the data, including creating a 2D grid, calculating distances, and preparing the velocity data.
3. Dataset Classes: Define dataset classes for boundary conditions and initial conditions.
4. BoundaryConditionDataset: Contains methods for preparing boundary condition data.
5. InitialConditionDataset: Contains methods for preparing initial condition data.
6. PDEDataset: Generates data points for the partial differential equation.
7. Main Code: The main part of the code that sets up the PINN model and fits it to the data.

## Running the Code
To run the code, simply execute it. The `if __name__ == "__main__":` block at the end of the code initializes and trains the PINN model. You can adjust the training parameters, such as the number of epochs, optimizer, and learning rate, to fit your specific needs.

Please make sure to set up the required data and paths as per your project requirements.

## Additional Notes
- Make sure you have the necessary data files, such as `vp.txt`, in the specified paths.
- You can customize the PINN architecture, dataset sizes, and training settings to suit your specific problem.

![Result 1](path_to_image1.png)
![Result 2](path_to_image2.png)
![Result 3](path_to_image3.png)


