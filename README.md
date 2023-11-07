# Physics-Informed Neural Networks (PINN) for 2D Acoustic Wave Simulation

## Overview
This code implements a Physics-Informed Neural Network (PINN) approach to simulate the propagation of sound in 2D space. It consists of two neural networks: the first network predicts the pressure at each point in space and time, while the second network performs the backward inference step to calculate the density of the medium. This PINN approach is based on the 2D acoustic wave equation. The advantages of this method include the mitigation of handcrafting boundary conditions, instant inference, and the avoidance of hand-crafted grid cells.

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

Feel free to contact the code author or maintainers for any questions or clarifications regarding the code.
