	
# pyNPM

pyNPM is a Python driver that implements the non-parametric probabilistic method, including the generation of stochastic Reduced Order Models, and the solution of the hyperparameter identification inverse problem for computational problems solved with solvers from the Aero-Suite (https://bitbucket.org/frg/workspace/repositories/).

## Directory Structure
*Main driver* 
- **source.d/**: Contains the main scripts to build the random matrices, interface with Aero-S, launch the simulations, and perform hyperparameter optimization. *(not problem specific)*

*Problem-specific subdirectories, need to create problem instance and run* 
- **input#ID.d/** 
  - **Input/**: Contains problem-specific solver input files.
  - **Scripts/**: Contains problem-specific functions for generating, modifying, and reading Aero-S input files (e.g., `aeros_runs.py`).
  - **parameters.py**: Contains the definitions of problem-dependent variables.

- **output#ID.d/** 
  - **Basis/**: Contains the Reduced Order Basis (ROB) files.
  - **Mesh/**: Contains outputs from hyperreduction.
  - **Results/**: Contains the probed reference results (e.g., HDM...).

*Note: #ID is an identifier used to refer to specific problem instances.*

*Subdirectory updated by code run* 
- **executables/**: Contains copies of the Aero-S and ROB executables.

## Usage

To run the code:
1. Create and populate the input#ID.d/ output#ID.d/ (e.g. input0.d/ and output0.d for problem ID 0) directories with problem and simulation specific files, and create a copy of the input#ID.d/parameters.py file populated with the corresponding problem specific parameters).
2. Modify the `Run.py` script to instantiate an object of the `SROMAeroS()` class (or a derived class) initialized with the path to the appropriate input and input ID directory.
3. From the command line or a batch script, allocate compute nodes as needed and execute:
    ```bash
    python3 Run.py < /dev/null
    ```

## Additional Notes
- Ensure the input#ID.d/ output#ID.d/ are created and populated before running the code, and most recent version of the aero-s and rob executables are linked within the input#ID.d/parameters.py file. Alternatively, copy the source.d/ subdirectory into a parent directory containing all problem specific input and output directories.  
- Refer to the inline comments and documentation within the code for detailed usage instructions and customization options.
- For any issues or questions, please contact mjazzi@stanford.edu
