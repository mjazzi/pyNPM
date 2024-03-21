	
# pyNPM

pyNPM is a Python driver that implements the non-parametric probabilistic method, including the generation of stochastic Reduced Order Models, and the solution of the hyperparameter identification inverse problem for computational problems solved with solvers from the Aero-Suite (https://bitbucket.org/frg/workspace/repositories/).

## Directory Structure

- **source.d/**: Contains the main scripts to build the random matrices, interface with Aero-S, launch the simulations, and perform hyperparameter optimization. *(not problem specific)*

- **input#ID.d/** *(problem specific)*
  - **Input/**: Contains problem-specific solver input files.
  - **Scripts/**: Contains problem-specific functions for generating, modifying, and reading Aero-S input files (e.g., `aeros_runs.py`).
  - **parameters.py**: Contains the definitions of problem-dependent variables.

- **output#ID.d/**(problem specific)*
  - **Basis/**: Contains the Reduced Order Basis (ROB) files.
  - **Mesh/**: Contains outputs from hyperreduction.
  - **Results/**: Contains the probed reference results (e.g., HDM...).

*Note: #ID is an identifier used to refer to specific problem instances.*

- **executables/**: Contains copies of the Aero-S and ROB executables.

## Usage

To run the code:
1. Create and populate the input.d/ output.d/ directories with problem and simulation specific files, and executables/.
2. Modify the `Run.py` script to instantiate an object of the `SROMAeroS()` class (or a derived class) initialized with the path to the appropriate input and input ID directory.
3. From the command line or a batch script, execute:
    ```bash
    python3 Run.py < /dev/null
    ```

## Additional Notes
- Ensure the input.d/ output.d/ and executables/ directory are created and populated before running the code.
- Refer to the inline comments and documentation within the code for detailed usage instructions and customization options.
- For any issues or questions, please contact mjazzi@stanford.edu
