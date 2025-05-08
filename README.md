# Final Python FWI Implementation

This folder contains the final Python implementation of **Frequency Domain Full Waveform Inversion (FWI)** using **JAX**.

## Contents

* **`fwi_script.py`**: Main script orchestrating the FWI pipeline and visualization.
* **`fwi_loss_function.py`**: Definition of the FWI loss function and optimization loop (LBFGS or CG variants).
* **`solve_helmholtz.py`**: Helmholtz equation solver (forward and adjoint wavefields).
* **`nonlinearcg.py`**: Nonlinear conjugate gradient implementation for model updates.
* **`requirements.txt`**: Python package dependencies (see below).
* **`RecordedData.mat`**: Example dataset (MATLAB format) used for reconstruction.

## Prerequisites

* **Python 3.8+**
* **JAX** and **jaxopt** for auto-differentiation and optimization
* **SciPy**, **Matplotlib**, **mat73**, etc. as listed in `requirements.txt`

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository and navigate to this folder:

   ```bash
   cd final_python
   ```
2. Ensure `RecordedData.mat` is present in this directory.
3. (Optional) Edit parameters inside `fwi_script.py` (e.g., number of iterations, element exclusions).
4. Run the FWI pipeline:

   ```bash
   python fwi_script.py
   ```
5. The script will generate:

   * Reconstructions of the sound speed map.
   * Convergence plots (true vs. estimated speed, search direction, gradients).

## Directory Structure

```
final_python/
├── fwi_script.py
├── fwi_loss_function.py
├── solve_helmholtz.py
├── nonlinearcg.py
├── requirements.txt
└── RecordedData.mat
```

## Output

* **Figures** comparing true vs. estimated speed maps, search directions, gradients, and convergence metrics.
* **Statistics** on runtime performance for JAX vs. MATLAB implementations (see slides in root).

## License

This project is released under the MIT License.

## Contact

For questions or contributions, please contact:

* Emilio Ochoa
* Naomi Guevara
* Liumei Ma
