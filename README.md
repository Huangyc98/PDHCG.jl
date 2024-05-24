# PDHCG.jl

This repository contains the official implementation of the restarted Primal-Dual Hybrid Conjugate Gradient (PDHCG) algorithm for solving large-scale convex quadratic programming problems.

Part of the code utilizes [PDQP.jl](https://github.com/jinwen-yang/PDQP.jl) by Jinwen Yang and Haihao Lu.

## Setup

A one-time step is required to set up the necessary packages on the local machine:

```sh
julia --project -e 'import Pkg; Pkg.instantiate()'
```

## Test

To test if the setup is successful, use the following command:

All commands below assume that the current directory is the working directory.

```sh
julia runtest.jl
```

If the setup is completed correctly, you should see the following output:

```.
--- Solver Test Summary ---
CPU test: SUCCESS
GPU test: SUCCESS
---------------------------
Test Summary: | Pass  Total   Time
Solver Test   |    2      2   <time>
```

## Running

### API for using PDHCG

`solve.jl` is the recommended script for using PDHCG, which uses command-line arguments to pass parameters. The results are written in JSON and text files.

```sh
$ julia --project scripts/solve.jl \
--instance_path=INSTANCE_PATH --output_directory=OUTPUT_DIRECTORY \ 
--tolerance=TOLERANCE --time_sec_limit=TIME_SEC_LIMIT --use_gpu=USE_GPU
```

`run.jl` is a script for using PDHCG to solve QP instances. It uses function calls for parameter passing.

### Solving Datasets

`run.jl` is suitable to solve instances in given datasets by setting `folder_path` item in this file.

  For example, to solve example files, you can set the `folder_path` as follows:
  
```sh
folder_path = "./example/" 
```

### Data Requirement

Input data shouldn't have any constraint with inf upper-bound and -inf lower-bound.

## Datasets

All datasets we used can be found at [https://anonymous.4open.science/r/QP_datasets/]

## Interpreting the output

A table of iteration stats will be printed with the following headings.

### runtime

- `#iter`: the current iteration number.
- `#kkt`: the cumulative number of times the KKT matrix is multiplied.
- `seconds`: the cumulative solve time in seconds.

### residuals

- `pr norm`: the Euclidean norm of primal residuals (i.e., the constraint violation).
- `du norm`: the Euclidean norm of the dual residuals.
- `gap`: the gap between the primal and dual objective.

### solution information

- `pr obj`: the primal objective value.
- `pr norm`: the Euclidean norm of the primal variable vector.
- `du norm`: the Euclidean norm of the dual variable vector.

### relative residuals

- `rel pr`: the Euclidean norm of the primal residuals, relative to the right-hand side.
- `rel dul`: the Euclidean norm of the dual residuals, relative to the primal linear objective.
- `rel gap`: the relative optimality gap.
  
### At the end of the run, the following summary information will be printed

- Total Iterations: The total number of Primal-Dual iterations.

- CG  iteration: The total number of Conjugate Gradient iterations.

- Solving Status: Indicating if it found an optimal solution.

### Others

- `json2csv.jl` is a utility to summarize all output files.

## License

```.
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
