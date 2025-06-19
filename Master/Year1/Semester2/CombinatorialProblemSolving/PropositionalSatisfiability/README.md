# Minimum Consistent Finite Automata - CPS Project
## Author: Matija Jakovac

This project solves the **Minimum Consistent Finite Automata** problem using Boolean Satisfiability with the **kissat** SAT solver.

## Requirements

To install dependencies on Ubuntu/Debian:
```bash
sudo apt update
sudo apt install g++ libgecode-dev graphviz
```

## Additional for SAT

Kissat SAT solver needs to be set up with running commands:

```bash
git clone https://github.com/arminbiere/kissat
cd kissat
./configure
make
```

To move it to your system path run:

```bash
sudo cp build/kissat /usr/local/bin
```

To verify correct installation run:

```bash
kissat --version
```

It should print something like:

```bash
[version x.x.x]
```


## Compilation

To compile the program, run:

```bash
g++ -O2 -std=c++17 -Wno-unused-result automaton_sat.cpp -o automaton_sat
```

To compile the checker run:

```bash
g++ -std=c++17 -g -O2 -Wall -Wextra -Wno-sign-compare -o checker checker.cc
```

## Execution

After compiling, run the solution on a single instance:

```bash
./automaton_sat < instances/sample.inp > out/sample.out
```

After getting output, check the output via checker:
```bash
./checker < out/sample.out
```

## Addition - Batch Processing

To run the program on **all `.inp` files** inside the `instances/` folder and validate the output using the `checker`:

1. Make sure `run_all.sh` is executable:
```bash
chmod +x run_all.sh
```

2. Run it:
```bash
./run_all.sh
```

This script will:
- Compile and run `automaton_sat` on each `.inp` file under a timeout of 60s
- Save `.out` files only for successful executions
- Pass the outputs to the `checker` for validation
- Print a summary of how many passed

## Output

All valid outputs will be saved to the `out/` directory with the same base name as their input `.inp` file.
