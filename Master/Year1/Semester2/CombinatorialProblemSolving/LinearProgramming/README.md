# Minimum Consistent Finite Automata - CPS Project
## Author: Matija Jakovac

This project solves the **Minimum Consistent Finite Automata** problem using Linear Programming with the **CPLEX** library in C++.

## Requirements

To install dependencies on Ubuntu/Debian:
```bash
sudo apt update
sudo apt install g++ libgecode-dev graphviz
```

Additional for LP

IBM ILOG CPLEX Studio 22.1 is needed to be downloaded from IBM website. When downloaded configure the PATH variables pointing
to /cplex and /concert. It would look like this:

```
CPLEX_HOME = /opt/ibm/ILOG/CPLEX_Studio2211/cplex
CONCERT_HOME = /opt/ibm/ILOG/CPLEX_Studio2211/concert
```

or if it is installed at home/{username}

```
CPLEX_HOME = /home/{username}/cplex/cplex
CONCERT_HOME = /home/{username}/cplex/concert
```


## Compilation

To compile the program, run:

```bash
g++ -O2 -std=c++17 automaton_lp.cpp -o automaton_lp -I$CPLEX_HOME/include -I$CONCERT_HOME/include -L$CPLEX_HOME/lib/x86-64_linux/static_pic -L$CONCERT_HOME/lib/x86-64_linux/static_pic -lilocplex -lconcert -lcplex -lpthread
```
To compile the checker run:

```bash
g++ -std=c++17 -g -O2 -Wall -Wextra -Wno-sign-compare -o checker checker.cc
```

## Execution

After compiling, run the solution on a single instance:

```bash
./automaton_lp < instances/sample.inp > out/sample.out
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
- Compile and run `automaton_lp` on each `.inp` file under a timeout of 60s
- Save `.out` files only for successful executions
- Pass the outputs to the `checker` for validation
- Print a summary of how many passed

## Output

All valid outputs will be saved to the `out/` directory with the same base name as their input `.inp` file.
