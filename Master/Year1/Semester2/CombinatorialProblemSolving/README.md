# Combinatorial Problem Solving - Minimum Consistent Finite Automaton

This project solves the **Minimum Consistent Finite Automaton** problem using three distinct optimization tools:
- Constraint Programming (CP)
- Linear Programming (LP)
- Propositional Satisfiability (SAT)

Each approach is implemented independently and located in its own subdirectory, solving the **same problem** described in `statement.pdf`.

---

## 🔍 Problem Summary

Given:
- A finite set of binary sequences **A** to be accepted
- A finite set of binary sequences **R** to be rejected  
  (such that A ∩ R = ∅)

The goal is to construct a **deterministic finite automaton (DFA)** with the **minimum number of states** that:
- Accepts all sequences in A
- Rejects all sequences in R

The automaton must be consistent with the sample and is free to behave arbitrarily on other inputs.

---

## 📁 Project Structure

Each method follows a similar folder structure:

- `<Method>/`
    - `src/`: Source code, scripts, and checker
    - `out/`: Output files for correctly solved instances
    - `README.md`: Instructions for setup and execution


Each `src` folder includes:
- The **source code** for solving one instance at a time
- A **checker** to validate and visualize solutions
- A **script** to batch run all instances in the `instances/` folder and write valid outputs to `out/`

Each method reads an instance from `stdin` and writes its solution to `stdout` in the specified format.

---

## 🧪 Performance Summary

Out of 100 test instances, the number of optimal solutions found within the time limit (60s per instance) is:

| Method | Solved Instances |
|--------|------------------|
| CP     | 63 / 100         |
| LP     | 95 / 100         |
| SAT    | 99 / 100         |

---

## 🛠️ Running the Code

For each method (`ConstraintProgramming`, `LinearProgramming`, `PropositionalSatisfiability`), consult the `README.md` inside its folder. It provides:
- Dependencies (e.g., solvers, GraphViz)
- Compilation and execution instructions (Linux)
- How to run the checker
- Example usage of the script for batch execution

