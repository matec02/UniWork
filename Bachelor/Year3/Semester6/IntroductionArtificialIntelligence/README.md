# Introduction to Artificial Intelligence – Laboratory Exercises

---

## Lab 1: Search Algorithms

**File:** `solution_1lab.py`

Implements classical search algorithms:
- **Breadth-First Search (BFS)**
- **Uniform Cost Search (UCS)**
- **A\* Search**

These algorithms solve state-space problems using different strategies (uninformed vs. informed). Also includes functionality to evaluate heuristics for **optimism** and **consistency**.

---

## Lab 2: Propositional Logic & Resolution

**File:** `solution_2lab.py`

Implements **resolution-based theorem proving**:
- Parses clauses from knowledge base
- Performs resolution refutation
- Supports dynamic clause addition/removal
- Can simulate an interactive "cooking assistant" that answers logical queries

---

## Lab 3: ID3 Decision Tree Learning

**File:** `solution_3lab.py`

Trains and evaluates a **decision tree classifier** using the **ID3 algorithm**:
- Calculates **information gain**
- Supports **tree depth restriction**
- Outputs prediction accuracy and a **confusion matrix**

---

## Lab 4: Neuroevolution with Genetic Algorithms

**File:** `solution_4lab.py`

Implements **neural network training via genetic algorithms**:
- Supports **one-layer** and **two-layer** sigmoid networks
- Uses evolutionary concepts: selection, crossover, mutation, elitism
- Evaluates training/test performance using **mean squared error**
