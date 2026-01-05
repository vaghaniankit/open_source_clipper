# Combined Pipeline Logic (Domain Checking + Runs)

This document describes the combined logic for initial and subsequent pipeline runs, including domain checking and classification behavior.

---

## Run 1 (Initial Run)

### a) Keyword Generation & Initial Pipeline Execution
- Generate **150 keywords** based on:
  - Description field
  - Inclusion criteria
  - Exclusion criteria  
  (Provided via frontend input)
- Execute the following search strategies:
  - Run all **150 generated keywords + "company"**
  - Select a **batch of 30 keywords** from the 150 and run:
    - `keyword + state + company`
    - For **50 states**
    - Total queries per batch: **30 × 50 = 1500**
- Since no prior data exists for the subsector:
  - Detect this as a **fresh state**
  - Run the **entire pipeline end-to-end**
  - **No domain filtering** is applied
- Create a **Domain Store**:
  - Store a **deduplicated list of domains** collected from search APIs
- Perform **classification**:
  - Based on the **inclusion–exclusion criteria** of this run

---

### b) Check Company Count & Incremental Expansion
- After the first pipeline execution:
  - Check if **1500 companies** have been collected
- If **less than 1500**:
  - Run another **batch of 30 keywords**
  - Execute `state + company` queries:
    - **30 × 50 = 1500 queries**
  - For each result:
    - Check if the domain exists in the **Domain Store**
    - For **new (unseen) domains**:
      - Run the remaining pipeline steps (extraction + classification)

---

### c) Repeat Until Target Is Reached
- Repeat step **(b)**:
  - Run a new batch of 30 keywords each time
  - Continue until **1500 companies** are collected

---

## Run 2 / Run 3 / Run 4 / Any Re-run

### a) When 1500 Companies Are Already Reached
- Generate **20 queries** from the new frontend input:
  - Description field
  - Inclusion criteria
  - Exclusion criteria
- Run the pipeline with:
  - Classification based on **current inclusion–exclusion criteria**
- For each result:
  - Check the domain against the **Domain Store**
  - For **unseen domains only**:
    - Run extraction and classification

---

### b) When 1500 Companies Are NOT Yet Reached
- Run:
  - **30 queries**, or
  - The **remaining unexecuted queries** from Run 1
  - Using `state + company` combinations
- Execute the **full pipeline** using the new frontend input:
  - Description field
  - Inclusion criteria
  - Exclusion criteria
- Always:
  - Check domains against the **Domain Store**
  - For **unseen domains**:
    - Run extraction and classification
