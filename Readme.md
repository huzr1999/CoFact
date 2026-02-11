# CoFact: Conformal Factuality Guarantees for Language Models under Distribution Shift

This repository contains the code for our paper *CoFact: Conformal Factuality Guarantees for Language Models under Distribution Shift*.  

## Getting Started  

The main entry point for dataset MedLFQA and WikiData is `main.py`, and for WildChat is `main_wildchat.py`. The code is structured to allow easy experimentation with different methods and distribution shifts.

### Steps to Run:  
1. Install dependencies listed in `requirements.txt`.  
2. Execute the script `run.sh`.  

### Configuring `run.sh`:  
The script includes several tunable arguments:  
- **`config`**: Specifies the path to the configuration file.  
  - Use `config/config_MedQA` for the MedLFQA dataset.  
  - Use `config/config_Wiki` for the WikiData dataset.  
  - Use `config/config_WildChat` for the WildChat dataset.
- **`method`**: Specifies the method for running the experiment.  
  - `CP-unconditional`: Runs SCP.  
  - `CP-conditional`: Runs CondCP.  
  - `Online`: Runs our method, CoFact.  
- **`shift`**: Specifies the type of distribution shift. Options include:  
  - `LinearShift`  
  - `SquareShift`  
  - `SineShift`  
  - `BernoulliShift`  

For WildChat, the `shift` argument is not applicable since we are using the natural distribution shift in the dataset. 

### Output:  
The results of each run will be saved in the `results` directory.  