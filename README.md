# Automated Social Science: Automated Social Science: Language Models as Scientist and Subjects
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
1. [Overview](#overview)
2. [Disclaimers](#disclaimers-important-please-read-before-using)
3. [Social Scenarios](#what-is-a-social-scenario)
4. [How to use](#getting-started)
5. [Outputs](#output-structure)
6. [Examples](#example-to-play)
7. [Code Architecture](#repository-architecture)


## Overview
This is the code for: https://arxiv.org/abs/2404.11794.
<p align="center">
  <img src="system_overview.jpg" alt="Your Image" width="650" >
</p>

If you make use of this code in any formal way, we would appreciate a citation:

```
@article{manning2024automated,
  title={Automated Social Science: Language Models as Scientist and Subjects},
  author={Manning, Benjamin S and Zhu, Kehang and Horton, John J},
  institution = "National Bureau of Economic Research",
  type = "Working Paper",
  year={2024}
  url"https://www.nber.org/papers/w32381",
}
```

If you are interested in knowing more about and exciting research field of LLMs and social science, there is a repo tracking the recent literature in this field: https://github.com/KeHang-Zhu/Awesome-LLM-for-Social-Science

## Disclaimers IMPORTANT!!!! PLEASE READ BEFORE USING!!!

- These instructions will be a lot easier to read if you have at least skimmed the paper.
- This code system is not perfectly robust, and it can generate unexpected errors. But we want to share what we did. The main point is for people to understand the process in the paper.
- If you run into problems, create an issue, and we will try to help you out or at least respond with an explanation for why it doesn't work.
- If you try and run everything from end-to-end, it can take a substantial amount of time. We have parallelized as much as we can, but the entire process from hypothesis generation to data analysis can take as much as three hours (e.g., the auction simulations in the original paper). This also does not account for when the system breaks, which it sometimes does unfortunately.
  - It can also cost several hundred dollars! So please be careful before launching hundreds of in silica simulations if you don't have any money in your Openai account that you aren't willing to burn.
- With this in mind, we have made the process more interactive such that the user will approve the different phases of the scientific process to minimize errors and keep the in the know about how the system progresses.
 - Of course, the most exciting part of the paper is the automation from start to finish, so we provide guidance on usage with the above warning for reasonably long wait times (hours), possible errors, and financial costs.
 - Note that if you only use 1 or 2 causes, the system shouldn't take too long or be too expensive. Once you get to 3+, though, the number of simulations increases nearly factorially. Hence the time-long wait times and expenses.

Some potential system failures that we expect some people to run into that we cannot preemptively solve are:

- OpenAI rate-limit errors. We have built in a lot of rate limiting, but depending our your specific account details, the system can stall.
- If the experiment didn't induce variation in the outcome, the system will be unable to estimate an SCM. For example, if your scenario is about two people bargaining and they never make a deal. Lavaan, the package we use for estimation, produces an error unless all variables have variance > 0. What this really means is that the experiment failed quite literally---the causes did not affect the outcome! Even if this happens the system will still output a `data.csv` where you can see that the outcome had 0 variance.
- Another related problem can occur if the outcome is conditional and never happens. In the bargaining scenario, if the outcome is the final price of a deal and a deal is never made, then the same error will occur as the previous.
- The LLM generating the SCM simply gives a bad response. While we have many checks in the system to avoid this, if the temperature to the "scientist" portion of the system is above zero, this can sometimes just happen and we do not have a method to perfectly check every possible natural language edge-case.
- In our experience, the system executes an experiment without an error roughly 70% of the time. Usually, just rerunning the process should solve most problems.

<!-- - PyPI: ... -->
<!-- - Documentation:  -->
<!-- - Discord: ... -->

## What is a social scenario?

A scenario is a simple natural language sentence describing multiple people interacting. The system takes in one scenario to "explore" in it's process. Here are the scenarios we use in the paper:

- Two people bargaining over a mug
- A judge is setting bail for a criminal defendant who committed 50,000 dollars in tax fraud.
- A person interviewing for a job as a lawyer.
- 3 bidders participating in an auction for a piece of art starting at fifty dollars.

### Important features of input scenarios

If your social scenarios stay in line with the following, they will be more likely to succeed. None of these features is absolutely necessary, but in our experience simulating these expeirments, your ideas will be much more likely to succesfully run if you follow these token rules:


- Social scenarios must generally reference at least two people. The system always tries to generate 2+ agents for the simulations, so if your scenario does not even implicitly reference 2+ people, it may not work.
   - e.g., "a person playing solitaire in her room alone while crying" would not be a good "social" scenario.
- In general, using social scenarios like those we have described above are good. Some features that work well with the system relevant to our examples are: 
   - Clear references to who the agents will be in the simulations.
   - Obvious quantifiable outcomes and causes.
- **IMPORANT**: The social scenario should be about some topic that can be accomplished only through exchanging text. If the scenario is "3 people building a tower" and the outcome is "whether the tower was built," the tower may never be built since, well, LLMs don't build towers (yet).

Here are a few more examples of simulations that we have successfully completed. Notice the consistency with the above instructions:
- A family arguing whether to get thai food or indian food for dinner.
- Two phd students discussing how many sections to add to their new paper.

## Getting started

### How to install:
Our package is compatible with Python 3.9 - 3.11.

- Start with creating a virtual environment and install the required packages
```
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```
- You need to have a .env file that contains your openai API key
```
touch .env
nano .env
```
In the text editor，replace the  ... with your actual OpenAI API key：
```
OPENAI_API_KEY = ...
```

### How to run the code:
- If you would like to try running the entire system from start to finish, only inputting a scenario of interest, you can use the following command:
```
python -m src end-to-end "scenario you are interested in" --n-causes N_causes
```

- If you would like to have the system generate a hypothesis and run an experiment, you can run the following. Then, follow the instructions in the command line and keep adding information when prompted. Often times, you may be asked to copy and paste information to proceed to the next step; we have added this feature so it is easier to follow and debug.
```
python -m src build-scm "scenario you are interested in"
```

- If you already have a JSON file for the Structural causal model that you have saved and would like to test, you can run the following:
```
python -m src run-experiment-with-scm "path/to/your/scm.json"
```

- If you want to add a new cause for an existing SCM (we only allow for one outcome node for now), you can use `build-cause-with-scm` to build the TARGET_CAUSE :
```
python -m src  build-cause-with-scm TARGET_CAUSE SCM_PATH
```
- If you want to delete a TARGET_CAUSE a, you can use the `delete-line` function.
```
python -m src  delete-cause TARGET_CAUSE TARGET_OUTCOME SCM_PATH 
```

- If you want to do data analysis with an existing `result_{scenario}.json` file, you can use the `analysis-data` function
```
python -m src analysis-data PATH-TO-YOUR-RESULT-JSON-FILE
```


### Optional Parameters
For the `end-to-end` and `run-experiment-with-scm`, we offer some optional parameters for the user to choose from. Raising the number of causes above 3 or the number of max iterations above 20 can dramatically increase the time it takes to run the process and the cost in OpenAI API calls.
1. `--n-causes`: number of causes to include in proposed SCM [default: 2]
2. `--mode`: the mode for running the experiment, either 'sequential' or 'parallel'.  [default: 'sequential']=
3. `--max-interactions`: maximum number of interactions for a single simulation [default: 20]
4. `--temp-scientist`: the temperature for hypothesis generation [default: 0.4]
5. `--temp-subject`: the temperature for the LLM agents to make decisions [default: 0.3]
6. `--subsample` / `--no-subsample`: Do full combinatorial simulation or subsampling.  [default: no subsample]
7. `--sample-proportion`: if you choose `--subsample` mode, you can set the subsampling proportion, 1 is full combination. [default: 1]

The detailed commands with optional parameters are listed below:
```
python -m src end-to-end "scenario-you-are-interested-in" --n-causes N_causes --mode "sequential" --max-interactions 20
python -m src run-experiment-with-scm "path/to/your/scm.json" --mode "parallel" --max-interactions 10
```

### How to get help:
- You can always run the help command to see all the functions
```
python -m src --help 
```
- Or if you want to know in detail how to use a command
```
python -m src `command` --help 
```
## Output Structure
By default, all output files are saved in the `experiment_logs/` directory. **It is important to delete this directory before each new experiment. Otherwise, the files might conflict. We recommend moving the files somewhere else for future use.** When running the commands, several different types of files are produced, detailed as follows:

1. **Structural Causal Models (SCM) Files**:
   - `scm_outcome_{scenario}.json`: Contains only the SCM only consists outcomes.
   - `scm_{scenario}.json`: Contains both outcomes and causes of the SCM.

2. **Agent Information Files**:
   - `agent_{scenario}.json`: Stores information related to the Large Language Model (LLM) agent, including the basic attributes, interaction type and order and the variations.

3. **Interaction History Files**:
   - `history_{scenario}.json`: Records all the interaction histories between agents, stored sequentially for each variation of the causes.

4. **Result Files from Interactions**:
   - `raw_result_{scenario}.json`: Contains all data including SCMs, agents, interaction histories, and measurement results as a dict.
   - `result_{scenario}.json`: Classifies and organizes the data by each variation of the causes, making it easier to analyze specific aspects of the results.

The following two sets of files (5. and 6.) will be stored in a subdirectory `analysis__{date_time}/` with the exact date-time the analysis was executed so they can be easily referenced. Each time time a data analysis is executed, a new subdirectory will be generated with these files.

5. **Result Files from Data Cleaning**:
   - `raw_data.csv`: Raw data from the experiments, still in string form. Each column in the .csv corresponds to one variable.
   - `meta_data.json`: Meta data for the raw data. Describes the type of each variable, the possible values it can take on, and a dictionary with the structure of the SCM
   - `scm_simple.json`: A cleaned up version of `scm_{scenario}.json`. 
   - `final_edge_dict.json`: A dictionary of the SCM structure removed from `meta_data.json` into its own file
   - `data.csv`: Cleaned data frame ready for analysis. Automatically includes columns for interaction variables.
   - `mapped_data.csv`: Copy of `data.csv` with shortened names for the variables to make visualization easier.
   - `final_mapping.json`: Dictionary mapping the column names between `data.csv` and `mapped_data.csv`

6. **Result Files from Data Cleaning**:
   - `{scenario}_scm.tex`: A latex file containing an automatically generated SCM in a pretty format the can be rendered to view the results.
   - `{scenario}_table.tex`:  A latex file containing an automatically generated table decsribing the experiment in a pretty format the can be rendered to view the results.
   - `estimates_df.csv`: The output from estimating the SCM directly generated in R using `lavaan`.

Each file is named according to the scenario it pertains to, ensuring organized and accessible output for further analysis.

## Example to play
- We have prepared some example structural causal models, the four from the paper, if you would like to rerun the simulated conversations.
```
python -m src run-experiment-with-scm "src/Example/3 bidders participating in an auction for a piece of art starting at fifty dollars.json"
``` 
```
python -m src run-experiment-with-scm "src/Example/a judge is setting bail for a criminal defendant who committed 50,000 dollars in tax fraud.json.json"
``` 
```
python -m src run-experiment-with-scm "src/Example/lawyer_interview_3var.json"
``` 
```
python -m src run-experiment-with-scm "src/Example/two people bargaining over a mug.json"
```


## Repository Architecture

This repository is organized to contain the source files of the entire package within the `src` directory. 

If you do not want to run any of the code but are just interested in exploring what we did, we suggest checking out the `end_to_end()` function in `cli.py`. This function simulates the entire process given a scenario description. If you read through the parts of the function throughout the repo, you can see how we organized the process. We also suggest looking at the text files in any directory labeled `\prompt_templates` as these are the prompts we used to make the system work.


Here's an overview of the structure and key components:

## `src` Directory
This package utilizes the Typer command-line interface framework for its operations.

- **`cli.py`**: This file contains the backend functionalities of the package. It houses the core logic that the command-line interface interacts with

- **`utils.py`**: helper functions

- **`__main__.py`**: entry point script


The `src` directory houses all modules supporting the functions. Within this directory, you'll find the following organizations:

### 1. `Human` Module---LLM as subjects

This module is designed to handle aspects related to agents within the system. Mainly representing the agents as independent LLMs and coordinating their interactions. One can think of this module as being relevant to the middle of the system's process, i.e., steps 5 and 6 in the figure at the top of this page.

- **Human.py**: Responsible for instantiating an agent. This script includes the class definitions and necessary functions to create and manage an agent within the system. We discuss generating agents in Section **A.2** of the paper and steps 5 and 6 in the figure at the top of this page.
  
- **Interaction.py**: Contains the various types of interactions that can occur. This file defines different interaction classes or methods that agents can use to interact within the system. We discuss determining how agents interact in Section **A.3** of the paper and steps 5 and 6.

- **prompt_templates**: Contains relevant prompts.

### 2. `JudeaPearl` Module---LLM as a scientist (and subject designer)

This module helps organize the beginning and end of the system's process, i.e., steps 1,2,3,4, and 7 in the figure at the top of this page. 

- **StructuralCausalModelBuilder.py**: This is tasked with setting up variables according to a given scenario description. It stores and manages various properties of the scenario, including the scenario description, the list of agents involved, and maintains multiple dictionaries to keep track of variables, edges, and untracked paths.

- **JudeaPeal.py**: This queries the LLM to get outcomes of interests and the relevant agents for the scenario. Relevant to section **A.1** in the paper and steps 1, 2 in the figure at the top of the page.

- **Variable.py**: Defines Variables and all of the attributes, both exogenous and endogenous. Relevant to section **A.1** and step 2.

- **VariableBuilder.py**: Builds variables. Relevant to section **A.1** and step 2

- **AgentBuilder.py**: Generates agents and selects the interaction protocol. Relevant to sections **A.2** and **A.3** and steps 3 and 4.

- **DataParser.py**: Gets the raw outcomes and causes to be cleaned from the conversation transcript. Relevant to section **A.3** and step 6.

- **DataCleaner.py**: Cleans the data, making everything into floats and ints. Relevant to section **A.3** and step 6.

- **DataAnalyst.py**: Builds the visualizations and lavaan strings for estimation. Relevant to section **A.4** and step 7.

- **estimate_sem.R**: Estimates the paths using Lavaan. Relevant to section **A.4** and step 7.

- **prompt_templates**: Contains relevant prompts.

### 3. `LLM` Module

This module is designed to invoke different Large Language Models (LLM) methods. This module facilitates the integration and utilization of various LLMs and their settings based on the package's needs.

### 4. `Question` Module

This module is designed to call different prompt methods. It includes functions and methods for designing and executing prompts that interact with the underlying models or data structures.

### 5. `Serialization` Module

This module is designed to convert intermediate results into JSONs. It ensures that data can be stored, shared, or further processing in a standardized way.

### 6. `Example` Module

Contains several pre-built SCM (Structural Causal Model) JSON files. These files serve as examples for building your own structural causal models using the framework provided in this repository.

---

## 🔧 Dependencies
The main third-party package requirement is `openai`.

## 💡 Contributing, Feature Asks, and Bugs
Interested collaborating in LLM as scientist and subjects? Found a nasty bug that you would like us to squash? Please send us an email at kehangzhu@gmail.com.
