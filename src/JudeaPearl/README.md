
NOTE THAT THIS README MIGHT NOT ALWAYS BE UP TO DATE

## StructuralCausalModelBuilder Class Documentation

### Description:
The `StructuralCausalModelBuilder` class is responsible for setting up variables according to a given scenario description. It stores and handles various properties of the scenario, including the scenario description, the list of agents in the scenario, and multiple dictionaries to keep track of variables, edges, and untracked paths.

The class is equipped with functionalities to write data to JSON files, draw the structural causal model (SCM), build different types of variables, and much more.

### Key Attributes:
- **template_dir**: The directory where the templates for prompts are stored.
- **scenario_description**: A string description of the scenario.
- **agents_in_scenario**: A list of agents involved in the scenario.
- **variables**: A list of variable names.
- **edge_dict**: A dictionary with variable names as keys and their corresponding variable class objects as values.
- **variable_dict**: A dictionary that holds keys as variable names and values as their corresponding variable class objects.
- **untracked_paths_dict**: A dictionary that stores untracked paths in the SCM.

### Key Methods:

#### `__init__(self, scenario_description, agents_in_scenario, template_dir = "prompt_templates")`
Initializes a `StructuralCausalModelBuilder` object with the provided scenario description, agents in the scenario, and a template directory.

#### `edge_dict_to_json(self, directory_path = None, filename = "scm_edge_structure.json")`
Writes the edge dictionary to a JSON file.

#### `_transform_edge_dict(self)`
Transforms the edge dictionary to a format without spaces or apostrophes.

#### `_sanitize_key(self, key)`
Replaces spaces with underscores and apostrophes with nothing in a given key.

#### `scm_to_json(self, directory_path = 'serial_test_vars/', filename=None)`
Writes the variable dictionary (keys are variable names and values are variable class objects) to a JSON file.

#### `__repr__(self)`
Returns a string representation of the `StructuralCausalModelBuilder` object.

#### `draw_scm(self, directory_path = 'serial_test_vars/')`
Draws the structural causal model and saves it as a PNG file.

#### `build_first_endogenous(self, variable_name, num_causes)`
Creates and sets up an outcome variable for a given scenario description.

#### `_get_descendants(self, edge_dict, variable)`
Returns all descendants (children, grandchildren, etc.) of a given variable in the edge dictionary.

#### `build_causal_variable(self, variable_name, possible_covariates, num_recursive_causes = 1, starting_depth = 0, limit_recursion_depth = 2)`
Creates and sets up an exogenous variable for a given scenario description. If the variable is determined to be endogenous, it gets causes and adds them to the variable dictionary and edge dictionary. This process continues recursively until all root notes are exogenous.

Note: If a recursion limit is reached, the method will limit the number of recursive causes to 1 to avoid potential infinite recursion.

### Class Inheritance:
This class inherits from the following classes:
- `PromptMixin`
- `LLMMixin`
- `RegisteredSerializable`


Title: **FirstEndogenousVariableBuilder and CausalVariableBuilder Classes and Their Functions**

## Class: FirstEndogenousVariableBuilder

This class inherits from `LLMMixin` and is responsible for building first endogenous variables. It uses the LLM system to create and operationalize endogenous variables based on the scenario and the agents involved.

**Attributes:**

- `scenario_description` (_string_): A string containing the scenario description.
- `agents_in_scenario` (_list_): A list of agents in the scenario.
- `template_dir` (_string_): Directory where the templates are stored for the prompts.
- `variable_name` (_string_): The name of the variable.

### Method: __init__(self, variable_name, scenario_description, agents_in_scenario, template_dir = "prompt_templates")

**Description:**

Initializes a `FirstEndogenousVariableBuilder` instance.

**Parameters:**

- `variable_name` (_string_): The name of the variable.
- `scenario_description` (_string_): Description of the scenario.
- `agents_in_scenario` (_list_): List of agents in the scenario.
- `template_dir` (_string_, optional): Directory where the templates are stored for the prompts. Default is 'prompt_templates'.

### Method: build_variable(self, num_causes)

**Description:**

Builds an endogenous variable.

**Parameters:**

- `num_causes` (_int_): The number of causes of the variable that should be generated.

**Returns:**

An instance of `EndogenousVariable`.

## Class: CausalVariableBuilder

This class inherits from `LLMMixin` and is responsible for building exogenous variables and checking if they might need to be endogenous variables. It uses the LLM system to create, operationalize, and categorize variables based on the scenario and the agents involved.

**Attributes:**

- `scenario_description` (_string_): A string containing the scenario description.
- `agents_in_scenario` (_list_): A list of agents in the scenario.
- `outcomes_for_build` (_list_): A list of outcomes for the build.
- `possible_covariates` (_list_, optional): A list of possible covariates.
- `template_dir` (_string_): Directory where the templates are stored for the prompts.
- `variable_name` (_string_): The name of the variable.

### Method: __init__(self, variable_name, scenario_description, agents_in_scenario, outcomes_for_build, possible_covariates = None, template_dir = "prompt_templates")

**Description:**

Initializes a `CausalVariableBuilder` instance.

**Parameters:**

- `variable_name` (_string_): The name of the variable.
- `scenario_description` (_string_): Description of the scenario.
- `agents_in_scenario` (_list_): List of agents in the scenario.
- `outcomes_for_build` (_list_): A list of outcomes for the build.
- `possible_covariates` (_list_, optional): A list of possible covariates.
- `template_dir` (_string_, optional): Directory where the templates are stored for the prompts. Default is 'prompt_templates'.

### Method: build_variable(self)

**Description:**

Builds an exogenous variable and checks if it might need to be endogenous.

**Parameters:**

None

**Returns:**

An instance of `ExogenousVariable` or `EndogenousVariable` depending on the operationalization

---
## Variable Class Documentation

### Description:
The `Variable` class is a key component of a causal model. It represents a single variable in the model, storing its attributes and functions related to it. This class also offers functionalities for operationalization, classification, level creation, measurement question creation, determination of whether the variable is latent or observed, and more.

### Key Attributes:
- **name**: The name of the variable.
- **scenario_description**: A string description of the scenario.
- **agents_in_scenario**: A list of agents involved in the scenario.
- **template_dir**: The directory where the templates are stored for the prompts.
- **operationalization_dict**: A dictionary storing the operationalization and the method to obtain the quantity of the variable.
- **variable_type**: A list that stores the type(s) of the variable (continuous, binary, ordinal, count, or nominal).
- **units**: A list storing the units of the variable.
- **levels**: A list that holds the levels of the variable.
- **agent_measure_question_dict**: A dictionary with agent names as keys and corresponding measurement questions as values.
- **measurement_aggregation**: A list to store the method to aggregate the information from the agents.
- **descendant_outcomes**: A list of variables that are direct descendants of the variable.
- **possible_covariates**: A list of all variables in the model besides the descendants (and causes, obviously).

### Key Methods:

#### `__init__(self, name, scenario_description, agents_in_scenario, descendant_outcomes = None, possible_covariates = None, template_dir = "prompt_templates")`
Initializes a `Variable` object with the provided name, scenario description, agents in the scenario, descendant outcomes, possible covariates, and template directory.

#### `__repr__(self)`
Returns a string representation of the `Variable` object.

#### `var_to_dict(self)`
Converts the variable object to a dictionary form, excluding methods and non-built-in attributes.

#### `var_to_json(self, directory_path = None)`
Converts the variable object to a dictionary form and writes it to a JSON file.

#### `operationalize_variable(self) -> None`
Operationalizes the variable and updates the operationalization_dict.

#### `classify_variable_type(self) -> None`
Classifies the type of the variable as continuous, binary, ordinal, count, or nominal and updates the variable_type attribute.

#### `get_variable_units(self) -> None`
Gets the units of the variable and updates the units attribute.

#### `create_levels(self, num_cont_lvls = 5) -> None`
Gets the levels of the variable and updates the levels attribute.

#### `create_measurement_questions(self) -> None`
Creates measurement questions for the variable and matches them to the relevant agent. Additionally, creates a method to aggregate the information from the agents.

### Class Inheritance:
This class inherits from the following classes:
- `PromptMixin`
- `LLMMixin`
- `RegisteredSerializable` 

### Decorators:
The methods in this class use a retry decorator, `retry_on_keyerror_decorator`, that catches `KeyError` exceptions and retries the function up to 5 times before finally raising the `KeyError`.
Sure, I can format it similar to the previous documentation:

### Class EndogenousVariable
---

This class extends `Variable` to handle endogenous variables, which are dependent variables whose value is determined within the scenario. 

Methods:
- `__init__`: Initializes an instance of `EndogenousVariable`, which extends the parent `Variable` class initialization method by introducing causes.
- `get_causes`: Generates a prompt and calls the LLM to retrieve the causes of the endogenous variable. Stores the returned explanations and causes.
- `add_causes`: A helper function that adds causes to the variable's list of causes.
- `remove_cause`: A method to remove a specific cause from the variable's list of causes.

## Class ExogenousVariable
---

This class extends `Variable` to handle exogenous variables, which are independent variables whose value is determined outside the scenario.

Methods:
- `__init__`: Initializes an instance of `ExogenousVariable`, extending the `Variable` class initialization.
- `check_if_endogenous`: Checks if the variable might actually be an endogenous variable, and if so, switches it to an endogenous variable.
- `change_to_endogenous`: If a variable is determined to be endogenous, this method changes it to an endogenous variable by creating a new `EndogenousVariable` instance and copying the attributes.
- `scenario_or_agent_variation`: Determines if the variable is relative to a scenario or an agent, and stores the result.
- `induce_variation_scenario`: Determines how to induce variation in the variable when it is relative to a scenario. Stores the attribute to be varied and its values.
- `induce_variation_individual`: Determines how to induce variation in the variable when it is relative to an individual agent. Stores the attribute to be varied, its values, and the agent.

## Class: JudeaPearl

This class inherits from the `PromptMixin`, `LLMMixin`, and `RegisteredSerializable` classes and represents a directed acyclic graph (DAG) for a given scenario description. It uses the LLM and Prompting systems to generate lists of agents and outcomes based on the scenario.

**Attributes:**

- `template_dir` (_string_): Directory of prompt templates.
- `scenario_description` (_string_): Description of the scenario.
- `outcomes_dict` (_dict_): Dictionary of outcomes.
- `outcomes` (_list_): List of outcomes.
- `agents_dict` (_dict_): Dictionary of agents.
- `agents` (_list_): List of agents.

### Method: __init__(self, scenario_description, template_dir = 'prompt_templates')

**Description:**

Initializes a `JudeaPearl` instance.

**Parameters:**

- `scenario_description` (_string_): Description of the scenario.
- `template_dir` (_string_, optional): Directory of prompt templates. Default is 'prompt_templates'.

### Method: __repr__(self)

**Description:**

Returns a string representation of the `JudeaPearl` instance.

**Parameters:**

None

### Method: get_human_agents(self)

**Description:**

Gets a list of human agents for a given scenario description.

**Parameters:**

None

**Returns:**

A list of agents.

### Method: outcome_generator(self, count = 10)

**Description:**

Generates a generator of outcomes for a given scenario description.

**Parameters:**

- `count` (_int_, optional): Number of outcomes to put in generator. Default is 10.

**Returns:**

A generator of outcomes.

---

You may want to add more details, examples, or additional explanation as necessary based on your actual usage and requirements. This is a basic documentation template that outlines the main components of your class and its methods.

