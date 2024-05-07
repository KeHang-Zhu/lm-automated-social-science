import typer
import os
import sys
import json
import random
import math

def generate_all_combinations_with_mapping(agentsInfo, variations):
    combined_dicts = []
    
    # Extract all the keys and the corresponding variation lists
    variation_keys = list(variations.keys())
    variation_list = [variations.get(var, {}) for var in variation_keys]
    
    # To store the attribute value mapping
    attribute_value_mapping = {}
    
    # Get the maximum length for each variation
    max_len_list = [max([len(values) for attribute_data in var.values() for attribute, values in attribute_data.items()]) for var in variation_list]
    
    # Use product to iterate over all possible combinations
    from itertools import product
    for idx, combination in enumerate(product(*[range(max_len) for max_len in max_len_list])):
        combined_dict = {}
        variation_dict = {}
        
        for role in agentsInfo:
            agent = agentsInfo[role].copy()
            
            # For each variation, update the attribute
            for var_index, variation_data in enumerate(variation_list):
                for attribute, attribute_values in variation_data.get(role, {}).items():
                    if combination[var_index] < len(attribute_values):
                        agent[attribute] = attribute_values[combination[var_index]]
                        variation_dict[variation_keys[var_index]] = attribute_values[combination[var_index]]
            
            combined_dict[role] = agent
        
        # Update the attribute value mapping
        attribute_value_mapping[str(idx)] = variation_dict
        combined_dicts.append(combined_dict)

    return combined_dicts, attribute_value_mapping


def get_info_from_scm(scm_dict):
    new_dict = {}
    ENDOGENOUS_VARIABLES =[]
    OPERATIONALIZATION = []
    if 'args' in scm_dict and 'variable_dict' in scm_dict['args']:
        variable_dict = scm_dict['args']['variable_dict']
        
        # go over variable_dict
        for key, value in variable_dict.items():
            if 'args' in value and 'agent_measure_question_dict' in value['args']:
                new_dict[key] = value['args']['agent_measure_question_dict']
            if value['class'] == 'EndogenousVariable':
                ENDOGENOUS_VARIABLES.append(key)
                OPERATIONALIZATION.append(value['args']['operationalization_dict']['operationalization'])
                
    measurementsInfo = new_dict
    return measurementsInfo, ENDOGENOUS_VARIABLES, OPERATIONALIZATION


def reorganize_data(attribute_value_mapping, data_to_save_all, scm):
    
    reordered_data = {}
    # for i, combo in enumerate(all_combinations_list):
    for key in attribute_value_mapping.keys():
        # reordered_data[str(i)] = {}
        # Add the matching agents, interaction, and survey entries
        reordered_data[key] = {}
        # for key in ['agents', 'interaction', 'survey']:
        #     if i <= len(data_to_save_all[key]):
        #         reordered_data[str(i)][key] = data_to_save_all[key][i-1]
        for subkey in ['agents', 'interaction', 'survey']:
            if int(key) < len(data_to_save_all[subkey]):
                reordered_data[key][subkey] = data_to_save_all[subkey][int(key)]
    
    combined_data = {
        "scm": scm,
        "data": reordered_data,
        "attribute_value_mapping": attribute_value_mapping
    }
    
    return combined_data

def ensure_directory(directory: str):
    """Ensure that the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_json(data, filename, directory):
    """Save data to a JSON file in the specified directory."""
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    return file_path
        
        
def get_valid_number(prompt):
    """Prompt the user repeatedly until a valid number is entered.
    
    Args:
    prompt (str): The prompt message to display to the user.
    
    Returns:
    int: The user-entered number after confirming it's a valid integer.
    """
    while True:
        try:
            value = typer.prompt(prompt)
            # Attempt to convert the input to an integer
            return int(value)
        except ValueError:
            # If conversion fails, inform the user that the input was not a valid number
            typer.echo("Error: Please enter a valid number.")


def subsampler(target_list: list, proportion: float):
    """
    Subsample a list by a specific proportion.

    Args:
    target_list (list): The list from which to subsample.
    proportion (float): The proportion of the list to sample, expressed as a float (e.g., 0.5 for 50%).

    Returns:
    list: A list containing a subsample of the original list.
    """
    if proportion > 1.000001:
        sys.exit("Error: Exceeding the maximal number! Enter a proportion from 0 to 1.")
    length = len(target_list)
    if length == 1:
        sample_number = 1
        subsample_list =  target_list
    else:
        sample_number = math.ceil(length * proportion) 
        subsample_list = random.sample(target_list, sample_number)
    return subsample_list