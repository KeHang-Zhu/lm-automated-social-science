"""This module provides the RS CLI."""
# src/cli.py

from pathlib import Path
from typing import Optional
import logging
import sys
import os
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

sys.path.append('./src/Human')
sys.path.append('./src/LLM')
sys.path.append('./src/JudeaPearl')
sys.path.append('./src/Question')

import typer
from typing import List
from .utils import generate_all_combinations_with_mapping, get_info_from_scm, reorganize_data, ensure_directory, save_json, get_valid_number, subsampler

from src import ERRORS, __app_name__, __version__, config, database

from src import __app_name__, __version__


from LLM import LanguageModel, LLMMixin, llm_json_loader
from AgentBuilder import AgentBuilder
from Human import Human
from Interaction import SocialInteraction
from StructuralCausalModelBuilder import StructuralCausalModelBuilder
from Prompting import PromptMixin
from JudeaPearl import JudeaPearl
from DataParser import DataParser
from DataCleaner import DataCleaner
from DataAnalyst import DataAnalyst

from jinja2 import Environment, FileSystemLoader

current_script_path = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_script_path, '../src/JudeaPearl/prompt_templates')
Interaction_templates_dir = os.path.join(current_script_path, '../src/Human/prompt_templates')
# templates_dir = '/Users/wonderland/Desktop/AgentHub/robot_scientist/JudeaPearl/prompt_templates'
env = Environment(loader=FileSystemLoader(templates_dir))

app = typer.Typer()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# @app.command()
# def init(
#     db_path: str = typer.Option(
#         str(database.DEFAULT_DB_FILE_PATH),
#         "--db-path",
#         "-db",
#         prompt="RS database location?",
#     ),
# ) -> None:
#     """Initialize the RS database."""
#     app_init_error = config.init_app(db_path)
#     if app_init_error:
#         typer.secho(
#             f'Creating config file failed with "{ERRORS[app_init_error]}"',
#             fg=typer.colors.RED,
#         )
#         raise typer.Exit(1)
#     db_init_error = database.init_database(Path(db_path))
#     if db_init_error:
#         typer.secho(
#             f'Creating database failed with "{ERRORS[db_init_error]}"',
#             fg=typer.colors.RED,
#         )
#         raise typer.Exit(1)
#     else:
#         typer.secho(f"The RS database is {db_path}", fg=typer.colors.GREEN)    
        
# @app.command()
def build_agent(scm_json: str, 
                temp_scientist: float = typer.Option(0.3, help="Temperature for the large language model scientist")):
    """Build the agents."""
    typer.echo(f"Building agent from file {scm_json}")
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=temp_scientist, system_prompt='You are a social scientist who loves research and coming up with ideas.')
    
    loaded_scm = StructuralCausalModelBuilder.deserialize(scm_json)
    
    agent_builder = AgentBuilder(template_dir=templates_dir)
    agent_builder.add_LLM(LLM)
    agent_builder.add_scm(loaded_scm)

    exp_role = agent_builder.backend_build_agents_no_extra_attr()
    
    varied_attributes_dict =  agent_builder.backend_return_varied_attributes()
    
    return exp_role, varied_attributes_dict

# @app.command()
def build_interaction(scm_json: str,
                      temp_scientist: float = typer.Option(0.3, help="Temperature for the large language model scientist")):
    """Build the interaction type."""
    # typer.echo(f"Building interaction from SCM JSON")
    
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=temp_scientist, system_prompt='You are a social scientist who loves research and coming up with ideas.')
    
    loaded_scm = StructuralCausalModelBuilder.deserialize(scm_json)
    agent_builder = AgentBuilder(template_dir=templates_dir)
    agent_builder.add_LLM(LLM)
    agent_builder.add_scm(loaded_scm)
    
    gen_func_type, order_dict = agent_builder.backend_get_interaction_info()
    
    return gen_func_type, order_dict

def perform_simulation(
        agent_list: List[str],  
        order_dict: dict, 
        scenario: str, 
        interaction_type: str, 
        ENDOGENOUS_VARIABLES: list, 
        OPERATIONALIZATION: str = typer.Option(None, help="a detailed description of how the variables are calibrated"), 
        max_interactions: int = typer.Option(20, help="Maximum number of interactions"),
        temp_subject: float = typer.Option(0.3, help="Temperature for the large language model agents")
    ):
    """Perform agent simulation"""
    # typer.echo(f"Processing scenario: {scenario} with max interactions: {max_interactions}")
    
    agentsInfo = agent_list
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=temp_subject)
    
    agents = {}
    agent_list = []
    conversation_history = []

    for agent_type, attributes in agentsInfo.items():
        agent = Human(attributes)
        agent.add_LLM(LLM)
        agents[agent_type] = agent
        agent_list.append(agent)

    # intialize the Interaction
    S = SocialInteraction(agent_list, scenario=scenario)
    S.add_LLM(LLM)
    generator = S.gen_func_dispatch[interaction_type](order_dict)
    # determine the order
    FirstAgent = agents[next(generator)]
    SecondAgent = agents[next(generator)]
    others = [agent for agent in agent_list if agent != FirstAgent]
    interactions = 0
    n_left = max_interactions - interactions
    
    ## Make first statement
    statement = FirstAgent.make_public_statement(others, scenario, interactions, n_left, conversation_history)
    print("No.0", FirstAgent.name, ":", statement['statement'])
    
    S.statements.append({FirstAgent.name: statement['statement']})
    conversation_history.append({FirstAgent.name: statement['statement']})

    # Continue to talk until the ending condition is met
    while True:
        interactions += 1
        n_left -= 1
        others = [agent for agent in agent_list if agent != SecondAgent]
        newstatement = SecondAgent.make_public_statement(others, scenario,interactions, n_left, conversation_history)
        name = SecondAgent.name
        FirstAgent = SecondAgent
        SecondAgent = agents[next(generator)]
        statement = newstatement
        print(f"No.{interactions}", name,":", statement['statement'])
        
        conversation_history.append({name: statement['statement']})
        S.statements.append({name: statement['statement']})

        # Determine the ending condition
        to_continue = FirstAgent.to_continue_or_to_finish(scenario, agent_list, OPERATIONALIZATION= OPERATIONALIZATION,ENDOGENOUS_VARIABLES=ENDOGENOUS_VARIABLES,history=conversation_history)
        if not to_continue:
            print('Ending condition is reached!')
            break
        if interactions >= max_interactions-1:
            break  
        
    return conversation_history

def parallel_run(func, combined_dicts, output_dir, **params):
    """
    Run an experiment with given parameters and a specific simulation or measurement function.

    Args:
    func (callable): A function to perform the experiment (e.g., perform_simulation or perform_measurement).
    combined_dicts (list): A list of dictionaries, each representing parameters for a single run.
    output_dir (str): The directory where results should be saved.
    **params: Arbitrary keyword arguments needed for the function `func`.
    """
    total_iterations = len(combined_dicts)
    typer.echo(f"There are {total_iterations} simulations in total")
    
    if not combined_dicts or not output_dir:
        typer.echo("Missing required parameters: 'combined_dicts' or 'output_dir'")
        return
    scenario = params.get('scenario', 'default_scenario')
    
    partial_call = partial(func, **params)
    with ProcessPoolExecutor() as executor:
        combined_results = list(executor.map(partial_call, combined_dicts))
        
    histories = combined_results
    data_to_save = {"histories": histories}
    save_json(data_to_save, f"history_{scenario}.json", output_dir)
    
    return histories
    
def sequential_run(func, combined_dicts, output_dir, **params):
    """
    Run an experiment sequentially with given parameters and a specific simulation or measurement function.

    Args:
    func (callable): A function to perform the experiment (e.g., perform_simulation or perform_measurement).
    combined_dicts (list): A list of dictionaries, each representing parameters for a single run.
    output_dir (str): The directory where results should be saved.
    **params: Arbitrary keyword arguments needed for the function `func`.
    """
    total_iterations = len(combined_dicts)
    typer.echo(f"There are {total_iterations} simulations in total")
    logs = []
    
    with tqdm(total=total_iterations, desc="Progress") as pbar:
        for agent_dict in combined_dicts:
            print(agent_dict)
            pbar.update(1)
            # Call the provided function with parameters unpacked and the current agent_dict
            history = func(agent_dict, **params)
            logs.append(history)
            data_to_save = {"histories": logs}

             # Save every logs at once after the loop to save the file correctly
            save_json(data_to_save, f"history_{params.get('scenario', 'default_scenario')}.json", output_dir)
    return logs
    

def call_measurement(history: str, measurementsInfo: str, agent_str: str, ENDOGENOUS_VARIABLES: List[str], SCNEARIO_DESCRIPTION: str, OPERATIONALIZATION:str):
    """Perform measurements based on the simulation history"""
    responses = {}
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=.0)
    prompt_mixin = PromptMixin()

    agents = {}
    agent_list = []
    NAME_AND_ROLE = []
    replacement_dict = {"your role is": "role", "your name": "name"}
    for _, each_agent in agent_str.items():
        NAME_AND_ROLE.append([f"{replacement_dict[key]}: {value}" for key, value in each_agent.items() if key in ["your role is", "your name"]])
    # print("=============",NAME_AND_ROLE)
    EXDOGENOUS = [variable_name for variable_name in measurementsInfo if variable_name not in ENDOGENOUS_VARIABLES]

    # measurementsInfo only has "oracle" question
    for agent_type, attributes in agent_str.items():
        agents[agent_type] = Human(attributes)
        agents[agent_type].add_LLM(LLM)
        agent_list.append(agents[agent_type])
            
    responses = {}

    for variable_name in measurementsInfo:
        if variable_name in ENDOGENOUS_VARIABLES:
            for agent_name, questions_data in measurementsInfo[variable_name].items():
                
                questions = questions_data if isinstance(questions_data, list) else [questions_data]
                
                for question in questions:
                    if variable_name not in responses:
                        responses[variable_name] = {}
                    if agent_name not in responses[variable_name]:
                        responses[variable_name][agent_name] = {}
                    
                    # Oracle case
                    if agent_name == 'oracle':
                        prompt_params = { 
                            "SCNEARIO_DESCRIPTION": SCNEARIO_DESCRIPTION,
                            "NAME_AND_ROLE": NAME_AND_ROLE,
                            "history": history,
                            "question":question,
                            "EXDOGENOUS": EXDOGENOUS,
                            "variable_name":variable_name,
                            "OPERATIONALIZATION": OPERATIONALIZATION
                            }
                        prompt = prompt_mixin.generate_prompt("survey_oracle.txt", template_dir = Interaction_templates_dir, **prompt_params)
                        
                        survey_answer = LLM.call_llm(prompt)
                        
                        responses[variable_name][agent_name][question] = survey_answer

                    # Agent cases
                    else:
                        others = [agent for agent in agent_list if agent != agent_name]
                        survey_answer = agents[agent_name].survey(others, SCNEARIO_DESCRIPTION, question, history, EXDOGENOUS=EXDOGENOUS, VARIABLE=variable_name, OPERATIONALIZATION=OPERATIONALIZATION)
                        # store to response
                        responses[variable_name][agent_name][question] = survey_answer
    return responses
    
# @app.command() 
def perform_measurement(args):
    """Perform measurements based on the simulation history"""
    history, agent_str, measurementsInfo, ENDOGENOUS_VARIABLES,scenario, OPERATIONALIZATION = args
    typer.echo(f"Performing measurement on the question {measurementsInfo}.")
    return call_measurement(history=history, agent_str=agent_str, measurementsInfo=measurementsInfo, ENDOGENOUS_VARIABLES= ENDOGENOUS_VARIABLES, SCNEARIO_DESCRIPTION=scenario, OPERATIONALIZATION= OPERATIONALIZATION)

@app.command() 
def end_to_end(
    scenario: str = typer.Argument(..., help="The scenario description for which the simulation is run."),
    n_causes: int = typer.Option(2, help="The number of causal factors to consider in the simulation."),
    mode: str = typer.Option("sequential", help="The mode for running the experiment, either 'sequential' or 'parallel'."), max_interactions: int = typer.Option(20, help="Maximum number of interactions"),
    temp_scientist: float = typer.Option(0.4, help="Temperature for the large language model agents"),
    temp_subject: float = typer.Option(0.3, help="Temperature for the large language model agents"),
    subsample: bool = typer.Option(False, help="Do full combinatorial simulation or subsampling."),
    sample_proportion: float = typer.Option(0.1, help="subsampling proportion.")
):
    """
    Run an end-to-end automated social science simulation.
    A good way to get acquainted with what we did is to go through the different parts of this function as it executes the entire social scientific process.
    WARNING: THIS FUNCTION MAY TAKE SEVERAL HOURS TO RUN AND CAN COST HUNDREDS OF USD$ IN OPENAI API CALLS.
    
    We suggest using very clear natural language sentences. 
    The more specific you are in the description of the scenario of interest, the more "focused" the experiment will likely be.
    In the paper, we used the following: 

    Two people bargaining over a mug.
    A judge is setting bail for a criminal defendant who committed 50,000 dollars in tax fraud.
    A person interviewing for a job as a lawyer.
    3 bidders participating in an auction for a piece of art starting at fifty dollars.

    This command will first build a structural cause model (SCM) with ONE outcome and n_causes.
    It will then automatically instantiate the agents relevant to test the proposed causal paths in the SCM.
    It will determine how these agents should interact and will then simulate their conversations in parallel.
    It then surveys the agents to determine what happened, cleans the data, and then automatically analyzes the data.

    Output:
            This function will generate many files, some of which include:
            1. A latex file with a figure of the fitted SCM and a table with the high level information about the simulation
            2. A csv with the path estimates
            3. A csv with the data from the simulations
            4. A json with the SCM, all variable details, and the full conversations from every single agent simulation.
            
    """
    output_dir = "experiment_logs"
    ensure_directory(output_dir)
    
    typer.echo("Build the SCM")
    agent_list = get_agents(scenario=scenario, temp_scientist=temp_scientist)

    outcome_list = get_outcome(n_outcomes=1, scenario=scenario, temp_scientist=temp_scientist)
    
    target_outcome = outcome_list[0]
    scm = build_outcome(target_outcome=target_outcome, scenario=scenario, agents=agent_list, temp_scientist=temp_scientist)
    typer.echo(f"Success in building outcome {target_outcome}!")
    ## check point
    save_json(scm, f"scm_outcome_{scenario}.json", output_dir)
    typer.echo(f"SCM saved to {output_dir}")
    
    ## Build cause
    cause_list = get_cause(target_outcome=target_outcome, n_causes=n_causes, scm_json=scm, temp_scientist=temp_scientist)
    
    for target_cause in cause_list['causes']:
        scm = build_cause(target_cause=target_cause, target_outcome=target_outcome, scm_json=scm, temp_scientist=temp_scientist)
        typer.echo(f"Success in adding the cause: {target_cause}!")
        saved_file_path=save_json(scm, f"scm_{scenario}.json", output_dir)
    
    ## check point
    typer.echo(f"Success in building the SCM and it's saved to {output_dir}")
    
    run_experiment_with_scm(
        scm_path = saved_file_path, 
        mode=mode, 
        max_interactions=max_interactions,
        temp_scientist=temp_scientist, 
        temp_subject=temp_subject,
        subsample=subsample,
        sample_proportion=sample_proportion)
    
    return "Success!"

@app.command() 
def run_experiment_with_scm(
    scm_path: str = typer.Argument(..., help="The path to the JSON file containing the SCM data."),
    mode: str = typer.Option("sequential", help="The mode for running the experiment, either 'sequential' or 'parallel'."), 
    max_interactions: int = typer.Option(20, help="Maximum number of interactions"),
    temp_scientist: float = typer.Option(0.3, help="Temperature for the large language model scientist"),
    temp_subject: float = typer.Option(0.3, help="Temperature for the large language model agents"),
    subsample: bool = typer.Option(False, help="Do full combinatorial simulation or subsampling."),
    sample_proportion: float = typer.Option(0.1, help="subsampling proportion.")
):
    """
    Run the simulation based on a structural causal model (SCM).
    The code will first instantiate all the agents involved in the scenario as separate LLM calls.
    And then engage these agents to make statements according to the pre-defined order_dict.
    The interactions will stop if the ending condition is met.
    """
    
    typer.echo(f"Running automated social science simulation for {scm_path}: ")
    output_dir = "experiment_logs"
    ensure_directory(output_dir)
    
    try:
        with open(scm_path, 'r') as file:
            scm = json.load(file)
        scm_dict = json.loads(scm)
        scenario = scm_dict["args"]["scenario_description"]
        measurementsInfo, ENDOGENOUS_VARIABLES, OPERATIONALIZATION = get_info_from_scm(scm_dict)
    except FileNotFoundError:
        logging.error("SCM file not found.")
        return "File Error!"
    except json.JSONDecodeError:
        logging.error("Error decoding SCM JSON.")
        return "JSON Error!"  
      
    ## Added the check point
    agent_filepath = os.path.join(output_dir, f"agent_{scenario}.json")
    ## give the scm (give by uploading)
    if os.path.exists(agent_filepath):
        print('agents info already built')
        # If the file exists, read the content
        try:
            with open(agent_filepath, "r") as file:
                data_dict = json.load(file)
                # print(type(data_dict))
            exp_role = data_dict["exp_role"]
            variations = data_dict["variations"]
            interaction_type = data_dict["interaction_type"]
            order_dict = data_dict["order_dict"]
            sample_dict = data_dict["combined_dicts"]
        except FileNotFoundError:
            logging.error("Agent file not found.")
            return "File Error!"
        except json.JSONDecodeError:
            logging.error("Error decoding Agent JSON.")
            return "JSON Error!"
        # measurementsInfo = data_dict["measurementsInfo"]
        combined_dicts, attribute_value_mapping = generate_all_combinations_with_mapping(exp_role, variations)

    else:
        exp_role, variations = build_agent(scm_json=scm, temp_scientist=temp_scientist)
        interaction_type, order_dict = build_interaction(scm_json=scm, temp_scientist=temp_scientist)
        print('Roles',exp_role, 'Attribute variations', variations)
        print('Interacton', interaction_type, 'Order',order_dict)
        ## add a check point
        combined_dicts, attribute_value_mapping = generate_all_combinations_with_mapping(exp_role, variations)
        
        if subsample is True:
            sample_dict = subsampler(combined_dicts, proportion=float(sample_proportion))
        else:
            sample_dict = combined_dicts
        
        data_to_save = {
            "exp_role": exp_role,
            "variations": variations,
            "interaction_type": interaction_type,
            "order_dict": order_dict,
            'measurementsInfo': measurementsInfo,
            'combined_dicts': sample_dict
        }
        
        # Save the string to a file
        save_json(data_to_save, f"agent_{scenario}.json", output_dir)
            
    ## Added the check point
    history_filepath = os.path.join(output_dir, f"history_{scenario}.json")
    if os.path.exists(history_filepath):
        print('agents history already there')
        # If the file exists, read the content
        try:
            with open(history_filepath, "r") as file:
                data_dict = json.load(file)
                histories = data_dict['histories']
            print('variations', len(histories)) 
        except FileNotFoundError:
            logging.error("History file not found.")
            return "File Error!"
        except json.JSONDecodeError:
            logging.error("Error decoding History JSON.")
            return "JSON Error!"
    else:
        params = {
            "scenario": scenario,
            "order_dict": order_dict,
            "interaction_type": interaction_type,
            "ENDOGENOUS_VARIABLES": ENDOGENOUS_VARIABLES,
            "OPERATIONALIZATION": OPERATIONALIZATION,
            "max_interactions": max_interactions,
            "temp_subject": temp_subject
        }
            
        if mode == "parallel":
            histories = parallel_run(perform_simulation, sample_dict, output_dir, **params)
        elif mode == "sequential":
            histories = sequential_run(perform_simulation, sample_dict, output_dir, **params)
            
    
    # Perform parallel measurements  
    with ProcessPoolExecutor() as executor:
        args = zip(histories, sample_dict, [measurementsInfo]*len(histories), [ENDOGENOUS_VARIABLES]*len(histories), [scenario]*len(histories),[OPERATIONALIZATION]*len(histories))
        surveys = list(executor.map(perform_measurement, args))
        
    data_to_save_all = {
            "scm": scm,
            "agents": sample_dict,
            "interaction": histories,
            "survey": surveys
        }
    # Save the string to a file
    save_json(data_to_save_all, f"raw_result_{scenario}.json", output_dir)
    
    ## re-order the JSON data and save
    reordered_json_str = reorganize_data(attribute_value_mapping, data_to_save_all, scm)
    save_json(reordered_json_str, f"result_{scenario}.json", output_dir)
         
         
    ## Analyze the data and fit the SCM
    data_path = os.path.join(output_dir, f"result_{scenario}.json")
    analysis_data(file_path=data_path, temp_scientist=temp_scientist)
    
    return "Success!"

@app.command() 
def build_scm(
    scenario: str,
    temp_scientist: float = typer.Option(0.4, help="Temperature for the large language model scientist")):
    """Build the structural causal model from scratch"""
    output_dir = "experiment_logs"
    ensure_directory(output_dir)
    
    typer.echo("Build the SCM")
    agent_list = get_agents(
        scenario=scenario, temp_scientist=temp_scientist)

    n_outcome = get_valid_number("How many outcomes do you want to generate?\n Please enter a number below:\n >>>> ")
    outcome_list = get_outcome(n_outcomes=n_outcome, scenario=scenario, temp_scientist=temp_scientist)
    typer.echo(f"These are {n_outcome} of possible outcomes: {outcome_list}")
    
    target_outcome = typer.prompt("which outcome do you want to build?\n Please write the outcome below:\n >>>> ")
    scm = build_outcome(target_outcome=target_outcome, scenario=scenario, agents=agent_list, temp_scientist=temp_scientist)
    
    ## check point
    save_json(scm, f"scm_outcome_{scenario}.json", output_dir)
    typer.echo(f"SCM saved to {output_dir}")
    
    ## Build cause
    n_causes = get_valid_number("How many causes do you want to generate?\n Please enter a number below:\n >>>>  ")
    cause_list = get_cause(target_outcome=target_outcome, n_causes=n_causes, scm_json=scm, temp_scientist=temp_scientist)
    typer.echo(f"These are {n_causes} of possible causes: {cause_list['causes']}")
    
    target_cause = typer.prompt("which cause do you want to build?\n Please write the cause below:\n >>>>  ")
    scm = build_cause(target_cause=target_cause, target_outcome=target_outcome, scm_json=scm, temp_scientist=temp_scientist)
    typer.echo("Success in building SCM!")
    
    ## check point
    save_json(scm, f"scm_{scenario}.json", output_dir)
    typer.echo(f"SCM saved to {output_dir}")
    
    return "Success!"

@app.command() 
def get_agents(
    scenario: str = typer.Argument(..., help="The scenario name for which to retrieve associated agents."),
    temp_scientist: float = typer.Option(0.4, help="Temperature for the large language model scientist")
):
    """
    Generate agents involved in a given scenario.
    
    This function fetches the names of agents participating in a specified scenario using LLM scientist. 
    """
    typer.echo(f"Get agent for scenario '{scenario}'")
    
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=temp_scientist, system_prompt='You are a social scientist who is interested in studying social scenarios.')
    JP = JudeaPearl(scenario, template_dir=templates_dir)
    JP.add_LLM(LLM)
    human_agents_list = JP.backend_get_human_agents()
    typer.echo(human_agents_list)
    return human_agents_list

# @app.command() 
def get_outcome(
    scenario: str, 
    n_outcomes: int,
    temp_scientist: float = typer.Option(0.4, help="Temperature for the large language model scientist")
    ):
    """
    Generating possible outcomes for scenario
    """
    typer.echo(f"Generating possible outcomes for scenario {scenario}")

    LLM = LanguageModel(family="openai", model="gpt-4", temperature=temp_scientist, system_prompt='You are an economist who is interested in studying social scenarios.')
    JP = JudeaPearl(scenario, template_dir=templates_dir)
    JP.add_LLM(LLM)
    outcomes = JP.backend_outcome_generator(count=n_outcomes)
    outcome_list = list(outcomes)
    typer.echo(outcome_list)
    return outcome_list

# @app.command() 
def build_outcome(
    target_outcome: str, 
    scenario: str, 
    agents: str,
    temp_scientist: float = typer.Option(0.4, help="Temperature for the large language model scientist")
    ):
    """
    Build the selected outcome for scenario
    """
    typer.echo(f"Build the selected outcome {target_outcome} for scenario {scenario}")
    
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=temp_scientist, system_prompt='You are a social scientist who loves research and coming up with ideas.')
    
    ## Define the scenario and agents
    scenario_description = scenario
    agents_in_scenario = agents
    # print(scenario_description, target_outcome, agents_in_scenario)
    
    # Instantiate the SCM
    scm = StructuralCausalModelBuilder(scenario_description, agents_in_scenario, template_dir=templates_dir)
    scm.add_LLM(LLM)

    ## only run for the first node
    endo_var = scm.backend_build_first_endogenous(target_outcome)
    typer.echo('####1#######')
    print(endo_var)
    
    ## Serializatin of scm
    scm_json = scm.serialize()
    
    return scm_json

# @app.command() 
def get_cause(
    target_outcome: str, 
    n_causes: int, 
    scm_json: str,
    temp_scientist: float = typer.Option(0.4, help="Temperature for the large language model scientist")
    ):
    """
    Get possible causes for given outcome
    """
    typer.echo(f"Get possible causes for given outcome {target_outcome}")
    
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=temp_scientist, system_prompt='You are a social scientist who loves research and coming up with ideas.')
    
    # Instantiate the SCM
    loaded_scm = StructuralCausalModelBuilder.deserialize(scm_json)
    loaded_scm.add_LLM(LLM)
    
    cause = loaded_scm.backend_get_causes_for_variable(target_outcome, num_causes = n_causes)
    typer.echo('####2#######')
    typer.echo(cause)
    
    cause_list = list(cause)
    
    ## Serializatin of scm
    scm_json = loaded_scm.serialize()
    
    return {'scm': scm_json, 'causes': cause_list}

# @app.command()
def build_cause(
    target_cause, 
    target_outcome: str, 
    scm_json: str,
    temp_scientist: float = typer.Option(0.4, help="Temperature for the large language model scientist")
    ):
    """
    Build the selected cause for outcome
    """
    typer.echo(f"Build the selected cause {target_cause} for outcome {target_outcome}")
    
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=temp_scientist, system_prompt='You are a social scientist who loves research and coming up with ideas.')
    
    loaded_scm = StructuralCausalModelBuilder.deserialize(scm_json)
    loaded_scm.add_LLM(LLM)
    
    loaded_scm.backend_add_cause(target_outcome, target_cause)
    typer.echo('####3#######') 
    typer.echo('Adding cause...')
    # print(loaded_scm)
    
    loaded_scm.backend_build_causal_variable(target_cause, loaded_scm.variables)
    typer.echo('####4#######')
    typer.echo('Building information...')
    print(loaded_scm)
    
    scm_json = loaded_scm.serialize()
    
    return scm_json

@app.command()
def build_cause_with_scm(
    target_cause: str = typer.Argument(..., help="The specific cause variable to build for the SCM."),
    scm_path: str = typer.Argument(..., help="File path to the JSON format SCM containing structure information."),
    temp_scientist: float = typer.Option(0.4, help="Temperature for the large language model scientist")
):
    """
    Build the selected cause for a Structural Causal Model (SCM).

    This function handles the logic of building a specific cause into an existing SCM. It adjusts the model
    to incorporate new causal relationships or updates existing ones based on the specified cause.

    Raises:
        FileNotFoundError: If the specified scm_path does not lead to a valid file.
        json.JSONDecodeError: If there is an error in decoding the JSON data from the specified file.
    """
    ## read from JSON file
    try:
        with open(scm_path, 'r') as file:
            scm = json.load(file)
        scm_dict = json.loads(scm)
        scenario = scm_dict["args"]["scenario_description"]
        target_outcome = scm_dict["args"]['variables'][0]
        print(f"Successfully loaded SCM data for {target_cause}.")
    except FileNotFoundError:
        logging.error("SCM file not found.")
        return "File Error!"
    except json.JSONDecodeError:
        logging.error("Error decoding SCM JSON.")
        return "JSON Error!"
    typer.echo(f"Build the selected cause {target_cause} for outcome {target_outcome}")
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=temp_scientist, system_prompt='You are a social scientist who loves research and coming up with ideas.')
    output_dir = "experiment_logs"
    ensure_directory(output_dir)
    
    # target_outcome = f"'{target_outcome}'"
    
    loaded_scm = StructuralCausalModelBuilder.deserialize(scm)
    loaded_scm.add_LLM(LLM)
    
    loaded_scm.backend_add_cause(target_outcome, target_cause)
    typer.echo('####3#######') 
    typer.echo('Adding cause...')
    # print(loaded_scm)
    
    loaded_scm.backend_build_causal_variable(target_cause, loaded_scm.variables)
    typer.echo('####4#######')
    typer.echo('Building information...')
    print(loaded_scm)
    
    scm_json = loaded_scm.serialize()
    save_json(scm_json, f"scm_{scenario}.json", output_dir)
    
    return "Success!"

@app.command()
def add_line(
    target_cause: str = typer.Argument(..., help="The cause variable you want to connect."),
    target_outcome: str = typer.Argument(..., help="The outcome variable that the cause affects."),
    scm_path: str = typer.Argument(..., help="The path to the JSON file containing the SCM data."),
    temp_scientist: float = typer.Option(0.4, help="Temperature for the large language model scientist")
):
    """
    Add a line in the DAG with an existing cause and target outcome.

    This command updates the DAG by adding a direct connection from an exisiting cause to a specified outcome. 
    It is used primarily to modify the structure of a causal model based on new insights or data.

    Args:
    target_cause (str): The name of the cause variable.
    target_outcome (str): The name of the outcome variable that is affected by the cause.
    scm_path (str): The file path to the structural causal model (SCM) stored in JSON format.
    """
    typer.echo(f"Adding a line from {target_cause} to {target_outcome} in SCM at {scm_path}.")
    output_dir = "experiment_logs"
    ensure_directory(output_dir)
    try:
        with open(scm_path, 'r') as file:
            scm = json.load(file)
        scm_dict = json.loads(scm)
        scenario = scm_dict["args"]["scenario_description"]
    except FileNotFoundError:
        logging.error("SCM file not found.")
        return "File Error!"
    except json.JSONDecodeError:
        logging.error("Error decoding SCM JSON.")
        return "JSON Error!"
    typer.echo(f"Add a cause relation between cause {target_cause} and  outcome {target_outcome}")
    
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=temp_scientist, system_prompt='You are a social scientist who loves research and coming up with ideas.')
    target_outcome = f"'{target_outcome}'"
    
    loaded_scm = StructuralCausalModelBuilder.deserialize(scm)
    loaded_scm.add_LLM(LLM)
    
    loaded_scm.backend_add_cause(target_outcome, target_cause)
    typer.echo('####3#######')
    print(loaded_scm)
    
    scm_json = loaded_scm.serialize()
    save_json(scm_json, f"scm_{scenario}.json", output_dir)
    
    return scm_json

@app.command() 
def delete_cause(
    target_cause: str = typer.Argument(..., help="The name of the cause variable to remove from the SCM."),
    target_outcome: str = typer.Argument(..., help="The name of the outcome variable affected by the cause."),
    scm_path: str = typer.Argument(..., help="The path to the JSON file containing the SCM data.")
):
    """
    Delete a cause in the Structural Causal Model (SCM).

    This command removes an existing causal link between the specified 'target_cause' and 'target_outcome'
    within the specified SCM. It is intended for modifying the structure of the causal model when a cause is
    deemed irrelevant or incorrectly specified.

    Args:
    target_cause (str): The identifier or name of the cause variable you wish to remove.
    target_outcome (str): The identifier or name of the outcome variable from which the cause is removed.
    scm_path (str): The filesystem path to the JSON file that stores the SCM data structure.
    """
    typer.echo(f"Attempting to delete the cause '{target_cause}' affecting '{target_outcome}' from SCM at {scm_path}.")
    
    output_dir = "experiment_logs"
    ensure_directory(output_dir)
    try:
        with open(scm_path, 'r') as file:
            scm = json.load(file)
        scm_dict = json.loads(scm)
        scenario = scm_dict["args"]["scenario_description"]
    except FileNotFoundError:
        logging.error("SCM file not found.")
        return "File Error!"
    except json.JSONDecodeError:
        logging.error("Error decoding SCM JSON.")
        return "JSON Error!"
    typer.echo(f"Delete the cause {target_cause}")
    
    loaded_scm = StructuralCausalModelBuilder.deserialize(scm)
    
    loaded_scm.backend_remove_cause(target_outcome, target_cause)
    typer.echo('####5#######')
    print(loaded_scm)
    
    scm_json = loaded_scm.serialize()
    save_json(scm_json, f"scm_{scenario}.json", output_dir)
    
    typer.echo("Deletion complete.")
    return scm_json

@app.command() 
def analysis_data(
    file_path: str = typer.Argument(..., help="The path to the JSON file containing the experimental data."),
    # scenario: str = typer.Argument(..., help="The scenario of the experimental data."),
    temp_scientist: float = typer.Option(0.1, help="Temperature for the large language model scientist")
    ):
    '''
    Analysis the experimental results to fit the pre-registered Structural Causal Model (SCM).
    
    The code will first parses the data and clean up the parsed data into a csv file.
    Then it will use the linear regression to fit the paths defined in the SCM.
    
    '''
    typer.echo(f"Analyzing the results at {file_path}.")
    timestring = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"experiment_logs/analyis__{timestring}"
    # output_dir = "experiment_logs"
    ensure_directory(output_dir)
    
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=temp_scientist)

    try:
        with open(file_path, 'r') as f:
            interaction_data = json.load(f)
    except FileNotFoundError:
        logging.error("SCM file not found.")
        return "File Error!"
    except json.JSONDecodeError:
        logging.error("Error decoding SCM JSON.")
        return "JSON Error!"
    
    # data parser
    data_parser = DataParser(interaction_data, template_dir = templates_dir)
    data_parser.add_LLM(LLM)
    data_parser.backend_clean_data(output_dir)
        
    # data cleaner
    raw_data = pd.read_csv(os.path.join(output_dir, "raw_data.csv"))
    with open(os.path.join(output_dir, "meta_data.json"), "r") as f:
        meta_data = json.load(f)
        
    data_cleaner = DataCleaner(interaction_data, raw_data, meta_data)
    data_cleaner.generate_final_df()
    data_cleaner.save_data(path=output_dir)
    
    # data analyst
    data = pd.read_csv(os.path.join(output_dir, "data.csv"))
    with open(os.path.join(output_dir, "meta_data.json"), "r") as f:
        meta_data = json.load(f)
    # with open(os.path.join(output_dir, "result.json"), "r") as f:
    #     interaction_data = json.load(f)
    with open(os.path.join(output_dir, "final_mapping.json"), "r") as f:
        final_mapping = json.load(f)
    with open(os.path.join(output_dir, "final_edge_dict.json"), "r") as f:
        final_edge_dict = json.load(f)
    with open(os.path.join(output_dir,"scm_simple.json"), "r") as f:
        scm_simple = json.load(f)
    
    data_analyst = DataAnalyst(
            data,
            meta_data,
            final_mapping,
            final_edge_dict,
            interaction_data,
            scm_simple,
            # scenario,
        )
    data_analyst.analyze_data(
            output_dir,
            final_output_dir=output_dir,
            interaction=False,
            std_estimates=False,
            )
    typer.echo(f"The analysis is stored in  {output_dir}.")

def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "--v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    return
