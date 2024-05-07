import sys
import json
from concurrent.futures import ThreadPoolExecutor
import logging
import random
import os
from typing import List, Tuple, Dict, Union, Optional, Any
sys.path.append('../LLM')
sys.path.append('../Serialization')
sys.path.append('../Question')

from LLM import LanguageModel, llm_json_loader
from Serialize import RegisteredSerializable
from Variable import ExogenousVariable, EndogenousVariable, Variable, retry_on_keyerror_decorator
from Prompting import PromptMixin
from StructuralCausalModelBuilder import StructuralCausalModelBuilder

class NotAddedError(Exception):
    '''
    Raised when an expected attribute or element has not been added.
    '''
    pass

class AgentBuilder(PromptMixin, RegisteredSerializable):
    def __init__(self, template_dir: str ="prompt_templates"):
        '''
        Initializes an AgentBuilder instance.

        Args:
            scm (StructuralCausalModel): the structural causal model to be added.
            template_dir (str, optional): the directory where the prompt templates are stored. Defaults to "prompt_templates".
        '''
        self.template_dir = template_dir
        # Slightly hacky fix for deserializing because of order of deserialize calls.
        # I actually spent a fair amount of time trying to fix this, and there isn't an immediately obvious solution that doesn't involve
        # potentially other problems (setting attributes to none or depth first search, but this could lead to recursion errors that seem possible
        # with classes referencing each other)

        self.scm: Optional[Any] = None
        self.scenario_description: str = ''
         # list of agents in the scenario for easy access
        self.agent_list: List[str] = []
        
        # Dictionary and list for managing agents, keys are agent roles, values are dictionaries of attributes
        self.agent_dict: Dict[str, Dict[str, Any]] = {}
        
        # Dictionaries for managing necessary agent information and attributes
        # keys are agent roles, values are lists of necessary information
        self.necessary_agent_info: Dict[str, List[Any]] = {}
        # keys are agent roles, values are dictionaries of attributes (from the necessary information)
        self.necessary_agent_attributes: Dict[str, Dict[str, Any]] = {}
        
        # Managing varied attributes.
        # keys are agent roles, values are dictionaries of varied attributes (only)
        self.varied_attributes_dict: Dict[str, Dict[str, Any]] = {}

        # list of attributes that will be varied for any agent in the SCM (experimnental) (easy access)
        self.varied_attributes: List[str] = []
        
        # Dictionary for interaction types and order.
        # key's are interaction_type, speaker_order
        self.interaction_type: Dict[str, Any]= {}

        # keys are order, central agent, either or both could be blank
        self.order_dict: Dict[str, Any] = {}
        
        # Dictionary for explanations.
        self.explanations_dict: Dict[str, str] = {}
    

    def add_scm(self, scm: StructuralCausalModelBuilder) -> None:
        '''
        Adds a structural causal model to the agent builder.

        Args:
            scm (StructuralCausalModel): the structural causal model to be added.
        '''
        self.scm = scm
        self.scenario_description = scm.scenario_description
        self.agent_list = scm.agents_in_scenario
        
        # Initialize agent-specific attributes.
        for agent in self.agent_list:
            self._initialize_agent(agent)
            
        # Populate varied attributes.
        for variable in self.scm.variable_dict.values():
            self._populate_varied_attributes(variable)

    def _initialize_agent(self, agent: str) -> None:
        '''
        Initialize agent-specific data structures.

        Args:
            agent (str): the agent to initialize.
        '''
        self.agent_dict[agent] = {}
        self.necessary_agent_info[agent] = []
        self.necessary_agent_attributes[agent] = {}

    def _populate_varied_attributes(self, variable: str) -> None:
        '''
        Populate varied attributes for the specified agent.

        Args:
            agent (str): the agent for which to populate attributes.
            variable (ExogenousVariable): the variable containing attribute information.
        '''
        if not isinstance(variable, ExogenousVariable):
            return
        
        self.varied_attributes_dict[variable.name] = {}
        
        # first do individual variables
        for agent in self.agent_list:
            self.varied_attributes_dict[variable.name][agent] = {}

            #for individual
            if variable.attribute_variation['varied_agent'] == agent:
                attr_name = variable.attribute_variation['attribute_name']
                #blank for kehang to populate
                self.agent_dict[agent][attr_name] = ''
                # adding to list for prompt generation
                self.varied_attributes.append(attr_name)
                self.varied_attributes_dict[variable.name][agent][attr_name] = variable.attribute_variation["attribute_values"]

            #if individual and public, but don't add to the agent who it's relevant too (already added above!)
            if variable.scenario_or_agent_var['variable_scope'] == 'individual':
                if variable.public_or_private_var['choice'] == 'public' and variable.attribute_variation['varied_agent'] != agent:
                    public_name = variable.public_or_private_var['public_name']
                    #blank for kehang to populate
                    self.agent_dict[agent][public_name] = ''
                    # adding to list for prompt generation
                    self.varied_attributes.append(public_name)
                    public_variation = variable.public_or_private_var['public_values']
                    self.varied_attributes_dict[variable.name][agent][public_name] = public_variation 

            # for scenario
            if variable.scenario_or_agent_var['variable_scope'] == 'scenario':
                attr_name = variable.attribute_variation['attribute_name']
                #blank for kehang to populate
                self.agent_dict[agent][attr_name] = ''
                # adding to list for prompt generation
                self.varied_attributes.append(attr_name)
                self.varied_attributes_dict[variable.name][agent][attr_name] = variable.attribute_variation["attribute_values"]

    
    def call_llm(self):
        raise NotImplementedError("This method gets implemented when you add an LLM")

    def add_LLM(self, LLM: LanguageModel) -> None:
        self.LLM = LLM
        self.call_llm = self.LLM.call_llm

    def __repr__(self):
        def format_dict(d, indent=0):
            lines = []
            for key, value in d.items():
                if isinstance(value, dict):
                    value_str = format_dict(value, indent + 4)  # 4 spaces for indentation
                else:
                    value_str = str(value)
                lines.append(f"{' ' * indent}{key}: {value_str}")
            return '\n'.join(lines)

        formatted_agent_dict = format_dict(self.agent_dict, 4)  # 4 spaces for initial indentation
        return f"Scenario: {self.scenario_description} \n Agents: \n{formatted_agent_dict}"
    
    def agent_dict_to_json(self) -> str:
        return json.dumps(self.agent_dict)

    def add_role_to_attributes(self) -> None:
        '''
        This function adds the role description to the agent attributes
        '''
        for agent in self.agent_list:
            # making the role the 1st attribute of the agent
            self.agent_dict[agent] = {'your role is': agent, **self.agent_dict[agent]}
    

############################### functions to consider adding MAYBE TODO's #########################################

    # a few functions that I might implelment later
    # ALL THESE TODO's are MAYBE TODOs, the don't need to be done but they could be useful

    def scenario_info_to_json(self):
        #TODO information that would be useful to the agent just to start a social interaction
        pass

    def unncessary_information_remover(self, agent):
        #TODO remove information that is not necessary for the agent to have from the get necessary information function
        # it often gets too much information
        pass

    def distribution_generator(self, distribution, **params):

        # TODO Make this function so that an LLM could "call" it with certain parameters to run
        # partially finished
        if distribution == 'normal':
            mean = 0
            std_dev = 1
            number = random.normalvariate(mean, std_dev)
        if distribution == "poisson":
            lambda_value = 5
            number = random.poisson(lambda_value)
        if distribution == "binomial":
            n = 10  # number of trials
            p = 0.5  # probability of success
            number = random.binomial(n, p)

        return number

############################### functions with prompts #########################################

    @retry_on_keyerror_decorator
    def REP_prompt(self, prompt, response):
        '''
        This function takes a prompt and response and asks the LLM to check their response

        Args:
            prompt (str): the prompt to be sent to the LLM
            response (str): the response from the LLM
        '''
        prompt_params = {
                    "previous_response": response,
                    "previous_prompt": prompt
                    }
        prompt = self.generate_prompt("REP_prompt.txt", template_dir = self.template_dir, **prompt_params)
        raw_output = self.call_llm(prompt)
        return llm_json_loader(raw_output)
    


    def apply_fuction_to_all_agents(self, function: Any) -> None:
        '''
        This function applies a function to all the agents in the scenario

        Args:
            function (function): the function to be applied to all the agents
        '''
        for agent in self.agent_list:
            function(agent)
    
    def get_name(self) -> None:
        '''
        This function gets the name of the agent from the LLM

        Args:
            agent (str): the agent whose name is being found
        '''
        prompt_params = { 
            "scenario_description": self.scenario_description,
            "agent_roles": self.agent_list
        }
        prompt = self.generate_prompt("get_name.txt", template_dir = self.template_dir, **prompt_params)
        raw_output = self.call_llm(prompt)
        name_dict = llm_json_loader(raw_output)
        names = name_dict['names']
        for name, agent in zip(names, self.agent_list):
            self.agent_dict[agent] = {'your name': name, **self.agent_dict[agent]}
        self.explanations_dict['name'] = name_dict['explanation']



    @retry_on_keyerror_decorator
    def get_agent_goals(self, agent: str) -> None:
        '''
        This function gets the goal of the agent within the context of the scenario and the other information the agent has

        Args:
            agent (str): the agent whose goal is being found
        '''
        prompt_params = {
                    "scenario_description": self.scenario_description,
                    "relevant_agents": self.agent_list,
                    "agent": agent
                    }
        prompt = self.generate_prompt("get_agent_goals.txt", template_dir = self.template_dir, **prompt_params)
        raw_output = self.call_llm(prompt)
        goal = llm_json_loader(raw_output)
        self.agent_dict[agent]['goal'] = goal['goal']
        self.explanations_dict['goal'] = goal['explanation']
        return goal
    
    @retry_on_keyerror_decorator
    def get_agent_constraints(self, agent: str) -> None:
        '''
        This function gets the constraints of the agent within the context of the scenario and the other information the agent has

        Args:
            agent (str): the agent whose constraints are being found    
        '''
        prompt_params = {
                    "scenario_description": self.scenario_description,
                    "relevant_agents": self.agent_list,
                    "agent": agent,
                    "goal": self.agent_dict[agent]['goal']
                    }
        prompt = self.generate_prompt("get_agent_constraints.txt", template_dir = self.template_dir, **prompt_params)
        raw_output = self.call_llm(prompt)
        constraint = llm_json_loader(raw_output)
        self.agent_dict[agent]['constraint'] = constraint['constraint']
        self.explanations_dict['constraint'] = constraint['explanation']
        return constraint
    

    def _get_agent_varied_attributes_dict(self, agent: str) -> Dict[str, Any]:
        '''
        This function gets the varied attributes for an agent

        Args:
            agent (str): the agent whose varied attributes are being found
        '''
        agent_varied_attributes = {}
        for var_name in self.varied_attributes_dict.keys():
            # only add the varied attributes that are relevant to the agent
            if self.varied_attributes_dict[var_name][agent]:
                attribute = list(self.varied_attributes_dict[var_name][agent].keys())[0]
                agent_varied_attributes[attribute] = self.varied_attributes_dict[var_name][agent][attribute]

        print(f'AGENT NAME: {agent} WITH THESE VARIED ATTRIBUTES: {agent_varied_attributes}')
        return agent_varied_attributes
    
    def _get_exo_endo_vars(self):
        endo_exo_vars = {}
        endo_exo_vars['ExogenousVariable'] = []
        endo_exo_vars['EndogenousVariable'] = []
        for var in self.scm.variable_dict.values():
            if isinstance(var, ExogenousVariable):
                endo_exo_vars['ExogenousVariable'].append(var.name)
            elif isinstance(var, EndogenousVariable):
                endo_exo_vars['EndogenousVariable'].append(var.name)
            else:
                raise ValueError(f'Unknown variable type: {type(var)}')
            
        return endo_exo_vars
    
    @retry_on_keyerror_decorator
    def get_necessary_info(self, agent: str) -> None:
        '''
        This function gets the information that the LLM deems necessary for the agent to have that the agent doesn't already have

        Args:
            agent (str): the agent whose information is being converted to attributes
        '''

        prompt_params = {
                    "scenario_description": self.scenario_description,
                    "relevant_agents": self.agent_list,
                    "agent": agent,
                    "goal": self.agent_dict[agent]['goal'],
                    "causes" : self._get_exo_endo_vars()['ExogenousVariable'],
                    "outcomes" : self._get_exo_endo_vars()['EndogenousVariable'],
                    "constraint" : self.agent_dict[agent]['constraint'],
                    "varied_attribute_dict": self.varied_attributes_dict,
                    "varied_attributes": [key for key in self.agent_dict[agent] if key in self.varied_attributes],
                    }
        
        prompt = self.generate_prompt("get_necessary_info.txt", template_dir = self.template_dir, **prompt_params)
        raw_output = self.call_llm(prompt)
        info = llm_json_loader(raw_output)
        self.necessary_agent_info[agent] = info['information']
        self.explanations_dict['get_necessary_info'] = info['explanation']

    @retry_on_keyerror_decorator
    def info_to_attributes(self, agent: str) -> None:
        '''
        This function takes the information the LLM has deemed necessary for the agent and converts it to attributes with values

        Args:
            agent (str): the agent whose information is being converted to attributes
        '''
        prompt_params = {
                    "scenario_description": self.scenario_description,
                    "relevant_agents": self.agent_list,
                    "num_agents": len(self.agent_list),
                    "causes" : self._get_exo_endo_vars()['ExogenousVariable'],
                    "outcomes" : self._get_exo_endo_vars()['EndogenousVariable'],
                    "agent": agent,
                    "goal": self.agent_dict[agent]['goal'],
                    "constraint": self.agent_dict[agent]['constraint'],
                    # varied_attrobutes still works becausecbh
                    "varied_attributes": [key for key in self.agent_dict[agent] if key in self.varied_attributes],
                    "varied_attribute_dict": self.varied_attributes_dict,
                    "necessary_info": self.necessary_agent_info[agent]
                    }
        
        prompt = self.generate_prompt("info_to_attributes.txt", template_dir = self.template_dir, **prompt_params)
        raw_output = self.call_llm(prompt)
        info_attributes = llm_json_loader(raw_output)
        #double check prompt
        info_attributes = self.REP_prompt(prompt, info_attributes)
        for index, attribute in enumerate(info_attributes['information']):
            self.agent_dict[agent][attribute] = info_attributes['values'][index]
            #saving them sepearately for ease of access
            self.necessary_agent_attributes[agent][attribute] = info_attributes['values'][index]
        self.explanations_dict['info_to_attributes'] = info_attributes['explanation']

    @retry_on_keyerror_decorator
    def check_info_mismatch(self) :
        '''
        This function checks if the information that the agent has is consistent with the information that the other agents have
        It does this quasi recursively comparing the first agent to the second, the first and second to the third, etc.
        '''

        # skip the first agent as the first reference
        shuffled_agent_list = self.agent_list.copy()
        shuffled_agent_list = random.sample(shuffled_agent_list, len(shuffled_agent_list))
        for index, agent in enumerate(shuffled_agent_list[1:], start=1 ):
            prompt_params = {
                "scenario_description": self.scenario_description,
                "relevant_agents": self.agent_list,
                "agent": agent,
                "goal": self.agent_dict[agent]['goal'],
                "constraint": self.agent_dict[agent]['constraint'],
                "causes" : self._get_exo_endo_vars()['ExogenousVariable'],
                "outcomes" : self._get_exo_endo_vars()['EndogenousVariable'],
                "varied_attributes": [key for key in self.agent_dict[agent] if key in self.varied_attributes],
                "varied_attribute_dict": self.varied_attributes_dict,
                "info_attributes" : self.necessary_agent_attributes[agent],
                "consistent_info_attributes" : {key: self.necessary_agent_attributes[key] for key in self.agent_list[:index]},
                "attribute_names" : self.necessary_agent_attributes[agent].keys(),
                "attribute_values": self.necessary_agent_attributes[agent].values()
            }
            prompt = self.generate_prompt("check_info_mismatch.txt", template_dir = self.template_dir, **prompt_params)
            raw_output = self.call_llm(prompt)
            checked_attributes = llm_json_loader(raw_output)
            for index, attribute in enumerate(checked_attributes['attributes']):
                self.agent_dict[agent][attribute] = checked_attributes['values'][index]
                self.necessary_agent_attributes[agent][attribute] = checked_attributes['values'][index]
            self.explanations_dict['check_info_mismatch'] = checked_attributes['explanation']

    def add_more_attributes_based_one_others():
        # TODO if necessary, maybe not
        pass

    def remove_attributes_based_on_others():
        # TODO if necessary, maybe not
        pass

    
    def attribute_underscore_adder(self, agent: str, attribute: str) -> None:
        '''
        This function adds underscores to an attribute name

        Args:
            agent (str): the agent whose attribute is being underscored
            attribute (str): the attribute being underscored
        '''
        updated_attribute = "_" + attribute
        self.agent_dict[agent][updated_attribute] = self.agent_dict[agent][attribute]
        del self.agent_dict[agent][attribute]
    
    def build_agents(self) -> None:
        if self.scm is None:
             raise NotAddedError("No agents found in the provided SCM.")
        '''
        This function builds the agents for a scenario based on a structural causal model. And returns them
        '''
        self.apply_fuction_to_all_agents(self.get_agent_goals)
        self.apply_fuction_to_all_agents(self.get_agent_constraints)
        self.apply_fuction_to_all_agents(self.get_necessary_info)
        self.apply_fuction_to_all_agents(self.info_to_attributes)
        self.check_info_mismatch()
        #DOUBLE CHECK!! - DO IT IN A DIFFERENT ORDER
        self.check_info_mismatch()
        self.get_name()
        self.add_role_to_attributes()

        for agent in self.agent_dict.keys():
            self.attribute_underscore_adder(agent, 'goal')
            self.attribute_underscore_adder(agent, 'constraint')

    def backend_build_agents(self) -> dict:
        if self.scm is None:
             raise NotAddedError("No agents found in the provided SCM.")
        '''
        This function builds the agents for a scenario based on a structural causal model. And returns them
        '''
        #TODO - add scenario to attributes?? should I?
        self.apply_fuction_to_all_agents(self.get_agent_goals)
        self.apply_fuction_to_all_agents(self.get_agent_constraints)
        self.apply_fuction_to_all_agents(self.get_necessary_info)
        self.apply_fuction_to_all_agents(self.info_to_attributes)
        print(self)
        self.check_info_mismatch()
        #DOUBLE CHECK!! - DO IT IN A DIFFERENT ORDER
        self.check_info_mismatch()
        self.get_name()
        self.add_role_to_attributes()

        for agent in self.agent_dict.keys():
            self.attribute_underscore_adder(agent, 'goal')
            self.attribute_underscore_adder(agent, 'constraint')

        return self.agent_dict
    
    def backend_build_agents_no_extra_attr(self) -> dict:
        if self.scm is None:
             raise NotAddedError("No agents found in the provided SCM.")
        '''
        This function builds the agents for a scenario based on a structural causal model. And returns them
        '''
        #TODO - add scenario to attributes?? should I?
        self.apply_fuction_to_all_agents(self.get_agent_goals)
        self.apply_fuction_to_all_agents(self.get_agent_constraints)
        self.get_name()
        self.add_role_to_attributes()

        for agent in self.agent_dict.keys():
            self.attribute_underscore_adder(agent, 'goal')
            self.attribute_underscore_adder(agent, 'constraint')

        return self.agent_dict
    
    def backend_get_interaction_info(self) -> Tuple :
        if self.scm is None:
             raise NotAddedError("No agents found in the provided SCM.")
        '''
        This function gets the interaction type and the order of the agents in the interaction from the LLM
        '''
        # Get the interaction type from the LLM
        self.get_interaction_type()
    
        # Map interaction types to their corresponding functions
        interaction_dispatcher = {
            'center random': self.get_center_agent_no_order,
            'center ordered': self.get_center_agent_ordered,
            'ordered': self.get_agent_order,
            # 'random', 'oracle after', 'oracle before' don't need to be mapped
        }

        # Call the respective function if interaction type is in the dispatcher, else return None
        func = interaction_dispatcher.get(self.interaction_type)
        if func:
            func()
            return self.interaction_type, self.order_dict
        else:
             # no order for this interaction type, just returning the agents list for kehang
             # 'random', 'oracle after', 'oracle before' don't need to be mapped
            # no order or central agent for this interaction type 
            self.order_dict['order'] = self.agent_list
            self.order_dict['central agent'] = ''
            return self.interaction_type, self.order_dict
        
    
    @retry_on_keyerror_decorator
    def get_interaction_type(self) -> None:
        '''
        This function gets the interaction type from the LLM
        '''
        prompt_params = {
                    "scenario_description": self.scenario_description,
                    "relevant_agents": self.agent_list
                    }
        prompt = self.generate_prompt("get_interaction_type.txt", template_dir = self.template_dir, **prompt_params)
        raw_output = self.call_llm(prompt)
        interaction_dict = llm_json_loader(raw_output)
        self.interaction_type = interaction_dict['interaction_type']
        self.explanations_dict['explanation'] = interaction_dict['explanation']

    @retry_on_keyerror_decorator
    def get_center_agent_no_order(self) -> None:
        '''
        This function gets the order of the agents in the interaction from the LLM if the interaction type is center random
        '''
        prompt_params = {
                    "scenario_description": self.scenario_description,
                    "relevant_agents": self.agent_list
                    }
        prompt = self.generate_prompt("get_center_agent_no_order.txt", template_dir = self.template_dir, **prompt_params)
        raw_output = self.call_llm(prompt)
        order_dict = llm_json_loader(raw_output)
        # no order for this interaction type, just returning the agents list for kehang
        self.order_dict['order'] = self.agent_list
        self.order_dict['central agent'] = order_dict['central agent']
        self.explanations_dict['explanation'] = order_dict['explanation']

    @retry_on_keyerror_decorator
    def get_center_agent_ordered(self) -> None:
        '''
        This function gets the order of the agents in the interaction from the LLM if the interaction type is center ordered
        '''
        prompt_params = {
                    "scenario_description": self.scenario_description,
                    "relevant_agents": self.agent_list
                    }
        prompt = self.generate_prompt("get_center_agent_ordered.txt", template_dir = self.template_dir, **prompt_params)
        raw_output = self.call_llm(prompt)
        order_dict = llm_json_loader(raw_output)
        self.order_dict['order'] = order_dict['order']
        self.order_dict['central agent'] = order_dict['central agent']
        self.explanations_dict['explanation'] = order_dict['explanation']


    @retry_on_keyerror_decorator
    def get_agent_order(self) -> None:
        '''
        This function gets the order of the agents in the interaction from the LLM if the interaction type is ordered
        '''
        prompt_params = {
                    "scenario_description": self.scenario_description,
                    "relevant_agents": self.agent_list
                    }
        prompt = self.generate_prompt("get_agent_order.txt", template_dir = self.template_dir, **prompt_params)
        raw_output = self.call_llm(prompt)
        order_dict = llm_json_loader(raw_output)
        self.order_dict['order'] = order_dict['order']
        # no central agent for this interaction type
        self.order_dict['central agent'] = []
        self.explanations_dict['explanation'] = order_dict['explanation']
    
    def backend_return_varied_attributes(self) -> dict:
        '''
        This function returns the varied attributes for each agent and their possible values
        '''
        return self.varied_attributes_dict
    
    def backend_add_attribute(self, agent: str, attribute: str, value: str) -> None:
        '''
        This function adds an attribute and value to an agent
        '''
        # THESE DON"T WORK FOR VARIED ATTRIBUTES
        # CAN ALSO CHANGE ATTRIBUTES
        self.agent_dict[agent][attribute] = value
        self.necessary_agent_attributes[agent][attribute] = value

    def backend_remove_attribute(self, agent: str, attribute: str) -> None:
        '''
        This function removes an attribute from an agent
        '''
        # THESE DON"T WORK FOR VARIED ATTRIBUTES
        del self.agent_dict[agent][attribute]
        del self.necessary_agent_attributes[agent][attribute]


    def backend_edit_variation_values_agent(self, variable_name: str, variation_list: List[Union[int,float]]) -> None:
        '''
        Edits the variation values of a variable

        Args:
            variable_name (str): name of the variable
            variation_values (dict): dictionary of variation values
        '''
        variable = self.scm.variable_dict[variable_name]
        
        if variable.variable_type in ('nominal', 'ordinal', 'binary'):
             raise ValueError(f'Cannot edit variation values for {variable.variable_type} variables')

        # else variable type is continuous or count
        self.scm.variable_dict[variable_name].attribute_variation['attribute_values'] = variation_list

        for agent in self.varied_attributes_dict[variable_name].keys():
            if not self.varied_attributes_dict[variable_name][agent]:
                continue
            else:
                for attribute in self.varied_attributes_dict[variable_name][agent].keys():
                    self.varied_attributes_dict[variable_name][agent][attribute] = variation_list
