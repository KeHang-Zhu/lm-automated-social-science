import sys
import json
import os
from typing import Dict, List, Union, Optional, Any
from enum import Enum

sys.path.append("../LLM")
sys.path.append("../Serialization")
sys.path.append("../Question")

from LLM import LLMMixin, llm_json_loader
from Serialize import RegisteredSerializable
from Prompting import PromptMixin


def retry_on_keyerror_decorator(func) -> None:
    """
    Decorator that catches KeyError exceptions raised by the wrapped function.
    If after 5 attempts the function still raises a KeyError, the decorator raises the KeyError.

    Args:
        func (Callable): The function to decorate.
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function that retries the decorated function upon KeyError.

        Args:
            *args: Variable length argument list to pass to the decorated function.
            **kwargs: Arbitrary keyword arguments to pass to the decorated function.
        """
        for _ in range(5):
            try:
                return func(*args, **kwargs)
            except KeyError as e:
                print(f"KeyError occurred: {e}. Retrying...")
        raise KeyError(f"Function {func.__name__} failed after 5 retries.")

    return wrapper


class Variable(PromptMixin, LLMMixin, RegisteredSerializable):
    """
    Class for a variable in a causal model

    Args:
        name (str): name of the variable
        scenario_description (str): description of the scenario
        agents_in_scenario (list): list of agents in the scenario
        template_dir (str): directory where the templates are stored for the prompts
    """

    def __init__(
        self,
        name: str,
        scenario_description: str,
        agents_in_scenario: List[str],
        descendant_outcomes: Optional[List[str]] = None,
        possible_covariates: Optional[List[str]] = None,
        template_dir: str = "prompt_templates",
    ):
        self.template_dir: str = template_dir
        self.name: str = name
        self.scenario_description: str = scenario_description
        self.agents_in_scenario: List[str] = agents_in_scenario
        # keys: operationalization, method_to_obtain_quantity
        self.operationalization_dict: Dict[str, str] = {}
        self.variable_type: List[str] = []
        # self.variable_enum_type: VariableType = None
        self.units: List[str] = []
        self.levels: List[str] = []
        # keys: agent names, values: measurement questions
        self.agent_measure_question_dict: Dict[str, str] = {}
        self.measurement_aggregation: List[str] = []
        # variables that are direct descendants of the variable
        self.descendant_outcomes: List[str] = (
            [descendant_outcomes]
            if isinstance(descendant_outcomes, str)
            else descendant_outcomes or []
        )
        # all variables in the model besides the descendants (and causes obviously)
        self.possible_covariates: List[str] = (
            [possible_covariates]
            if isinstance(possible_covariates, str)
            else possible_covariates or []
        )
        # removing possible overlap between descendants and covariates, also removes itself
        self.possible_covariates: List[str] = list(
            set(self.possible_covariates) - set(self.descendant_outcomes) - {self.name}
        )

        # explanations for all variables
        self.explanations_dict: Dict[str, str] = {}

    def __repr__(self):

        self_dict = self.__dict__.copy()
        del self_dict["explanations_dict"]

        attrs = "\n".join(
            f"{k}: {v}"
            for k, v in self_dict.items()
            if not callable(v) and type(v).__module__ == "builtins"
        )
        return f"{type(self).__name__}:\n{attrs}"

    def var_to_dict(self) -> Dict[str, str]:
        return {
            "__class__": self.__class__.__name__,
            **{
                k: v
                for k, v in self.__dict__.items()
                if not callable(v)
                and type(v).__module__ == "builtins"
                and k != "template_dir"
            },
        }

    def var_to_json(self, directory_path=None) -> None:
        translate_map = str.maketrans({"'": None, " ": "_"})
        filename = f"{self.name}.json".translate(translate_map)

        if directory_path:
            filename = os.path.join(directory_path, filename)

        with open(filename, "w") as json_file:
            json.dump(self.var_to_dict(), json_file)

    ############################### functions with prompts #########################################
    @retry_on_keyerror_decorator
    def REP_prompt(self, prompt, response, variation_check=None) -> None:
        """
        This function takes a prompt and response and asks the LLM to check their response

        Args:
            prompt (str): the prompt to be sent to the LLM
            response (str): the response from the LLM
        """
        prompt_file = "REP_prompt.txt" if not variation_check else "REP_variation.txt"
        prompt_params = {"previous_response": response, "previous_prompt": prompt}
        prompt = self.generate_prompt(
            prompt_file, template_dir=self.template_dir, **prompt_params
        )
        raw_output = self.call_llm(prompt)
        return llm_json_loader(raw_output)

    @retry_on_keyerror_decorator
    def classify_variable_type(self) -> None:
        """
        Classifies the type of variable as continuous, binary, ordinal, count, or nominal

        Args:
            None
        """
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "relevant_agents": self.agents_in_scenario,
            "operationalization": self.operationalization_dict,
        }
        prompt = self.generate_prompt(
            "classify_variable_type.txt",
            template_dir=self.template_dir,
            **prompt_params,
        )
        raw_output = self.call_llm(prompt)
        variable_type = llm_json_loader(raw_output)
        self.explanations_dict["variable_type"] = variable_type["explanation"]
        self.variable_type = variable_type["variable_type"]

    @retry_on_keyerror_decorator
    def get_variable_units(self) -> None:
        """
        Gets the units of the variable.

        Args:
            None
        """
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "relevant_agents": self.agents_in_scenario,
            "operationalization": self.operationalization_dict,
            "variable_type": self.variable_type,
        }
        prompt = self.generate_prompt(
            "get_variable_units.txt", template_dir=self.template_dir, **prompt_params
        )
        raw_output = self.call_llm(prompt)
        units = llm_json_loader(raw_output)
        self.explanations_dict["units"] = units["explanation"]
        self.units = units["units"]

    @retry_on_keyerror_decorator
    def create_levels(self, num_cont_lvls: int = 5) -> None:
        """
        Gets the levels of the variable

        Args:
            num_cont_lvls (int): number of levels for a continuous variable
        """
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "relevant_agents": self.agents_in_scenario,
            "operationalization": self.operationalization_dict,
            "variable_type": self.variable_type,
            "units": self.units,
            "num_cont_lvls": num_cont_lvls,
        }
        prompt = self.generate_prompt(
            "create_levels.txt", template_dir=self.template_dir, **prompt_params
        )
        raw_output = self.call_llm(prompt)
        levels = llm_json_loader(raw_output)
        self.explanations_dict["levels"] = levels["explanation"]
        self.levels = levels["levels"]

    @retry_on_keyerror_decorator
    def create_measurement_questions(self) -> None:
        """
        Creates measurement questions for the variable and matches them to the relevant agent
        Additionally, creates a method to aggregate the information from the agents.

        Args:
            None
        """
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "relevant_agents": self.agents_in_scenario,
            "operationalization": self.operationalization_dict,
            "variable_type": self.variable_type,
            "units": self.units,
            "levels": (
                self.levels
                if self.variable_type != "continuous"
                else "variable is continuous, levels are not applicable"
            ),
        }
        prompt = self.generate_prompt(
            "create_measurement_questions.txt",
            template_dir=self.template_dir,
            **prompt_params,
        )
        raw_output = self.call_llm(prompt)
        measurement_questions = llm_json_loader(raw_output)
        measurement_questions = self.REP_prompt(prompt, measurement_questions)
        self.agent_measure_question_dict = {
            key: measurement_questions[key]
            for key in measurement_questions
            if key not in ["aggregation", "explanation"]
        }
        self.explanations_dict["measurement_questions"] = measurement_questions[
            "explanation"
        ]
        self.measurement_aggregation = measurement_questions["aggregation"]


class EndogenousVariable(Variable):
    """
    This class is specifically for endogenous variables.

    Args:
        name (str): name of the variable
        scenario_description (str): description of the scenario
        agents_in_scenario (list): list of agents in the scenario
        causes (Union[List[str], str]): A cause or list of causes to be added.
        descendant_outcomes (Union[List[str], str]): A child outcome or list of child outcomes to be added.
        possible_covariates (Union[List[str], str]): A covariate or list of possible_covariates to be added.
        template_dir (str): directory where the templates are stored for the prompts
    """

    def __init__(
        self,
        name: str,
        scenario_description: str,
        agents_in_scenario: List[str],
        causes: Optional[Union[str, List[str]]] = None,
        descendant_outcomes: Optional[List[str]] = None,
        possible_covariates: Optional[List[str]] = None,
        template_dir: str = "prompt_templates",
    ) -> None:
        super().__init__(
            name,
            scenario_description,
            agents_in_scenario,
            descendant_outcomes,
            possible_covariates,
            template_dir,
        )
        self.causes: List[str] = [causes] if isinstance(causes, str) else causes or []

    @retry_on_keyerror_decorator
    def operationalize_variable(self) -> None:
        """
        Gets an operationalization for a given variable
        Currently operationalizing such that there are no likert scales allowed

        Args:
            None
        """
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "relevant_agents": self.agents_in_scenario,
        }

        prompt = self.generate_prompt(
            "operationalize_variable.txt",
            template_dir=self.template_dir,
            **prompt_params,
        )
        raw_output = self.call_llm(prompt)
        operationalization_dict = llm_json_loader(raw_output)
        operationalization_dict = self.REP_prompt(prompt, operationalization_dict)
        self.explanations_dict["operationalization_dict"] = operationalization_dict[
            "explanation"
        ]
        self.operationalization_dict["operationalization"] = operationalization_dict[
            "operationalization"
        ]
        self.operationalization_dict["method_to_obtain_quantity"] = (
            operationalization_dict["method_to_obtain_quantity"]
        )

    @retry_on_keyerror_decorator
    def get_causes(self, num_causes: int) -> None:
        """
        Gets the causes of the endogenous variable also adds them

        Args:
            num_causes (int): number of causes to get
        """
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "relevant_agents": self.agents_in_scenario,
            "operationalization": self.operationalization_dict,
            "num_causes": num_causes,
            "descendant_outcomes": (
                "There are no other outcomes yet"
                if not self.descendant_outcomes
                else self.descendant_outcomes
            ),
            "possible_covariates": (
                "There are no other variables yet"
                if not self.possible_covariates
                else self.possible_covariates
            ),
        }
        prompt = self.generate_prompt(
            "get_exogenous_causes.txt", template_dir=self.template_dir, **prompt_params
        )
        raw_output = self.call_llm(prompt)
        causes_dict = llm_json_loader(raw_output)
        self.explanations_dict["causes"] = causes_dict["explanation"]
        self.causes = causes_dict["causes"]

    def add_causes(self, causes: Union[List[str], str]) -> None:
        """
        Add causes to the variable. Just a little helper function.

        Args:
            causes (Union[List[str], str]): A cause or list of causes to be added.
        """
        causes = [causes] if isinstance(causes, str) else causes
        self.causes.extend(causes)

    def remove_causes(self, causes):

        causes = [causes] if isinstance(causes, str) else causes
        if isinstance(causes, list):
            for cause in causes:
                self.causes.remove(cause)
        else:
            self.causes.remove(causes)


class ExogenousVariable(Variable):
    """
    This class is specifically for exogenous variables.

    Args:
        name (str): name of the variable
        scenario_description (str): description of the scenario
        agents_in_scenario (list): list of agents in the scenario
        relevant_outcome (str): name of the relevant outcome
        possible_covariates (Union[List[str], str]): A covariate or list of possible_covariates to be added.
        template_dir (str): directory where the templates are stored for the prompts
    """

    def __init__(
        self,
        name: str,
        scenario_description: str,
        agents_in_scenario: List[str],
        descendant_outcomes: List[str],
        possible_covariates: Optional[List[str]] = None,
        template_dir: str = "prompt_templates",
    ) -> None:
        super().__init__(
            name,
            scenario_description,
            agents_in_scenario,
            descendant_outcomes,
            possible_covariates,
            template_dir,
        )
        # keys: variable_scope, relevant_entity
        self.scenario_or_agent_var: Dict[str, Any] = {}
        # keys: attribute_name, attribute_values, varied_agent
        self.attribute_variation: Dict[str, Any] = {}
        # keys:
        self.public_or_private_var: Dict[str, Any] = {}
        # always empty:
        self.causes: List[Any] = []
        #
        self.variation_mapping = {}

    @retry_on_keyerror_decorator
    def operationalize_variable(self) -> None:
        """
        Gets an operationalization for a given variable
        Currently operationalizing such that there are no likert scales allowed

        Args:
            None
        """
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "descendant_outcomes": (
                "There are no other outcomes yet"
                if not self.descendant_outcomes
                else self.descendant_outcomes
            ),
            "relevant_agents": self.agents_in_scenario,
        }
        prompt = self.generate_prompt(
            "operationalize_variable_cause.txt",
            template_dir=self.template_dir,
            **prompt_params,
        )
        raw_output = self.call_llm(prompt)
        operationalization_dict = llm_json_loader(raw_output)
        operationalization_dict = self.REP_prompt(prompt, operationalization_dict)
        self.explanations_dict["operationalization_dict"] = operationalization_dict[
            "explanation"
        ]
        self.operationalization_dict["operationalization"] = operationalization_dict[
            "operationalization"
        ]
        self.operationalization_dict["method_to_vary"] = operationalization_dict[
            "method_to_vary"
        ]

    @retry_on_keyerror_decorator
    def check_if_endogenous(self) -> str:
        """
        Checks if the variable is an endogenous variable determined during the scenario.
        Switches it to endogenous variable if it is.

        Args:
            None
        """
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "relevant_agents": self.agents_in_scenario,
            "operationalization": self.operationalization_dict["operationalization"],
            "variable_type": self.variable_type,
            "units": self.units,
            "levels": self.levels,
        }
        prompt = self.generate_prompt(
            "check_if_endogenous.txt", template_dir=self.template_dir, **prompt_params
        )
        raw_output = self.call_llm(prompt)
        when_outcome_determined = llm_json_loader(raw_output)

        return (
            "endogenous"
            if "during" in when_outcome_determined["when_determined"]
            else "exogenous"
        )

    def change_to_endogenous(self) -> EndogenousVariable:
        """
        Changes the variable to an endogenous variable

        Args:
            None
        """
        # Initialize EndogenousVariable instance with the same arguments
        variable = EndogenousVariable(
            self.name, self.scenario_description, self.agents_in_scenario
        )

        variable.__dict__.update(self.__dict__)

        # Copy over specific attributes from EndogenousVariable __init__
        for attr in ["causes", "descendant_outcomes", "possible_covariates"]:
            if not hasattr(self, attr):
                setattr(variable, attr, {} if attr == "causes" else [])

        return variable

    @retry_on_keyerror_decorator
    def scenario_or_agent_variation(self) -> None:
        """
        Determines if the variable is relative to a scenario or an agent

        Args:
            None
        """
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "relevant_agents": self.agents_in_scenario,
            # both operationalization and method to vary are needed for this prompt
            "operationalization": self.operationalization_dict,
            "variable_type": self.variable_type,
            "units": self.units,
        }
        prompt = self.generate_prompt(
            "scenario_or_agent_variation.txt",
            template_dir=self.template_dir,
            **prompt_params,
        )
        raw_output = self.call_llm(prompt)
        scenario_or_agent_var = llm_json_loader(raw_output)
        self.explanations_dict["scenario_or_agent_var"] = scenario_or_agent_var[
            "explanation"
        ]
        self.scenario_or_agent_var["variable_scope"] = scenario_or_agent_var[
            "variable_scope"
        ]
        self.scenario_or_agent_var["relevant_entity"] = scenario_or_agent_var[
            "relevant_entity"
        ]

    @retry_on_keyerror_decorator
    def induce_variation_scenario(self) -> None:
        """
        Determines how to induce variation in the variable if it's relative to a scenario
        Chooses attribute for induced variation

        Args:
            None
        """
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "relevant_agents": self.agents_in_scenario,
            # both operationalization and method to vary are needed for this prompt
            "operationalization": self.operationalization_dict,
            "variable_type": self.variable_type,
            "units": self.units,
            "levels": self.levels,
            "descendant_outcomes": self.descendant_outcomes,
            "covariates": (
                "no other variables"
                if not self.possible_covariates
                else self.possible_covariates
            ),
        }
        prompt = self.generate_prompt(
            "induce_variation_scenario.txt",
            template_dir=self.template_dir,
            **prompt_params,
        )
        raw_output = self.call_llm(prompt)
        attribute_variation = llm_json_loader(raw_output)
        attribute_variation = self.REP_prompt(prompt, attribute_variation)
        # if self.variable_type == "ordinal":
        # attribute_variation = self.REP_prompt(prompt, attribute_variation, True)
        self.explanations_dict["attribute_variation"] = attribute_variation[
            "explanation"
        ]
        self.attribute_variation["attribute_name"] = attribute_variation[
            "attribute_name"
        ]
        self.attribute_variation["attribute_values"] = attribute_variation[
            "attribute_values"
        ]
        # not generated in prompt because always the same, but want dictionary to be consistent
        self.attribute_variation["varied_agent"] = "scenario"

    @retry_on_keyerror_decorator
    def induce_variation_individual(self) -> None:
        """
        Determines how to induce variation in the variable if it's relative to a scenario
        Chooses attribute for induced variation

        Args:
            None
        """
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "relevant_agents": self.agents_in_scenario,
            # both operationalization and method to vary are needed for this prompt
            "operationalization": self.operationalization_dict,
            "variable_type": self.variable_type,
            "units": self.units,
            "levels": self.levels,
            "descendant_outcomes": self.descendant_outcomes,
            "covariates": self.possible_covariates,
        }
        prompt = self.generate_prompt(
            "induce_variation_individual.txt",
            template_dir=self.template_dir,
            **prompt_params,
        )
        raw_output = self.call_llm(prompt)
        attribute_variation = llm_json_loader(raw_output)
        attribute_variation = self.REP_prompt(prompt, attribute_variation)
        if self.variable_type == "ordinal":
            attribute_variation = self.REP_prompt(prompt, attribute_variation, True)
        self.explanations_dict["attribute_variation"] = attribute_variation[
            "explanation"
        ]
        self.attribute_variation["attribute_name"] = attribute_variation[
            "attribute_name"
        ]
        self.attribute_variation["attribute_values"] = attribute_variation[
            "attribute_values"
        ]
        # make all strings for json
        self.attribute_variation["attribute_values"] = [
            str(item) for item in self.attribute_variation["attribute_values"]
        ]
        self.attribute_variation["varied_agent"] = attribute_variation["varied_agent"]

    def fix_ordinal_numeric_variation(self) -> None:
        """
        Fixes the ordinal numeric variation to be in the correct order

        Args:
            None
        """
        self.attribute_variation["attribute_values"] = sorted(
            self.attribute_variation["attribute_values"]
        )

    @retry_on_keyerror_decorator
    def individual_variation_align(
        self, other_causes_dict: Dict[str, Variable]
    ) -> None:
        """
        If there is already at least one cause and the variable is count or continuous, checks to make sure that the variation is aligned with the cause
        For example, if the existing cause is "buyer's budget" and the budget variation is: [5, 10, 15, 20],
        and the current variable is "seller's valuation", then the seller's valuation variation should be overlapping with the buyer's budget variation.

        Args:
            None
        """
        varied_attributes = [
            cause.attribute_variation
            for cause in other_causes_dict.values()
            if not isinstance(cause, EndogenousVariable)
        ]

        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "relevant_agents": self.agents_in_scenario,
            "operationalization": self.operationalization_dict["operationalization"],
            "variable_type": self.variable_type,
            "units": self.units,
            "descendant_outcomes": self.descendant_outcomes,
            "varied_attribute": self.attribute_variation["attribute_name"],
            "varied_agent": self.attribute_variation["varied_agent"],
            "variation_number": len(self.attribute_variation["attribute_values"]),
            "attribute_values": self.attribute_variation["attribute_values"],
            "other_variables": list(other_causes_dict.keys()),
            "other_variable_info": varied_attributes,
        }
        prompt = self.generate_prompt(
            "individual_variation_align.txt",
            template_dir=self.template_dir,
            **prompt_params,
        )
        raw_output = self.call_llm(prompt)
        final_variation = llm_json_loader(raw_output)
        self.attribute_variation["attribute_values"] = final_variation[
            "attribute_values"
        ]
        self.explanations_dict["align_attribute_variation"] = final_variation[
            "explanation"
        ]

    @retry_on_keyerror_decorator
    def public_or_private_variation(self) -> None:
        """
        Determines if the variable is public or private if it is individual variation
        This is necessary to determine if the attributes for the agents in the simulation are to be determined jointly or one by one.
        """
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": self.name,
            "descendant_outcomes": self.descendant_outcomes,
            "relevant_agents": self.agents_in_scenario,
            "operationalization": self.operationalization_dict["operationalization"],
            "variable_type": self.variable_type,
            "units": self.units,
            "agent": self.attribute_variation["varied_agent"],
            "attribute_name": self.attribute_variation["attribute_name"],
            "attribute_values": self.attribute_variation["attribute_values"],
        }

        prompt = self.generate_prompt(
            "public_or_private_variation.txt",
            template_dir=self.template_dir,
            **prompt_params,
        )
        raw_output = self.call_llm(prompt)
        public_private_dict = llm_json_loader(raw_output)
        self.explanations_dict["public_or_private_var"] = public_private_dict[
            "explanation"
        ]
        self.public_or_private_var["choice"] = public_private_dict["choice"]
        self.public_or_private_var["public_name"] = public_private_dict["public_name"]
        if self.public_or_private_var["choice"] == "public":
            self.public_or_private_var["public_values"] = self.attribute_variation[
                "attribute_values"
            ]
            # make all strings for json
            self.public_or_private_var["public_values"] = [
                str(item) for item in self.public_or_private_var["public_values"]
            ]
        else:
            self.public_or_private_var["public_values"] = []


if __name__ == "__main__":
    print("\n ### NEW TEST ####\n")
    if False:
        exo = ExogenousVariable(
            "mood of buyer",
            "two people bargaining over a mug",
            ["buyer", "seller"],
            "final price of the mug",
        )
        endo = EndogenousVariable(
            "final price of the mug",
            "two people bargaining over a mug",
            ["buyer", "seller"],
        )
        print(endo.__dict__)
        print("\n\n")
        print(exo.__dict__)
        # LLM = LanguageModel(family = "openai", model = "gpt-4", temperature = 0.3)
        # description_agent_dict = {"two people bargaining over a mug": ['buyer', 'seller']}
        #  Variable('final price of the mug', "two people bargaining over a mug",  ['buyer', 'seller'])
        # var.add_LLM(LLM)
        # print(f"###### VARIABLE: {var.name} ######")
        # print('###### OPERATIONALIZATION: ######')
        # print(var.operationalize_variable()['operationalization'])

    if True:
        # LLM = LanguageModel(family = "openai", model = "gpt-4", temperature = 0.2)
        directory = "serial_test_vars/a judge is setting bail for a criminal defendant/"
        for filename in os.listdir(directory):
            print(filename)
            if filename.endswith("SER.json"):
                print("#############################")
                print(filename)
                with open(directory + filename, "r") as f:
                    data = json.load(f)
                print(type(data))

                if "EndogenousVariable" in data:
                    var = EndogenousVariable.deserialize(data)
                if "ExogenousVariable" in data:
                    var = ExogenousVariable.deserialize(data)
                print(var.lat_or_obs == "latent")
                if var.lat_or_obs == "latent":
                    print(var)
                    # var.add_LLM(LLM)
                    print(var.add_observed_proxies())

                    # else:
                    #     print(f'observed: {var.name}')

    if False:

        def replace_string_in_json(file, old_string, new_string):
            with open(file, "r") as f:
                data = json.load(f)

            data_as_str = json.dumps(data)
            data_as_str = data_as_str.replace(old_string, new_string)

            new_data = json.loads(data_as_str)

            with open(file, "w") as f:
                json.dump(new_data, f)

        directory = "serial_test_vars/outcomes/"
        old_string = "OutcomeVariable"  # The string to be replaced
        new_string = "EndogenousVariable"  # The new string

        for filename in os.listdir(directory):
            if filename.endswith(".json"):  # specify file type
                file = directory + filename.lower()
                replace_string_in_json(file, old_string, new_string)
