import sys
import json
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import pandas as pd
import numpy as np
import math
from collections import Counter

sys.path.append("./src/LLM")
sys.path.append("./src/Serialization")
sys.path.append("./src/Question")

from LLM import LanguageModel, llm_json_loader, LLMMixin
from Serialize import RegisteredSerializable
from Variable import (
    Variable,
    retry_on_keyerror_decorator,
    ExogenousVariable,
    EndogenousVariable,
)
from Prompting import PromptMixin
from StructuralCausalModelBuilder import StructuralCausalModelBuilder


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class DataParser(PromptMixin, LLMMixin, RegisteredSerializable):
    """
    This class is responsible for cleaning/parsing the data from the interaction data.
    """

    def __init__(
        self,
        scm_interaction_data: Dict[str, Dict],
        template_dir: str = "prompt_templates",
    ) -> None:
        self.template_dir = template_dir

        # Initialize the Structural Causal Model (SCM)
        self._initialize_scm(scm_interaction_data)

        # the conversations between the agents, interactions, and survey data
        self.interaction_data = scm_interaction_data["data"]
        # mapping of which variables were exogenously manipulated for the given simualtion in what way
        self.attribute_value_mapping = scm_interaction_data["attribute_value_mapping"]
        self.scenario_description = self.scm.scenario_description
        self.agent_list = self.scm.agents_in_scenario
        self.variables = self.scm.variables
        # data from for analysis
        self.data_frame = pd.DataFrame(columns=self.variables)

        self.meta_data = {}

        # keys are variable names, values are dictionaries with keys as level names and values as numerical mappings.
        self.level_value_dict = self._populate_level_value_dict()
        # keys are variable names, values are dictionaries with keys as attribute variations and values as level names.
        self.level_variation_dict = self._populate_level_variation_dict()

        self.explanations_dict = {}

    def _initialize_scm(self, scm_interaction_data: Dict[str, Dict]):
        """
        Initialize the SCM based on the provided interaction data.

        Args:
            scm_interaction_data (Dict[str, Dict]): Data related to SCM interactions.
        """
        scm_data = scm_interaction_data["scm"]
        serialized_scm = (
            json.dumps(scm_data) if isinstance(scm_data, dict) else scm_data
        )
        self.scm = StructuralCausalModelBuilder.deserialize(serialized_scm)

    def __repr__(self) -> str:
        return f"""Scenario: {self.scenario_description}\n
                    Agents: {self.agent_list}\n
                    Variables: {self.variables} \n
                    Data Frame: {self.data_frame}"""

    def _populate_level_value_dict(self) -> None:
        """
        This function populates the level value dictionary.
        Keys are variable names, values are dictionaries with keys as level names and values as numerical mappings.

        Args:
            None
        """
        level_value_dict = {}
        for var_name in self.scm.variables:
            variable = self.scm.variable_dict[var_name]
            level_values = [i + 1 for i in range(len(variable.levels))]
            # if binary make 1, 0
            if variable.variable_type == "binary":
                level_values = [0, 1]
            if (
                variable.variable_type == "continuous"
                or variable.variable_type == "count"
            ):
                level_value_dict[variable.name] = {}
            else:
                # nominal, ordinal, binary
                level_value_dict[variable.name] = dict(
                    zip(variable.levels, level_values)
                )

        return level_value_dict

    def _populate_level_variation_dict(self) -> None:
        """
        This function populates the level variation dictionary.
        Keys are variable names, values are dictionaries with keys as attribute variations and values as level names.

        Args:
            None
        """
        level_variation_dict = {}
        for var_name in self.scm.variables:
            variable = self.scm.variable_dict[var_name]

            if isinstance(variable, ExogenousVariable):
                level_variation_dict[variable.name] = dict(
                    zip(
                        variable.attribute_variation["attribute_values"],
                        variable.levels,
                    )
                )

            # else eogenous variable
            else:
                level_variation_dict[variable.name] = {}

        return level_variation_dict

    def _check_multiple_question_per_measure(self, input_dict: Dict) -> bool:
        """
        This function checks if a dictionary has more than one key or if any of the keys have more than one value.

        Args:
            input_dict: a dictionary

        Returns:
            True if the dictionary has more than one key or if any of the keys have more than one value.
            False otherwise.
        """
        # Check if dictionary has more than one key
        if len(input_dict) > 1:
            return True

        # Check if any key has more than one value
        for key, value in input_dict.items():
            if isinstance(value, list) and len(value) > 1:
                return True
            elif isinstance(value, set) and len(value) > 1:
                return True
            elif isinstance(value, dict) and len(value) > 1:
                return True

        return False

    def parse_single_question_per_measure(
        self, var_name: str, survey: Dict
    ) -> Union[float, int]:
        """
        This function parses the data from a survey question that doesn't need aggregation and returns a single value for the variable.
        """
        variable = self.scm.variable_dict[var_name]
        for agent in survey[var_name].keys():
            for question, answer_str in survey[var_name][agent].items():
                answer_dict = json.loads(answer_str)
                answer = answer_dict["answer"]
                data_single = self.parse_single_question(
                    variable, question, answer, agent
                )

        return data_single

    def exogenous_data_parse(
        self, var_name: str, interaction_num: str
    ) -> Union[float, int]:
        """
        This function parses the exogenous data from the interaction data.
        """
        variable = self.scm.variable_dict[var_name]
        raw_data = self.attribute_value_mapping[interaction_num][var_name]
        if variable.variable_type == "continuous":
            data_single = float(re.findall(r"\d+", raw_data)[0])
        elif variable.variable_type == "count":
            data_single = int(re.findall(r"\d+", raw_data)[0])
        elif variable.variable_type in ("nominal", "binary", "ordinal"):
            level_match = self.level_variation_dict[var_name][raw_data]
            data_single = self.level_value_dict[var_name][level_match]
        else:
            raise ValueError(f"Unknown variable type: {variable.variable_type}")
        print(f"{variable.name}: ", data_single)
        return data_single

    def aggregate_multiple_data(self, var_name: str, survey: Dict) -> Union[float, int]:
        """
        This function aggregates data from multiple questions per variable.
        """
        variable = self.scm.variable_dict[var_name]
        raw_data = []
        for agent in survey[var_name].keys():
            # doesn't work for multiple questions per agent for the same variable
            for question, answer_str in survey[var_name][agent].items():
                answer_dict = json.loads(answer_str)
                data_single = self.parse_single_question(
                    variable, question, answer_dict["answer"], agent
                )
                raw_data.append(data_single)

        aggregation = self.get_aggregation_method(variable, raw_data)

        data_to_clean = [i for i in raw_data if not math.isnan(i)]

        return self.mechanistic_aggregation(data_to_clean, aggregation)

    @retry_on_keyerror_decorator
    def parse_single_question(
        self, variable: Variable, question: str, answer: str, agent: str
    ) -> Union[float, int]:
        """
        This function parses the data from a survey questions and returns a single value for the variable.
        """
        if agent == "oracle":
            agent = "the oracle, an additional agent who can read the transcript of the interaction and answer questions about the scenario"

        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": variable.name,
            "relevant_agents": self.agent_list,
            "agent": agent,
            "answer": answer,
            "question": question,
            # these two don't matter if the variable is continuous or count
            "levels": variable.levels,
            "level_values": list(self.level_value_dict[variable.name].values()),
        }

        prompt_template_dict = {
            "continuous": "get_continuous_data.txt",
            "binary": "get_binary_data.txt",
            "count": "get_count_data.txt",
            "ordinal": "get_ordinal_data.txt",
            "nominal": "get_nominal_data.txt",
        }

        prompt_template = prompt_template_dict[variable.variable_type]
        prompt = self.generate_prompt(
            prompt_template, template_dir=self.template_dir, **prompt_params
        )
        raw_output = self.call_llm(prompt)
        data_dict = llm_json_loader(raw_output)
        answer = data_dict["answer"]
        if answer == "na":
            return np.NaN

        # if there are any problems parsing the data, just return nan so that the code compiles
        try:
            return (
                float(answer) if variable.variable_type == "continuous" else int(answer)
            )
        except ValueError:
            return np.NaN

    @retry_on_keyerror_decorator
    def get_aggregation_method(self, variable: Variable, measurements: List) -> Dict:
        """
        This function gets the aggregation method for a variable.
        """

        print(variable.measurement_aggregation)
        prompt_params = {
            "scenario_description": self.scenario_description,
            "variable_name": variable.name,
            "variable_type": variable.variable_type,
            "measurements": measurements,
            "relevant_agents": self.agent_list,
            "level_value_dict": self.level_value_dict,
            "aggregation_method": variable.measurement_aggregation,
        }
        prompt = self.generate_prompt(
            "get_aggregation_method.txt",
            template_dir=self.template_dir,
            **prompt_params,
        )
        raw_output = self.call_llm(prompt)
        aggregation_dict = llm_json_loader(raw_output)
        aggregation = aggregation_dict["aggregation"]
        return aggregation

    def mechanistic_aggregation(
        self, data_list: List[Union[float, int]], aggregation: str
    ) -> Union[float, int]:
        """
        This function aggregates the data from a list of measurements.
        """
        if aggregation == "average":
            return self.average_data(data_list)
        elif aggregation == "sum":
            return self.sum_data(data_list)
        elif aggregation == "max":
            return self.max_data(data_list)
        elif aggregation == "min":
            return self.min_data(data_list)
        elif aggregation == "mode":
            return self.mode_data(data_list)
        else:
            raise ValueError(
                f"UNKNOWN AGGREGATION METHOD--PLEASE TRY AGAIN {aggregation}"
            )

    def average_data(self, data_list: List[Union[float, int]]) -> Union[float, int]:
        if not data_list:
            return 0.0  # or raise ValueError("data_list must not be empty")
        return sum(data_list) / len(data_list)

    def sum_data(self, data_list: List[Union[float, int]]) -> Union[float, int]:
        return sum(data_list)

    def max_data(self, data_list: List[Union[float, int]]) -> Union[float, int]:
        if not data_list:
            return float("-inf")  # or raise ValueError("data_list must not be empty")
        return max(data_list)

    def min_data(self, data_list: List[Union[float, int]]) -> Union[float, int]:
        if not data_list:
            return float("inf")  # or raise ValueError("data_list must not be empty")
        return min(data_list)

    def mode_data(self, data_list: List[Union[float, int]]) -> Union[float, int]:
        if not data_list:
            return None  # or raise ValueError("data_list must not be empty")
        count = Counter(data_list)
        mode = max(count, key=count.get)
        return mode

    def process_interaction(self, interaction_num):
        """
        This function takes the interaction data and returns a dictionary for a single interaction with the data from the survey questions.
        Also adds the data to the data frame.

        Args:
            interaction_num (str): The interaction number to be indexed from the interaction data.
        """
        if not self.interaction_data[interaction_num]:
            return None

        survey = self.interaction_data[interaction_num]["survey"]
        single_observation = {}

        for var_name in self.attribute_value_mapping[interaction_num].keys():
            data_single = self.exogenous_data_parse(var_name, interaction_num)
            single_observation[var_name] = data_single

        for var_name in survey.keys():
            # if multiple questions per variable (still only one per agent!)
            if self._check_multiple_question_per_measure(survey[var_name]):
                data_single = self.aggregate_multiple_data(var_name, survey)
                single_observation[var_name] = data_single
            # only one question to be answered per variable
            else:
                single_observation[var_name] = self.parse_single_question_per_measure(
                    var_name, survey
                )

        self.data_frame.loc[len(self.data_frame)] = single_observation

        return single_observation

    def get_data_from_interactions(self):
        with ProcessPoolExecutor() as executor:
            # Submit tasks to the executor
            futures = {
                executor.submit(
                    self.process_interaction, interaction_num
                ): interaction_num
                for interaction_num in self.interaction_data.keys()
            }

            # Process results
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    self.data_frame.loc[len(self.data_frame)] = result

    def gather_meta_data(self) -> Dict:
        """
        This function gathers the meta data for the scenario.
        """
        self.meta_data["variables"] = {}
        self.meta_data["scm_structure"] = self.scm.edge_dict
        for var_name, variable in self.scm.variable_dict.items():
            self.meta_data["variables"][var_name] = {}
            self.meta_data["variables"][var_name][
                "variable_type"
            ] = variable.variable_type
            self.meta_data["variables"][var_name]["level_value_dict"] = (
                self.level_value_dict[var_name]
            )
            self.meta_data["variables"][var_name]["endo_or_exo"] = type(
                variable
            ).__name__

    def write_data(self, folder_path: str) -> None:
        """
        This function saves the data DataFrame and meta_data DataFrame to separate CSV files.
        """
        # Ensure the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the data DataFrame
        data_file_path = os.path.join(folder_path, "raw_data.csv")
        self.data_frame.to_csv(data_file_path, index=False)

        # Save the meta_data DataFrame
        meta_data_file_path = os.path.join(folder_path, "meta_data.json")
        # Specify the filename

        for key, value in self.meta_data.items():
            if isinstance(value, set):
                self.meta_data[key] = list(value)

        # Writing JSON data
        with open(meta_data_file_path, "w") as file:
            json.dump(self.meta_data, file, cls=SetEncoder)

    def backend_clean_data(self, path: str) -> None:
        """
        this function cleans all the data in the data frame
        """
        self.get_data_from_interactions()
        self.gather_meta_data()
        self.write_data(path)


if __name__ == "__main__":
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=0.1)
    directory_path = "/Users/benjaminmanning/Desktop/test/"
    file_name = "result.json"
    file_path = os.path.join(directory_path, file_name)
    with open(file_path, "r") as f:
        interaction_data = json.load(f)

    # data parser

    data_parser = DataParser(interaction_data)
    data_parser.add_LLM(LLM)
    print(data_parser.level_variation_dict)

    if False:
        print(data_analyst.level_value_dict)
        data_analyst.get_data_from_interactions()
        data_analyst.gather_meta_data()
        print(data_analyst.data_frame)
        print(data_analyst.meta_data)
        data_analyst.save_data(directory_path)
