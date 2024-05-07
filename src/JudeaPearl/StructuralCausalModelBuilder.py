import sys
import matplotlib.pyplot as plt
import networkx as nx
import json
from typing import List, Dict, Union, Optional
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

sys.path.append("../LLM")
sys.path.append("../Serialization")
sys.path.append("../Question")

from LLM import LanguageModel, LLMMixin
from Variable import EndogenousVariable, ExogenousVariable
from Serialize import RegisteredSerializable
from Prompting import PromptMixin
from VariableBuilder import FirstEndogenousVariableBuilder, CausalVariableBuilder


class StructuralCausalModelBuilder(PromptMixin, LLMMixin, RegisteredSerializable):
    """
    Creates and sets up variables for a given scenario description

    Args:
        scenario_description (str): description of the scenario
        agents_in_scenario (list): list of agents in the scenario
        template_dir (str): directory where the templates are stored for the prompts
    """

    def __init__(
        self,
        scenario_description: str,
        agents_in_scenario: List[str],
        template_dir: str = "prompt_templates",
    ) -> None:
        self.template_dir: str = template_dir
        self.scenario_description: str = scenario_description
        self.agents_in_scenario: List[str] = agents_in_scenario
        # a list of variable names
        self.variables: List[str] = []
        # keys: parent, values: set of children
        self.edge_dict: Dict[str, str] = {}
        # keys: variable names, values: variable class objects
        self.variable_dict: Dict[str, Union[EndogenousVariable, ExogenousVariable]] = {}

    def _transform_edge_dict(self) -> Dict[str, List[str]]:
        """
        Helper function to transform the edge dictionary to a format without spaces or apostraphes
        """
        return {
            self._sanitize_key(key): [self._sanitize_key(item) for item in value]
            for key, value in self.edge_dict.items()
        }

    def _sanitize_key(self, key) -> str:
        """
        Helper function replace spaces with underscores and apostraphes with nothing
        """
        return key.replace(" ", "_").replace("'", "")

    def __repr__(self):
        attrs = "\n".join(
            f"{k}: {v}\n"
            for k, v in self.__dict__.items()
            if not callable(v) and type(v).__module__ == "builtins"
        )
        return f"{type(self).__name__}:\n{attrs}"

    def edge_dict_to_json(
        self,
        directory_path: Optional[str] = None,
        filename: str = "scm_edge_structure.json",
    ) -> None:
        """
        Writes the edge dictionary to a json file

        Args:
            directory_path (str, optional): directory to write the file to. Defaults to None.
            filename (str, optional): name of the file to write to. Defaults to "scm_edge_structure.json".
        """
        if directory_path:
            filename = os.path.join(directory_path, filename)

            transformed_edge_dict = self._transform_edge_dict()

        with open(filename, "w") as json_file:
            json.dump(transformed_edge_dict, json_file)

    def scm_to_json(
        self, directory_path: str = "serial_test_vars/", filename: Optional[str] = None
    ) -> None:
        """
        Writes the variable dictionary (variable names are keys and values are variable class objects) to a json file

        Args:
            directory_path (str, optional): directory to write the file to. Defaults to 'serial_test_vars/'.
        """
        # Use the variable's name as the filename if it's not specified
        if not filename:
            filename = self.scenario_description.replace(" ", "_").replace("'", "")
            filename = filename + ".json"

        var_data = {}

        variable_count = 1
        for var_name, var in self.variable_dict.items():

            try:
                assert var_name == var.name
            except AssertionError:
                logging.error(
                    f"Variable name {var_name} key does not match the name in the variable of the Variable Class object {var.name}\n updating to match the key"
                )
                var.name = var_name

            var_data[f"Variable{variable_count}"] = var.var_to_dict()

            variable_count += 1

        # Write data to JSON file
        with open(directory_path + filename, "w") as f:
            json.dump(var_data, f)

    def backend_scm_to_json(self) -> str:
        """
        Converts the variable dictionary (variable names are keys and values are variable class objects) to a json string

        Returns:
            str: JSON string representing the variable dictionary
        """

        var_data = {}

        variable_count = 1
        for var_name, var in self.variable_dict.items():

            try:
                assert var_name == var.name
            except AssertionError:
                logging.error(
                    f"Variable name {var_name} key does not match the name in the variable of the Variable Class object {var.name}\n updating to match the key"
                )
                var.name = var_name

            var_data[f"Variable{variable_count}"] = var.var_to_dict()

            variable_count += 1

        # Convert data to JSON string
        json_str = json.dumps(var_data)

        return json_str

    def draw_scm(self, directory_path="serial_test_vars/") -> None:
        """
        Draws the structural causal model and saves it to a png file

        Args:
            directory_path (str, optional): directory to write the file to. Defaults to 'serial_test_vars/'.
        """
        # Initialize a directed graph
        G = nx.DiGraph()
        for node in self.edge_dict:
            for edge in self.edge_dict[node]:
                G.add_edge(node, edge)

        circle_nodes_red = []
        circle_nodes_blue = []
        square_nodes_red = []
        square_nodes_blue = []

        for _, variable in self.variable_dict.items():
            if isinstance(variable, ExogenousVariable):
                if variable.lat_or_obs == "latent":
                    circle_nodes_red.append(variable.name)
                else:
                    square_nodes_red.append(variable.name)
            if isinstance(variable, EndogenousVariable):
                if variable.lat_or_obs == "latent":
                    circle_nodes_blue.append(variable.name)
                else:
                    square_nodes_blue.append(variable.name)

        pos = nx.spring_layout(G, seed=42)
        plt.title(
            f"THE SCENARIO: '{self.scenario_description}' \n WITH THESE AGENTS: '{self.agents_in_scenario}'"
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=circle_nodes_red,
            node_color="red",
            node_size=500,
            node_shape="o",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=circle_nodes_blue,
            node_color="skyblue",
            node_size=500,
            node_shape="o",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=square_nodes_red,
            node_color="red",
            node_size=500,
            node_shape="s",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=square_nodes_blue,
            node_color="skyblue",
            node_size=500,
            node_shape="s",
        )
        nx.draw_networkx_edges(G, pos, edge_color="k", arrowstyle="-|>", arrowsize=20)
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(directory_path, f"{self.scenario_description}_scm_graph.png")
        )
        plt.close()

    def build_first_endogenous(
        self, variable_name: str, num_causes: int
    ) -> EndogenousVariable:
        """
        Creates and sets up an outcome variable for a given scenario description - doesn't add causes.

        Args:
            name (str): name of the variable
            num_causes (int): number of causes for this outcome variable
        """
        variable_builder = FirstEndogenousVariableBuilder(
            variable_name,
            self.scenario_description,
            self.agents_in_scenario,
            template_dir=self.template_dir,
        )
        variable_builder.add_LLM(self.LLM)
        variable = variable_builder.build_variable(num_causes)

        # add the variable the list of variables
        self.variables.append(variable.name)

        # add edges, causes are parents to the outcome
        for cause in variable.causes:
            self.edge_dict.setdefault(cause, set()).add(variable.name)
            self.variables.append(cause)

        self.variable_dict[variable.name] = variable
        return variable

    def _get_descendants(self, edge_dict: Dict[str, str], variable: str) -> List[str]:
        """
        Helper function to get the descendants of a variable - all the variables that are children of the variable, children of those children, etc.

        Args:
            edge_dict (dict): dictionary of edges
            variable (str): name of the variable
        """
        descendants = []
        visited = set()

        def depth_first_search(node):
            for child in edge_dict.get(node, []):
                if child not in visited:
                    descendants.append(child)
                    visited.add(child)
                    depth_first_search(child)

        depth_first_search(variable)
        return descendants

    def recursively_build_causal_variable(
        self,
        variable_name: str,
        possible_covariates: List[str],
        num_recursive_causes: int = 1,
        starting_depth: int = 0,
        limit_recursion_depth: int = 2,
    ) -> None:
        """
        Creees and recusviely builds causal variables for a given scenario description

        Args:
            name (str): name of the variable
            possible_covariates (list): are all the variables in the edge_dict that are NOT descendants of the variable.
            num_recursive_causes (int): number of causes if the variable is endogenous
            starting_depth (int): depth of the recursion current
            limit_recursion_depth (int): limit of the recursion depth
        """

        descendant_outcomes = self._get_descendants(self.edge_dict, variable_name)
        variable_builder = CausalVariableBuilder(
            variable_name,
            self.scenario_description,
            self.agents_in_scenario,
            descendant_outcomes,
            possible_covariates,
            template_dir=self.template_dir,
        )
        variable_builder.add_LLM(self.LLM)
        # check if the variable is endogenous/exogenous and then build it
        variable_checked = variable_builder.build_variable()

        if isinstance(variable_checked, ExogenousVariable):
            self.variable_dict[variable_checked.name] = variable_checked

        if isinstance(variable_checked, EndogenousVariable):
            variable_checked.get_causes(num_causes=num_recursive_causes)
            self.variable_dict[variable_checked.name] = variable_checked

            # Doing two loops here to make sure that the causes are added to the edge dict before the variable itself is instantiated
            for cause in variable_checked.causes:
                self.edge_dict.setdefault(cause, set()).add(variable_checked.name)
                self.variables.append(cause)

            # Instantiate the variable if it is not already instantiated
            for cause in variable_checked.causes:
                if cause not in list(self.variable_dict.keys()):

                    # if the starting depth is greater than the limit, then limit the number of recursive causes to 1
                    if starting_depth >= limit_recursion_depth:
                        num_recursive_causes = 1

                    # recursive call to recursively_build_causal_variable
                    self.recursively_build_causal_variable(
                        cause,
                        possible_covariates,
                        num_recursive_causes,
                        starting_depth + 1,
                        limit_recursion_depth,
                    )

                else:

                    print(f"cause: {cause}")
                    print(f"variables: {self.variables}")
                    print("POTENTIAL INFINITE RECURSION OF CAUSES BECAUSE OF REPEAT")
                    sys.exit()

    def backend_build_first_endogenous(self, variable_name: str) -> EndogenousVariable:
        """
        Creates and sets up an outcome variable for a given scenario description - doesn't add causes.

        Args:
            name (str): name of the variable
            num_causes (int): number of causes for this outcome variable
        """
        variable_builder = FirstEndogenousVariableBuilder(
            variable_name,
            self.scenario_description,
            self.agents_in_scenario,
            template_dir=self.template_dir,
        )
        variable_builder.add_LLM(self.LLM)
        variable = variable_builder.build_variable(num_causes=0)
        # logging variable
        self.variables.append(variable.name)
        self.variable_dict[variable.name] = variable
        return variable

    def backend_build_causal_variable(
        self, variable_name: str, possible_covariates: List[str]
    ):
        """
        Creates and sets up an exogenous variable for a given scenario description.
        Possible covariates are all the variables in the edge_dict
        ONLY BUILD THE VARIABLE AFTER IT'S BEEN "ADDED" `backend_add_cause`

        Args:
            name (str): name of the variable
            possible_covariates (list): are all the variables in the edge_dict that are NOT descendants of the variable.
        """

        descendant_outcomes = self._get_descendants(self.edge_dict, variable_name)
        variable_builder = CausalVariableBuilder(
            variable_name,
            self.scenario_description,
            self.agents_in_scenario,
            descendant_outcomes,
            possible_covariates,
            self.variable_dict,
            template_dir=self.template_dir,
        )
        variable_builder.add_LLM(self.LLM)
        variable = variable_builder.build_variable()
        self.variable_dict[variable.name] = variable
        return variable

    def backend_get_causes_for_variable(
        self, variable_name: str, num_causes: int
    ) -> List[str]:
        """
        Gets the causes for a given variable. Does not log it in the graph nor in the causes of the variable object in the graph.
        This is because the user will decide which causes to add to the variable after seeing the potential causes.

        Args:
            name (str): name of the variable
            num_causes (int): number of causes for this outcome variable
        """
        already_added_causes = set(self.variable_dict[variable_name].causes)
        variable = self.variable_dict[variable_name]
        variable.get_causes(num_causes=num_causes)
        ##deleting when added automatically - part of functionality with recursive SCMs that's hard to delete and Variable base class
        new_causes = list(set(variable.causes) - already_added_causes)
        variable.remove_causes(new_causes)
        return new_causes

    def backend_add_cause(self, child_outcome_name: str, cause_name: str) -> None:
        """
        Adds cause to the variable, and the edge dict, and list of variables.
        Should be used after user choose the causes they want from the potential causes.

        Args:
            child_outcome_name (str): name of the outcome variable
            cause_name (str): name of the cause variable
        """
        self.edge_dict.setdefault(cause_name, set()).add(child_outcome_name)
        self.variables.append(cause_name)
        self.variable_dict[child_outcome_name].add_causes(cause_name)

    def backend_remove_cause(self, child_outcome_name: str, cause_name: str) -> None:
        """npm
        Removes cause from the variable, and the edge dict, and list of variables.
        Usually don't need unless the user goes back later since variables are not added at first.

        Args:
            child_outcome_name (str): name of the outcome variable
            cause_name (str): name of the cause variable
        """
        # delete arrow from edge dict
        self.edge_dict[cause_name].remove(child_outcome_name)
        # if that causes has no other children, delete it from the edge dict (almost always true)
        if not self.edge_dict[cause_name]:
            del self.edge_dict[cause_name]
        # delete the cause from the list of variables
        self.variables.remove(cause_name)
        # delete the cause from the outcome variable's causes
        self.variable_dict[child_outcome_name].remove_causes(cause_name)
        # delete the cause from the variable dictionary
        del self.variable_dict[cause_name]

    def backend_edit_variation_values(
        self, variable_name: str, variation_list: List[Union[int, float]]
    ) -> None:
        """
        Edits the variation values of a variable

        Args:
            variable_name (str): name of the variable
            variation_values (dict): dictionary of variation values
        """
        variable = self.variable_dict[variable_name]

        if variable.variable_type in ("nominal", "ordinal", "binary"):
            raise ValueError(
                f"Cannot edit variation values for {variable.variable_type} variables"
            )

        # else variable type is continuous or count
        self.variable_dict[variable_name].attribute_variation[
            "attribute_values"
        ] = variation_list


if __name__ == "__main__":
    if True:
        # NOTE PREDETERMINED CAUSES
        LLM = LanguageModel(family="openai", model="gpt-4", temperature=0.01)
        scenario_description = "A person who cares a lot about fairness at a store buying a snow shovel after a snowstorm"
        agents_in_scenario = ["store employee", "customer"]
        outcome_variable = "whether the person buys the snow shovel"
        # backend_get_causes_for_variable
        scm = StructuralCausalModelBuilder(scenario_description, agents_in_scenario)
        scm.add_LLM(LLM)
        endo_var = scm.backend_build_first_endogenous(outcome_variable)
        print("####1#######")
        print("####2#######")
        scm.backend_add_cause(
            endo_var.name, "customer's previous experience with snow shovels"
        )
        print("####3#######")
        print("####4#######")
        # list of exogenous variables that are not the same as the currect cause nor the outcome
        possible_covariates = [
            var
            for var in scm.variables
            if var
            not in [endo_var.name, "customer's previous experience with snow shovels"]
        ]
        scm.backend_build_causal_variable(
            "customer's previous experience with snow shovels", possible_covariates
        )
        # print("####5#######")
        # scm.backend_add_cause(endo_var.name, "store employee's sales skills")
        # possible_covariates = [
        #     var
        #     for var in scm.variables
        #     if var not in [endo_var.name, "store employee's sales skills"]
        # ]
        # scm.backend_build_causal_variable(
        #     "store employee's sales skills", possible_covariates
        # )
        # print("####6#######")
        # cause_3 = "store employee's mood"
        # scm.backend_add_cause(endo_var.name, cause_3)
        # possible_covariates = [
        #     var for var in scm.variables if var not in [endo_var.name, cause_3]
        # ]
        # scm.backend_build_causal_variable(cause_3, possible_covariates)
        # print(scm)
        # scm.scm_to_json('/Users/benjaminmanning/Desktop/')
        scm_serialized = scm.serialize()
        with open(
            f"/Users/benjaminmanning/Desktop/{scenario_description}.json", "w"
        ) as f:
            json.dump(scm_serialized, f)
