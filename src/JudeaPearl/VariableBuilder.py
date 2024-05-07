import sys
from typing import Dict, List, Union, Optional, Any


sys.path.append("../LLM")

from LLM import LLMMixin, LanguageModel
from Variable import EndogenousVariable, ExogenousVariable


class NominalException(Exception):
    """Exception raised for errors in the creation of a nominal variable."""

    def __init__(
        self,
        message="Nominal variables are not supported. Please try a different variable that has some sort of order, whether it be continuous, count, binary, or ordinal",
    ):
        self.message = message
        super().__init__(self.message)


class EndogenousException(Exception):
    """Exception raised for errors in the creation of a nominal variable."""

    def __init__(
        self,
        message="One of the causes has been classified as an endogenous variable. The system does not currently support this. You may be able to fix the problem by just trying again at a non-zero temperature. Otherwise, try a different cause.",
    ):
        self.message = message
        super().__init__(self.message)


class FirstEndogenousVariableBuilder(LLMMixin):
    """
    This class is responsible for building First endogenous variables.

    Attributes:
        scenario_description (str): A string containing the scenario description.
        agents_in_scenario (list): A list of agents in the scenario.
        template_dir (str): directory where the templates are stored for the prompts
    """

    def __init__(
        self,
        variable_name: str,
        scenario_description: str,
        agents_in_scenario: List[str],
        template_dir: str = "prompt_templates",
    ) -> None:

        self.template_dir: str = template_dir
        self.variable_name: str = variable_name
        self.scenario_description: str = scenario_description
        self.agents_in_scenario: List[str] = agents_in_scenario

    def build_variable(self, num_causes: int) -> EndogenousVariable:
        """
        This method builds an endogenous variable.
        """
        variable = EndogenousVariable(
            self.variable_name,
            self.scenario_description,
            self.agents_in_scenario,
            template_dir=self.template_dir,
        )
        variable.add_LLM(self.LLM)
        variable.operationalize_variable()
        variable.classify_variable_type()
        if variable.variable_type == "nominal":
            raise NominalException()
        variable.get_variable_units()
        variable.create_levels()
        variable.create_measurement_questions()
        if num_causes > 0:
            variable.get_causes(num_causes)

        return variable


class CausalVariableBuilder(LLMMixin):
    """
    This class is responsible for building exogenous variable and checking if they might need to be endogenous variables.

    Attributes:
        scenario_description (str): A string containing the scenario description.
        agents_in_scenario (list): A list of agents in the scenario.
        outcomes_for_build (list): A list of outcomes for the variable.
        possible_covariates (list): A list of possible covariates for the variable.
        template_dir (str): directory where the templates are stored for the prompts
    """

    def __init__(
        self,
        variable_name: str,
        scenario_description: str,
        agents_in_scenario: List[str],
        outcomes_for_build: List[str],
        possible_covariates: Optional[List[str]] = None,
        variable_dict: Optional[Dict[str, Any]] = None,
        template_dir: str = "prompt_templates",
    ) -> None:

        self.template_dir: str = template_dir
        self.variable_name: str = variable_name
        self.scenario_description: str = scenario_description
        self.agents_in_scenario: List[str] = agents_in_scenario
        self.outcomes_for_build: List[str] = outcomes_for_build
        self.possible_covariates: Optional[List[str]] = possible_covariates

        if self.possible_covariates is not None and len(self.possible_covariates) > 0:
            self.variable_dict = {
                key: value
                for key, value in variable_dict.items()
                if key in self.possible_covariates
            }
        else:
            self.variable_dict = variable_dict

    def build_variable(self) -> ExogenousVariable:
        """
        This method builds an exogenous variable
        """
        variable = ExogenousVariable(
            self.variable_name,
            self.scenario_description,
            self.agents_in_scenario,
            self.outcomes_for_build,
            self.possible_covariates,
            template_dir=self.template_dir,
        )
        variable.add_LLM(self.LLM)
        variable.operationalize_variable()
        variable.classify_variable_type()
        if variable.variable_type == "nominal":
            raise NominalException()
        variable.get_variable_units()
        variable.create_levels()
        variable.scenario_or_agent_variation()

        if "individual" in variable.scenario_or_agent_var["variable_scope"]:
            variable.induce_variation_individual()

            # check if there are yet other causes and if if the current cause is continuous or count
            if self.possible_covariates and variable.variable_type in [
                "continuous",
                "count",
            ]:

                # check if any of the other causes are continuous or count
                other_count_cont_causes = [
                    cause
                    for cause in self.possible_covariates
                    # if self.variable_dict[cause].variable_type in ["continuous", "count"]
                    if cause != self.variable_name
                    and self.variable_dict[cause].variable_type
                    in ["continuous", "count"]
                ]
                if other_count_cont_causes:

                    # if they are only keep keys that are in other_count_cont_causes
                    variable_dict = {
                        key: value
                        for key, value in self.variable_dict.items()
                        if key in other_count_cont_causes
                    }

                    # align the variation of the current cause with the other causes if continuous or count and so is the current cause
                    variable.individual_variation_align(variable_dict)

            variable.public_or_private_variation()

        try:
            if "scenario" in variable.scenario_or_agent_var["variable_scope"]:
                variable.induce_variation_scenario()
        except AttributeError as e:
            print(f"Attribute error occurred: {e}")
            sys.exit(1)

        return variable


if __name__ == "__main__":
    LLM = LanguageModel(family="openai", model="gpt-4", max_tokens=500, temperature=0.3)
    scenario_dict = {
        "a person getting cognitive behavioral therapy": ["therapist", "patient"],
        "an auction for a single contract with many bidders": [
            "auctioneer",
            "bidder 1",
            "bidder 2",
            "bidder 3",
            "bidder 4",
            "bidder 5",
            "contract owner",
        ],
        "speed dating with many people": [
            "speed dater 1",
            "speed dater 2",
            "speed dater 3",
            "speed dater 4",
            "speed dater 5",
            "speed dater 6",
            "speed dater 7",
            "speed dater 8",
            "speed dater 9",
            "speed dater 10",
            "event organizer",
        ],
        "a person is interviewing for a job with an employer": [
            "interviewer",
            "job candidate",
        ],
        "a judge is setting bail for a criminal defendant": [
            "judge",
            "defendant",
            "prosecutor",
            "defense attorney",
        ],
        "a policeman arresting a thief": ["policeman", "thief"],
        "a family is deciding where to go on vacation": [
            "mother",
            "father",
            "child 1",
            "child 2",
        ],
        "two people bargaining over a mug": ["buyer", "seller"],
        # "the ultimatum game": ['proposer', 'responder'],
        # 'the iterated prisoner dilemma in game theory': ['player 1', 'player 2'],
        # "the dictator game": ['allocator', 'recipient']
    }
    scenario_description = "A person who cares a lot about fairness at a store buying a snow shovel after a snowstorm"
    agents_in_scenario = ["store employee", "customer"]

    if True:
        first_endo = FirstEndogenousVariableBuilder(
            "whether the person buys the snow shovel",
            scenario_description,
            agents_in_scenario,
        )
        first_endo.add_LLM(LLM)
        first_endo_var = first_endo.build_variable(num_causes=2)
        print("############\n")
        print(first_endo_var)
        causal_var = CausalVariableBuilder(
            first_endo_var.causes[0],
            scenario_description,
            agents_in_scenario,
            first_endo_var.name,
            first_endo_var.possible_covariates,
        )
        causal_var.add_LLM(LLM)
        causal_var = causal_var.build_variable()
        print("############\n")
        print(causal_var)
        causal_var = CausalVariableBuilder(
            first_endo_var.causes[1],
            scenario_description,
            agents_in_scenario,
            first_endo_var.name,
            first_endo_var.possible_covariates,
        )
        causal_var.add_LLM(LLM)
        causal_var = causal_var.build_variable()
        print("############")
        print(causal_var)

    # scenario_description = "a person is interviewing for a job as a waiter"
    # agents_in_scenario = ["interviewer", "job candidate"]

    if False:
        first_endo = FirstEndogenousVariableBuilder(
            "whether the person gets the job", scenario_description, agents_in_scenario
        )
        first_endo.add_LLM(LLM)
        first_endo_var = first_endo.build_variable(num_causes=2)
        print("############\n")
        print(first_endo_var)
        causal_var = CausalVariableBuilder(
            first_endo_var.causes[0],
            scenario_description,
            agents_in_scenario,
            first_endo_var.name,
            first_endo_var.possible_covariates,
        )
        causal_var.add_LLM(LLM)
        causal_var = causal_var.build_variable()
        print("############\n")
        print(causal_var)
        causal_var = CausalVariableBuilder(
            first_endo_var.causes[1],
            scenario_description,
            agents_in_scenario,
            first_endo_var.name,
            first_endo_var.possible_covariates,
        )
        causal_var.add_LLM(LLM)
        causal_var = causal_var.build_variable()
        print("############")
        print(causal_var)

    if False:
        causal_var = CausalVariableBuilder(
            "The buyer's budget",
            scenario_description,
            agents_in_scenario,
            "Wether or not a deal occurs",
        )
        causal_var.add_LLM(LLM)
        causal_var = causal_var.build_variable()
        print(causal_var)
