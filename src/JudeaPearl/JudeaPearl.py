import sys
from typing import Dict, List, Union

sys.path.append('../LLM')
sys.path.append('../Serialization')
sys.path.append('../Question')

from LLM import LanguageModel, LLMMixin, llm_json_loader
from Serialize import RegisteredSerializable
from Prompting import PromptMixin

class JudeaPearl(PromptMixin, LLMMixin, RegisteredSerializable):
    def __init__(self, scenario_description: str, template_dir: str = 'prompt_templates') -> None:
        '''Comes up with a DAG for a given scenario description'''
        self.template_dir: str = template_dir
        self.scenario_description: str = scenario_description
        self.outcomes_dict: Dict[str, str] = {}
        self.outcomes: List[str] = []
        # keys: agents, explanation
        self.agents_dict: Dict[str, str] = {}
        self.agents: List[str] = []

    def __repr__(self):
        string =  f"""Studying these outcomes: {self.outcomes},
        with these agents: {self.agents},
        for this scenario: {self.scenario_description}"""
        return string

    ############################### functions with prompts #########################################

    def get_human_agents(self) -> List[str]:
        '''
        Gets a list of human agents for a given scenario description
        
        Args:
            None
        '''
        prompt_params = {
                    "scenario_description": self.scenario_description,
                    }
        prompt = self.generate_prompt("get_human_actors.txt", template_dir = self.template_dir,  **prompt_params)
        raw_output = self.call_llm(prompt)
        self.agents_dict = llm_json_loader(raw_output)
        self.agents = self.agents_dict['agents']
        return self.agents
    
    def backend_get_human_agents(self) -> List[str]:
        '''
        Gets a list of human agents for a given scenario description
        
        Args:
            None
        '''
        prompt_params = {
                    "scenario_description": self.scenario_description,
                    }
        prompt = self.generate_prompt("get_human_actors.txt", template_dir = self.template_dir,  **prompt_params)
        raw_output = self.call_llm(prompt)
        self.agents_dict = llm_json_loader(raw_output)
        self.agents = self.agents_dict['agents']
        return self.agents

    def outcome_generator(self, count: int = 10) -> List[str]:
        '''
        Generates a genarator outcomes for a given scenario description
        
        Args:
            count (int): number of outcomes to put in generator
        '''
        prompt_params = {
            "scenario_description": self.scenario_description,
            "agents": self.agents
            }
        prompt = self.generate_prompt("outcome_generator1.txt", template_dir = self.template_dir, **prompt_params)

        raw_output  = self.call_llm(prompt)
        output_list = llm_json_loader(raw_output)

        outcomes =  [item.strip() for item in output_list]
        index = 0
        while True:
            if index < len(outcomes):
                self.outcomes.append(outcomes[index])
                yield outcomes[index]
                index += 1
                if index >= count:
                    break
            else:

                prompt_params = {
                    "scenario_description": self.scenario_description, 
                    "outcomes": outcomes,
                    "agents": self.agents
                    }
                prompt = self.generate_prompt("outcome_generator2.txt", template_dir = self.template_dir, **prompt_params)
                raw_output = self.call_llm(prompt)
                output_list = llm_json_loader(raw_output)
                newoutcomes =  [item.strip() for item in output_list]
                outcomes.extend(newoutcomes)
                if index >= count:
                    break


    def backend_outcome_generator(self, count: int = 10) -> List[str]:
        '''
        Generates a genarator outcomes for a given scenario description
        
        Args:
            count (int): number of outcomes to put in generator
        '''
        prompt_params = {
            "scenario_description": self.scenario_description,
            "agents": self.agents
            }
        prompt = self.generate_prompt("outcome_generator1.txt", template_dir = self.template_dir, **prompt_params)

        raw_output  = self.call_llm(prompt)
        output_list = llm_json_loader(raw_output)

        outcomes =  [item.strip() for item in output_list]
        index = 0
        while True:
            if index < len(outcomes):
                self.outcomes.append(outcomes[index])
                yield outcomes[index]
                index += 1
                if index >= count:
                    break
            else:

                prompt_params = {
                    "scenario_description": self.scenario_description, 
                    "outcomes": outcomes,
                    "agents": self.agents
                    }
                prompt = self.generate_prompt("outcome_generator2.txt", template_dir = self.template_dir, **prompt_params)
                raw_output = self.call_llm(prompt)
                output_list = llm_json_loader(raw_output)
                newoutcomes =  [item.strip() for item in output_list]
                outcomes.extend(newoutcomes)
                if index >= count:
                    break


if "__main__" == __name__:
    LLM = LanguageModel(family = "openai", model = "gpt-4", temperature = .3, system_prompt = 'You are an economist who is interested in studying social scenarios.')
    # LLM = LanguageModel(family = 'replicate', model = 'llama13b_v2_chat', temperature = .1)
    scenario_descriptions = [
    "a family arguing whether or not to get a dog",
    "a couple deciding whether to move in together",
    "a couple deciding whether to get married"
                            ]
    # scenario_descriptions = [
    #    'the iterated prisoner dilemma in game theory',
    #     'the ultimatum game',
    #     'the dictator game']
    for scenario_description in scenario_descriptions:
        print(f'##### {scenario_description} #####')
        JP = JudeaPearl(scenario_description)
        JP.add_LLM(LLM)
        print(JP.get_human_agents())
        outcomes = JP.outcome_generator(count = 5)
        # print(type(outcomes))
        for outcome in outcomes:
            print(outcome)
    # print(type(JP.outcomes))
    # print(JP.outcomes)
    # outcomes = [outcome for outcome in JP.outcome_generator(count = 1)]
    # print(outcomes)
    # outcome_cause_dict = JP.outcome_cause_dict(outcomes, count = 2)
    # print(outcome_cause_dict)
    # json_dags_list = [dag for dag in JP.json_dag_generator(outcome_cause_dict)]
    # print(json_dags_list)
    # for dag in json_dags_list:
    #     SEM = StructuralEquationModel(dag_json=dag)
    #     SEM.print_model()
    #     SEM.graph_sem()