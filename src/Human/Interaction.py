import sys
import itertools
import random 

sys.path.append('./src/LLM')
sys.path.append('./src/Serialization')

from LLM import LanguageModel
from Serialize import RegisteredSerializable
from Prompting import PromptMixin

params = {
    "family":"openai",
    "model": "gpt-4", #"gpt-3.5-turbo", 
    "temperature":0.7   
}
L = LanguageModel(**params)

class SocialInteraction(RegisteredSerializable, PromptMixin): 
    def __init__(self, agents, scenario, template_dir: str ="prompt_templates"):
        
        self.agents = agents
        self.statements = []
        self.template_dir = template_dir
        self.scenario = scenario
        self.gen_func_dispatch = {
            'ordered': self.infinite_loop_generator, 
            'random': self.random_generator,
            'center ordered': self.center_order,
            'center random': self.center_random,
            'ask_oracle_prescriptively': self.ask_oracle_prescriptively,
            'ask_oracle_post': self.ask_oracle_post,
            }   
         
        
    def call_llm(self):
        raise NotImplementedError("This method gets implemented when you add an LLM")

    def add_LLM(self, LLM):
        self.LLM = LLM
        self.call_llm = self.LLM.call_llm

    @staticmethod
    def infinite_loop_generator(items):
        """Yeilds the next agent in the list, looping around"""
        order = items['order']
        cycle = itertools.cycle(order)
        while True:
            yield next(cycle)
    
    @staticmethod
    def random_generator(items):
        """Yeilds a random agent from the list"""
        order = items['order']
        while True:
            yield random.choice(order)
            
    # @staticmethod
    def center_order(self,items):
        """Yields the central agent every other time and cycles through the other agents in between"""
        # print(">>>>>>>>")
        # center = items['central agent']
        other_agents = items['order']
        # if center is not None and center in self.agents:
        #     other_agents = [agent for agent in self.agents if agent != center]
        cycle = itertools.cycle(other_agents)
        while True:
            yield items['central agent']
            yield next(cycle)
            
    @staticmethod       
    def center_random(items):
        """Yields the central agent every other time and yields a random agent from the rest of the list in between"""
        central_agent = items['central agent']
        other_agents = [agent for agent in items['order'] if agent != central_agent]
        while True:
            yield items['central agent']
            if other_agents:
                yield random.choice(other_agents)
    
    def ask_oracle_prescriptively(self, items):
        """Asks the oracle to pick the next agent based on the past messages"""
        # self.add_LLM(L)
        agent_list = items['order']
        # print(self.scenario)
        
        while True:
            #  oracle function
            prompt_params = {
                    "scenario": self.scenario,
                    "statements": self.statements,
                    "agent_list": agent_list
                    }
            ask_template = self.generate_prompt("ask_oracle_prescriptively.txt", template_dir = self.template_dir, **prompt_params)
            pick_up = self.call_llm(ask_template)
            # print(pick_up)
            next_agent = pick_up['choice_of_next_agent']
            yield next_agent        
    
           
    def ask_oracle_post(self, items):
        """Lets each agent respond after one speaks, then yields the response picked by the oracle and wipes the others"""
        agent_list = items['order']
        responses = {}
        while True:
            for agent in self.agents:
                name = agent.name
                role = agent.role
                prompt_params_agent = {
                    "name": name,
                    "role":role,
                    "scenario": self.scenario,
                    "statements": self.statements,
                    "agent_thoughts": responses,
                    "agent_list": agent_list
                    }
                ask_template = self.generate_prompt("ask_agent_thoughts.txt", template_dir = self.template_dir, **prompt_params_agent)
                responses[name] = agent.call_llm(ask_template)['thoughts']
            
            prompt_params = {
                    "scenario": self.scenario,
                    "statements": self.statements,
                    "agent_thoughts": responses,
                    "agent_list": agent_list
                    }
            oracle_template = self.generate_prompt("ask_oracle_post.txt", template_dir = self.template_dir, **prompt_params)
            
            pick_up = self.call_llm(oracle_template)
            next_agent = pick_up['choice_of_next_agent']
            if next_agent:
                yield next_agent     


    def survey_agents(self, question):
        responses = {}
        for agent in self.agents:
            response = agent.survey(question)
            responses[agent.name] = response
        return responses

    def interact(self, gen_func_type = "ordered", max_interactions = 20, verbose = False):
        generator = self.gen_func_dispatch[gen_func_type](self.agents)
        FirstAgent = next(generator)
        SecondAgent = next(generator)
        statement = FirstAgent.make_statement(SecondAgent)
        if verbose:
           print(f"{FirstAgent.name}: {statement}\n")
        interactions = 0
        while True:
            newstatement = SecondAgent.get_response(statement, FirstAgent)
            if verbose:
                print(f"{SecondAgent.name}: {newstatement}\n")
            FirstAgent = SecondAgent
            SecondAgent = next(generator)
            statement = newstatement
            interactions += 1
            if not FirstAgent.do_you_want_to_continue():
                break
            if interactions > max_interactions:
                break

    def report_memory(self, memory_type = "simple"):
        for agent in self.agents:
            print(f"agent's memory:\n")
            print(agent.show_memory())
#            if memory_type == "simple":
#                print(agent.memory)
#            if memory_type == "complete":
#                print(agent.complete_memory)


