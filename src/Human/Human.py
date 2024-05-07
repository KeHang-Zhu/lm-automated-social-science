import sys
import time
import inspect 
import functools
import json

from collections import namedtuple

sys.path.append('../LLM')
sys.path.append('../Serialization')

from LLM import LanguageModel
from Serialize import RegisteredSerializable

def dict_to_string(d):
    s = []
    for key, value in d.items():
        # remove 'name' and 'role'
        if key not in ['your name', 'your role is']:
            s.append(f"{key}: {value}")
    return "".join(s)

def list_to_string(lst):
    # Check if lst is None and return an empty string
    if lst is None:
        return ""
    
    strings = []
    for d in lst:
        for key, value in d.items():
            strings.append(f"{key}: {value}")
    return ''.join(strings)


MemoryInput = namedtuple("MemoryInput", "args kwargs time")
MemoryRecord = namedtuple("MemoryRecord", "function inputs output time")

def is_yes(str):
    """Returns True if the string is some form of a yes""" 
    return "yes" in str.lower() in str.lower()

class MemoryLocation(RegisteredSerializable):
    """This will let us expand memory types more easily in the future"""
    def __init__(self):
        self.complete = []
        self.simple = []

    def __repr__(self):
        return f"complete: {self.complete}\nsimple: {self.simple}"

def remember(*memory_types):
    """
    This decorator remembers the inputs and outputs of a function.
    
    args: me
    
    """
    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            inputs = MemoryInput(args, kwargs, time.time())
            result = func(self, *args, **kwargs)
            record = MemoryRecord(func.__name__, inputs, result, time.time())
            
            for memory_type in memory_types:
                if hasattr(self.memory_locations, memory_type):
                    memory_list = getattr(self.memory_locations, memory_type)
                    memory_list.append(record)
                    setattr(self.memory_locations, memory_type, memory_list)
                else: 
                    raise ValueError("Invalid memory type. Refactor MemoryLocations class.")
                
            return result
        return wrapper
    return actual_decorator

class Human(RegisteredSerializable):
    def __init__(self, attributes):
        for key, value in attributes.items():
            if key == "LLM" and isinstance(value, dict):  # Convert dictionaries to LanguageModel
                value = LanguageModel.from_dict(value)
            setattr(self, key, value)
        self.attributes = attributes
        self.name = attributes['your name']
        #Memory of all things that an agent has said or another agent has said to it
        self.memory_locations = MemoryLocation()
        try:
            _ = self._goal  # check to see if goal is defined
        except AttributeError:
            raise Exception("You must specify a goal for the human.")
        
    
    def call_llm(self):
        raise NotImplementedError("This method gets implemented when you add an LLM")

    def add_LLM(self, LLM):
        self.LLM = LLM
        self.call_llm = self.LLM.call_llm

    @staticmethod
    def public_knowledge(counterparty):
        # return [f"{key}: {value}" for key, value in 
        #     counterparty.attributes.items() if not key.startswith("_")]
        # Dictionary for replacement
        replacement_dict = {"your role is": "role", "your name": "name"}
    
        # List comprehension to generate the output
        return [f"{replacement_dict[key]}: {value}" for key, value in 
            counterparty.attributes.items() if key in ["your role is", "your name"]]

    
    # @remember('complete')
    def current_context(self, history = None):
        context = f"""In this conversation you are {self.attributes['your role is']} named {self.attributes['your name']} with the following characteristics: {dict_to_string(self.attributes)}.Here is the conversation in the scenario so far: {list_to_string(history)}.
        """
        return context 
    
    @remember('complete')
    def final_context(self, group_knowledge, scenario_description, history):
        return f"""        
        You are person with the following characteristics: {dict_to_string(self.attributes)}.You have just participated in this conversation: {list_to_string(history)}.  which was a simulation of this scenario {scenario_description}, and was with these other people: {group_knowledge}. During the conversation your goal was: "{self._goal}" and you had the following constraint: {self._constraint}
        """
    @remember('complete')
    def survey(self, counterparties, scenario_description, question, history, EXDOGENOUS, VARIABLE, OPERATIONALIZATION):
        group_knowledge = [self.public_knowledge(counterparty) for counterparty in counterparties]
        context = self.final_context(group_knowledge, scenario_description, history)
        
        prompt = f"""{context}
        Your task is to answer the following question: '{question}'. When answering the question, please keep the following things in mid:
        1. You should base you answer frist on your personal characteristics provided to you and the past interactions you had in the simulated conversation.
        2. This simulated conversation was run as an experiment to test the effects of changing different attributes on the {EXDOGENOUS}.
        Your answer to the question will be directly used to operaionalize the measurement of the {VARIABLE} for data data analysis, which we originally chose to operationalize like this: {OPERATIONALIZATION}
        You should try as hard as possibly to accurately answer the question within the context of your characteristics, conversation, and usage for analyzing the simulation, but if you truly cannot answer the questions, you can say that you don't know.
        Format your response as a json in this form and make sure that all keys and items are in double quotes correctly:{{"explanation": "short explanation for choice”, "answer": "your answer to the question do get the data for the analysis."}}.
        """
        # an auction for a single contract with many bidders
        
        return self.call_llm(prompt)


    def is_rational(self, statement, history= None):
        # return True
        pass


    def make_public_statement(self, counterparties, scenario_description, round, n_left, history = None):
        group_knowledge = [self.public_knowledge(counterparty) for counterparty in counterparties]
        # prompt = f"""
        # {self.current_context()}
        # You are making a statement to a group of people, with following attributes: {group_knowledge}.
        # Please remember that all information that you wish to communicate must be stated directly to the other people you are speaking with.
        # Your statement:
        # """
        if not history:
            STRING = self.current_context(history)+ "You will be the first person to speak"
        else:
            STRING = self.current_context(history)

        prompt = f"""     
        You are currently participating in a conversation in this scenario {scenario_description}. 
        So far, there have been {round} total statements made in this conversation by all the agents. There will be at most {n_left} combined statements by all agents before the conversation automatically ends.
The people participating in the scenario have these roles and names: {group_knowledge}. {STRING}
It is your turn to speak.
Please remember that all information that you wish to communicate must be stated directly to the other people you are speaking with.
You should be concise and focus on accomplishing your goal within your constraints in the conversation with a minimal number of words.
Provide your natural response to this conversation without any other text:
        """
        
        # print("<<<<>>>>>", prompt)
        statement = self.call_llm(prompt)
        # is_rational = self.is_rational(statement, history)
        # return {'statement':statement, 'is_rational': is_rational}
        return {'statement':statement}


    def to_continue_or_to_finish(self, scenario, agents,ENDOGENOUS_VARIABLES, OPERATIONALIZATION, history=None):
        group_knowledge = [self.public_knowledge(agent) for agent in agents]
        prompt = f"""
        You are are a social scientist running a simulation of the following scenario: {scenario}. You are studying the behavior of these agents: {group_knowledge}. Here is the conversation between the agents so far: {history}. You must determine whether to continue or not based on what makes the most sense given the conversation so far.For example, if the agents seem like they are both mid-conversation, you should say continue. Conversely, if the agents are saying goodby to each other and it seems like it's reasonable to end the conversation like a normal conversation would end, then you should complete the conversation. Determine whether the conversation should continue or if is complete. Format your response as a json in this form and make sure that all keys and items are in double quotes correctly: {{"explanation": "short explanation for whether the simulation is complete or if it should continue","choice": "complete or continue"}}
        """
        
        response = self.call_llm(prompt)
        print(response)
        if "continue" in response.lower():
            return True
        else:
            return False 

    
    def how_to_you_think_other_person_will_respond(self, question):
        pass

    @remember('simple', 'complete')
    def does_this_response_help_your_goal(self, statement, response):
        prompt = f"""
        You recevied the following statement from someone else: {statement}
        You are considering this statment in light of your goal.
        Your goal is: {self._goal}.
        You are planning to respond: {response}.
        Thinking step by step, is this response consistent with your goals (yes/no)?"""
        return self.call_llm(prompt)
    
    def show_memory(self):
        return self.memory_locations
    
    def __eq__(self, other):
        if isinstance(other, Human):
            # Compare the relevant attributes here and return True if they are equal
            return self.attributes == other.attributes
        return False
