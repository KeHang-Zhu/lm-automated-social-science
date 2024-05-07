import openai
import replicate
from dotenv import load_dotenv
import os
import sys
import json
import random
from json.decoder import JSONDecodeError
import logging
from retrying import retry
from typing import Dict, List, Union

load_dotenv()

sys.path.append('./src/Serialization')
# current_script_path = os.path.dirname(os.path.abspath(__file__))

from Serialize import RegisteredSerializable

class LanguageModel(RegisteredSerializable):
    def __init__(self, model: str, family: str, temperature: float, max_tokens = None, system_prompt: str = "") -> None:
        self.model: str = model
        self.family: str = family
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.system_prompt: str = system_prompt
        openai.organization = os.getenv('ORGANIZATION_ID')
        openai.api_key = os.getenv('OPENAI_API_KEY')

        self.family_model_mapping = {
            "openai": {
                "text-davinci-003": 'call_openai_api',
                "gpt-3.5-turbo": 'call_openai_api_35',
                "gpt-4": 'call_openai_api_35'
            },
            "replicate": {
                "llama70b-v2-chat": 'call_llama70b_v2',
                "llama13b-v2-chat": 'call_llama13b_v2'
            }
        }
        
        if self.family not in self.family_model_mapping:
            raise ValueError(f"Family '{family}' not supported.")

        if self.model not in self.family_model_mapping[self.family]:
            raise ValueError(f"Model '{model}' not supported for the '{family}' family.")

        self.call_llm = getattr(self, self.family_model_mapping[self.family][self.model])


    def __repr__(self):
        string = f'''Family: {self.family}\nModel: {self.model}\nTemperature: {self.temperature}'''
        return string

    def list_valid_LLMs(self) -> None:
        for family, model in self.family_model_mapping.items():
            print(f'{family}: {list(model.keys())}')

    @retry(wait_exponential_multiplier = 1000, wait_exponential_max = 10000, stop_max_attempt_number = 100)
    def call_openai_api_35(self, prompt: str) -> str:        
        try:
            response = openai.ChatCompletion.create(
                model = self.model,
                messages = [{"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}],
                max_tokens = None if self.max_tokens is None else self.max_tokens,
                temperature = self.temperature
                )
            return response["choices"][0]["message"]["content"]
        
        except openai.error.RateLimitError as e:
            logging.exception("Rate limit exceeded. Retrying...")
            raise e

    @retry(wait_exponential_multiplier = 1000, wait_exponential_max = 10000, stop_max_attempt_number = 100)
    def call_openai_api(self, prompt: str) -> str:
        try:
            response = openai.Completion.create(
                engine = self.model,
                prompt = prompt,
                max_tokens = 100,
                n = 1,
                stop = None,
                temperature = self.temperature
            )
            return response.choices[0].text.strip()
        except openai.error.RateLimitError as e:
            logging.exception("Rate limit exceeded. Retrying...")
            raise e
    
    def call_llama70b_v2(self, prompt: str) -> str:
        model = "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf"
        output = replicate.run(model,
                        input={"prompt":prompt,
                        "top_p": 1,
                        "system_prompt": """You are a helpful assistant.""",
                        "temperature": self.temperature, 
                        "max_length": 500,
                        "repetition_penalty": 1.5
                        }
                    )
        result = ''.join(output)
        return result
    
    def call_llama13b_v2(self, prompt: str, top_p: float = 1, max_length: int = 500, repetition_penalty: float = 1) -> str:
        model = "a16z-infra/llama-2-13b-chat:d5da4236b006f967ceb7da037be9cfc3924b20d21fed88e1e94f19d56e2d3111"
        output = replicate.run(model,
                        input={"prompt":prompt,
                        "top_p": 1,
                        "system_prompt": """You are a helpful assistant.""",
                        "temperature": self.temperature, 
                        "max_length": 500,
                        "repetition_penalty": 1.5
                        }
                    )
        result = ''.join(output)
        return result
        
class LLMMixin:
    def add_LLM(self, LLM: 'LanguageModel') -> None:
        self.LLM = LLM

    def call_llm(self, prompt: str) -> str:
        if hasattr(self, 'LLM'):
            return self.LLM.call_llm(prompt)
        else:
            raise NotImplementedError("This method gets implemented when you add an LLM")
        
def llm_json_loader(raw_llm_output: str) -> Dict[str, str]:
    '''
    Reads the JSON output from LLM and raises an exception if it's invalid
    
    Args:
        llm_output (str): JSON output from LLM
        known_keys (list, optional): A list of keys that should be in the JSON. Defaults to None as we might not know the key names
    '''
    llm_output: str = raw_llm_output
    for attempt in range(3):
        try:
            llm_json = json.loads(llm_output.lower())
            return llm_json
        
        except JSONDecodeError as e:
            print(f'Attempt {attempt+1}: LLM returned invalid JSON BUT WILL CALL LLM AGAIN TO FIX{llm_output}', e.doc, e.pos)
            error_doc, error_pos = e.doc, e.pos

        try:
            llm_output = json_corrector(llm_output, error_doc, error_pos)

        except Exception as e:
            print(f'Attempt {attempt+1}/3: LLM cleanup returned an error: {str(e)}\nATTEMP {attempt+2}/3: INITIALIZE LLM AGAIN AND TRY TO PARSE THE JSON AGAIN')

    # If after three attempts we still couldn't parse the JSON, raise an exception
    print(f'ORIGINAL STRING FROM LLM: {raw_llm_output}')
    print(f'FINAL FAILED STRING FROM LLM: {llm_output}')
    raise JSONDecodeError("INVALID JSON")

def json_corrector(llm_output: str, error_doc, error_pos) -> str:
    '''
    Sends json to LLM to fix it and returns the fixed json
    '''

    LLM = LanguageModel(family = "openai", model = "gpt-4", temperature = 0.1)

    cleanup_prompt = f'''The following json is invalid: {llm_output}
with the following error: {error_doc} at position {error_pos}.
There could also be other errors in the json that were not reached.
Please fix all errors in the JSON so that it's valid while still mainting the information in the text.
Format your output as the fixed, valid JSON ithout any other text so it can be loaded directly into python as a json string.
'''
    llm_cleaned_output = LLM.call_llm(cleanup_prompt)

    return llm_cleaned_output.lower()



if __name__ == "__main__":
    if False:
        """Test that the LLM can be serialized and deserialized"""
        params = {
            "family":"openai",
            "model": "text-davinci-003", #"gpt-3.5-turbo", 
            "temperature": 0.7   
        }
        LLM = LanguageModel(**params)
        json_str = LLM.serialize()
    if False:
        LLM = LanguageModel(family = 'replicate', model = 'llama13b-v2-chat', temperature = .1)
        output = LLM.call_llm('''In the following scenario: "A person getting cognitive behavioral therapy",
    Who are the individual human agents in a simple simulation of this scenario?
    The agents should have specified roles.
    For example, if the scenario was "negotiating to buy a car", then the agents should not be ["person 1", "person 2"], but should be ["seller", "buyer"]
    Only include agents that would speak during the scenario.
    Respond with a list of individual human agents, with the roles as their titles.
    Do not include a plurality of agents as a single agent in the list.
    Evey item in the list must be a singular agent, even if this makes the list long.
    For example, if the scenario was "a criminal case" the correct list of roles would be:
    ["judge", "defendant", "prosecutor", "defense attorney", "juror 1", "juror 2"]
    And the following would be incorrect: 
    ["judge", "defendant", "lawyers", "jurers"]
    You should respond with a python list as displayed in the correct example.
    Respond with a JSON in the following format and do not include any other text outside of the json:
    {"agents": ["agent 1", "agent 2", "agent 3", "agent 4"],
    "explanation": "explanation for choice of agents"}''')
        print(output)
        # print(LLM.call_llm('hey how are you?'))
        LLM.list_valid_LLMs()
        
    if True:
        # LLM = LanguageModel(family="openai", model="gpt-4", temperature=.4)
        # print(LLM.call_llm('hey how are you?'))
        openai.api_key = os.getenv('OPENAI_API_KEY')
        response = openai.chat.completions.create(
                model = 'gpt-4',
                messages = [{"role": "system", "content": ''},
                            {"role": "user", "content": 'hey how are you?'}],
                max_tokens = 256,
                temperature = 0.5
                )
        print(response)
