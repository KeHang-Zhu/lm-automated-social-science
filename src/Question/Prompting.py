from jinja2 import Template, Environment, meta, FileSystemLoader
import json
from typing import Dict, List, Union
from json.decoder import JSONDecodeError
import sys

sys.path.append('./src/LLM')

class PromptLibrary:
    '''
    This class represents a library of prompt templates.

    Args:
        template_dir (str, optional): The directory where the templates are stored. Defaults to "prompt_templates".
    '''
    def __init__(self, template_dir: str = "prompt_templates") -> None:
        self.template_dir: str = template_dir
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
    
    def show_templates(self) -> List[str]:
        '''
        Returns a list of the templates in the library.
        '''
        return self.env.loader.list_templates()
    
    def get_template_string(self, template_name: str) -> str:
        '''
        Returns the string of a template.
        '''
        return self.env.loader.get_source(self.env, template_name)[0]


class PromptBuilder:
    """This class reprsents a question sent to an LLM (or human).
    It uses the jinja2 templating language to generate the prompt.
    For details on jinja2: https://jinja.palletsprojects.com/en/3.0.x/
    This templating language gives us lots of flexibility in how we ask questions.
    """
    def __init__(self, template_string: str):
        self.template_string: str = template_string
        self.template = Template(self.template_string)

    def __add__(self, fragment: 'PromptBuilder') -> 'PromptBuilder':
        "Adds two PromptBuilders together and returns a new PromptBuilder."
        return PromptBuilder(self.template_string + fragment.template_string)
 
    def append(self, addendum: str) -> None:
        "Adds a string to the end of the template and re-renders the template."
        self.template_string += addendum
        self.template = Template(self.template_string)

    def build_prompt(self, object: Union[Dict, object]) -> str:
        "Generates the prompt using the template and the data. Expects a dictionary or an object"
        data = object if isinstance(object, dict) else object.__dict__
        if not self._valid_data(data):
                raise ValueError(f"The data is not valid. The valid data is : {self.get_variables()}")
        output = self.template.render(data) 
        return output
    
    def _valid_data(self, data) -> bool:
        "Checks that the data contains all the variables used in the template."
        required_fields = set(self.get_variables())
        offered_fields = set(data.keys())
        return required_fields.issubset(offered_fields)

    def get_variables(self) -> List[str]:
        "Returns a list of the variables used in the template."
        env = Environment()
        ast = env.parse(self.template_string)
        variables = meta.find_undeclared_variables(ast)
        return list(variables)

class PromptMixin:
    def generate_prompt(self, prompt_name: str, template_dir: str = "prompt_templates", **kwargs) -> str:
        '''
        Creates a prompt from a template and a dictionary of variables to fill in the template.

        Args:
            prompt_name (str): The name of the template file in the template directory.
            template_dir (str, optional): The directory where the template is stored. Defaults to "prompt_templates".
            **kwargs: The variables to fill in the template.
        '''
        prompt_library = PromptLibrary(template_dir)
        prompt_string = prompt_library.get_template_string(prompt_name)
        prompt_template = PromptBuilder(prompt_string)
        prompt = prompt_template.build_prompt(kwargs)
        return prompt
    
    def get_prompt_variables(self, prompt_name: str , template_dir: str = "prompt_templates") -> List[str]:
        '''
        Gets the variables used in a template.

        Args:
            prompt_name (str): The name of the template file in the template directory.
            template_dir (str, optional): The directory where the template is stored. Defaults to "prompt_templates".
        '''
        prompt_library = PromptLibrary(template_dir)
        prompt_string = prompt_library.get_template_string(prompt_name)
        env = Environment()
        ast = env.parse(prompt_string)
        variables = meta.find_undeclared_variables(ast)
        return list(variables)


if __name__ == "__main__":
    template_string = """
    Hello, {{ name }}!
    Today is {{ day }}.
    """
    data = {
        'name': 'John',
        'day': 'Monday'
    }
    question = PromptBuilder(template_string)
    print(f"Is the data valid? {question._valid_data(data)}")
    output = question.build_prompt(data)
    print(output)
    print("The variables used in the template:")
    print(question.get_variables())
    class Example: 
        def __init__(self, **kwargs):
            self.name = "John"
            self.day = "Tuesday"
            self.week = "first"

    e = Example()
    print(question.build_prompt(e.__dict__))
    print(question.build_prompt(e))


    COT = PromptBuilder("This step by step")
    BetterQuestion = question + COT
    print(BetterQuestion.build_prompt(e))

    ### test prompt library
    library = PromptLibrary(template_dir='test_prompts')

    print(library.show_templates())

    p1 = PromptBuilder(library.get_template_string('test1.txt'))
    p2 = PromptBuilder(library.get_template_string('test2.txt'))
    print(p1.get_variables())
    params = {
        'name': 'Richard Feynman',
        'hobby': 'bongos',
        'home': 'Rockaway',
        'school': 'MIT'
    }
    print(p1.build_prompt(params))
    print(p2.template_string)

 


