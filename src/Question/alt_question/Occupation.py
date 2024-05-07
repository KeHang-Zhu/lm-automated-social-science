import sqlalchemy
import random
import json
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from LLM import LanguageModel

from jinja2 import Template
from jinja2 import Environment, FileSystemLoader

from Question import Question

class PromptLibrary:
    def __init__(self, template_dir = "prompt_templates"):
        self.template_dir = template_dir
        self.env = Environment(loader=FileSystemLoader(self.template_dir))

    def show_templates(self):
        return self.env.loader.list_templates()
    
    def get_template_string(self, template_name):
        return self.env.loader.get_source(self.env, template_name)[0]


# Connect to the database and reflect the schema
engine = create_engine('sqlite:///occupation-task.db')
Base = automap_base()
Base.prepare(engine, reflect=True)

def get_row(table_name):
    TableClass = Base.classes[table_name]
    session = Session(engine)
    row = session.query(TableClass).first()
    return TableClass(**row.__dict__)

class Object:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key.startswith('_'):
                pass
            else:
                setattr(self, key, value)
    
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
    
    def ask(self, question, data = None):
        """Asks a question to the model using the data in self.__dict__ as paramters unless 
        passed a data paramter."""
        if data:
            prompt = question.ask(data)
        else:
            prompt = question.ask(self.__dict__)
        return self.call_llm(prompt)

    def add_LLM(self, LLM):
        self.LLM = LLM
        self.call_llm = LLM.call_open_ai_apt_35

    def add_prompt_library(self, prompt_library): 
        self.prompt_library = prompt_library

    def __repr__(self):
        return f'<Object {self.__class__.__name__}>'

class Task(Object):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.occupation = None
    
    def add_occupation(self, Occupations):
        for _, occupation in Occupations.items():
            if occupation.onetsoc_code == self.onetsoc_code:
                self.occupation = occupation
                break

    def how_does_llm_help(self):
        q = Question(library.get_template_string("llm_usefulness.txt"))
        return self.ask(q)

    def easy_to_asess_output(self): 
        q = Question(library.get_template_string("llm_rubin.txt"))
        return self.ask(q)

class Occupation(Object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tasks = []

    def add_tasks(self, Tasks):
        for _, task in Tasks.items():
            if task.onetsoc_code == self.onetsoc_code:
                self.tasks.append(task)

    def assess_tasks(self):
        self.scores = {}
        for task in self.tasks:
            print(f"Now working on task: {task.task}")
            task.add_LLM(self.LLM)
            score_rubin = task.easy_to_asess_output()
            score_llm = task.how_does_llm_help()
            self.scores[task.task] = {'score_rubin': score_rubin, 'score_llm': score_llm}

    def task_ordering(self, task1, task2):
        d = dict{"task1": task1, "task2": task2}
        q = Question(library.get_template_string("task_ordering.txt"))
        return self.ask(q, d)


if __name__ == "__main__":
    params = dict({
        "model": "gpt-3.5-turbo",
        "family": "openai",
        "temperature": 1.0
    })
    L = LanguageModel(**params)
    library = PromptLibrary()

    if True:
        task1 = Task(**{'task': "Mix ingredients"})
        task1.add_LLM(L)
        library = PromptLibrary()

        q1 = Question(library.get_template_string("llm_usefulness.txt"))
        q2 = Question(library.get_template_string("llm_rubin.txt"))
        q3 = Question(library.get_template_string("task_ordering.txt"))

        print(task1.ask(q1))
        print(task1.ask(q2))

    if False:
        table_name = 'task_statements'
        TableClass = Base.classes[table_name]
        session = Session(engine)
        rows = session.query(TableClass).all()
        Tasks = {row.__dict__['task_id']:Task(**row.__dict__) for row in rows}

        table_name = 'occupation_data'
        TableClass = Base.classes[table_name]
        session = Session(engine)
        rows = session.query(TableClass).all()


        Occupations = {row.__dict__['onetsoc_code']:Occupation(**row.__dict__) for row in rows}
        [occupation.add_tasks(Tasks) for _, occupation in Occupations.items()]
        [task.add_occupation(Occupations) for _, task in Tasks.items()]

        index = int(input(f'Enter a number between 1 and {len(Occupations)}: '))
        code, o = list(Occupations.items())[index]

        params = dict({
            "model": "gpt-3.5-turbo",
            "family": "openai",
            "temperature": 1.0  
        })
        L = LanguageModel(**params)
        o.add_LLM(L)
        task1 = Task(**{'task': "Mix ingredients"})
        task2 = Task(**{'task': "Bake cake"})
        task3 = Task(**{'task': "Design a nuclear reactor"})

        order = o.task_ordering(task1, task2)
        print(order)

        #relationship = o.task_related(task1, task3)
        #print(relationship)

        #index = random.choice(range(len(Occupations)))
        # ask for input between 1 and range(len(Occupations))
        if False:
            index = int(input(f'Enter a number between 1 and {len(Occupations)}: '))
            code, o = list(Occupations.items())[index]
            print(f'Occupation: {o.title}')
            params = dict({
                "model": "gpt-3.5-turbo",
                "family": "openai",
                "temperature": 1.0
            })
            L = LanguageModel(**params)
            o.add_LLM(L)
            print("Assesing tasks")
            o.assess_tasks()
            print(o.scores)



