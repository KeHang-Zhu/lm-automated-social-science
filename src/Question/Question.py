from jinja2 import Template
from jinja2 import Environment, meta

class Question:
    """This class reprsents a question sent to an LLM (or human).
    It uses the jinja2 templating language to generate the prompt.
    For details on jinja2: https://jinja.palletsprojects.com/en/3.0.x/
    This language gives us lots of flexibility in how we ask questions.
    """
    def __init__(self, template_string):
        self.template_string = template_string
        self.template = Template(self.template_string)

    def ask(self, object):
        "Generates the prompt using the template and the data. Expects a dictionary or an object"
        data = object if isinstance(object, dict) else object.__dict__
        if not self._valid_data(data):
                raise ValueError("The data is not valid.")
        output = self.template.render(data) 
        return output
    
    def _valid_data(self, data):
        "Checks that the data contains all the variables used in the template."
        required_fields = set(self.get_variables())
        offered_fields = set(data.keys())
        return required_fields.issubset(offered_fields)

    def get_variables(self):
        "Returns a list of the variables used in the template."
        env = Environment()
        ast = env.parse(self.template_string)
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
    question = Question(template_string)
    print(f"Is the data valid? {question._valid_data(data)}")
    output = question.ask(data)
    print(output)
    print("The variables used in the template:")
    print(question.get_variables())

    class Example: 
        def __init__(self, **kwargs):
            self.name = "John"
            self.day = "Tuesday"
            self.week = "first"

    e = Example()
    print(question.ask(e.__dict__))
    print(question.ask(e))

 


