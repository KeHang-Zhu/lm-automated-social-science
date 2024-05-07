"""RS entry point script."""
# src/__main__.py
import json
import typer
from dotenv import load_dotenv

from src import cli, __app_name__
# from .cli import build_agent, build_interaction, perform_simulation, perform_measurement

import sys
sys.path.append('./src/Example')

load_dotenv()

app = typer.Typer()

def main():
    cli.app(prog_name=__app_name__)
    # filepath = './src/Example_SCM/lawyer_interview_3var.json'

    # with open(filepath, 'r') as file:
    #     scm = json.load(file)
    # scm_dict = json.loads(scm)
    
    # scenario = scm_dict["args"]["scenario_description"]
    # max_interactions = 50
    # build_agent(scm_dict)
    
    # ## get the measurement Info
    # measurementsInfo, ENDOGENOUS_VARIABLES, OPERATIONALIZATION = get_measurement(scm_dict)
    # print('measurementsInfo',measurementsInfo, ENDOGENOUS_VARIABLES, OPERATIONALIZATION)

if __name__ == "__main__":
    main()