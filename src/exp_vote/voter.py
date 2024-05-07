from dotenv import load_dotenv
import os, sys
# Get the chat history.
import json
import random

base_path = '/Users/wonderland/Desktop/2023Fall/robot_scientist/src'

modules = ['LLM', 'Human', 'JudeaPearl', 'Question', 'Serialization', 'Interaction']
for module in modules:
    sys.path.append(os.path.join(base_path, module))

try:
    from LLM import LanguageModel, LLMMixin, llm_json_loader
    from Human import Human
    from Interaction import SocialInteraction
except ImportError as e:
    print(f"Error importing modules: {e}")


current_script_path = os.path.dirname(os.path.abspath(__file__))

## load the .env file
load_dotenv()

def call_openai(agent_list, order_dict, scenario, interaction_type, max_interactions):
    agentsInfo = agent_list
    
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=.0)
    L_S = LanguageModel(family="openai", model="gpt-4", temperature=.7)
    
    agents = {}
    agent_list = []
    conversation_history = []
    conversation_history_simple = []

    for agent_type, attributes in agentsInfo.items():
        agent = Human(attributes)
        agent.add_LLM(LLM)
        agents[agent_type] = agent
        agent_list.append(agent)
        
    print(order_dict)
    S = SocialInteraction(agent_list, scenario=scenario)
    S.add_LLM(L_S)
    generator = S.gen_func_dispatch[interaction_type](order_dict)
    FirstAgent = agents[next(generator)]
    SecondAgent = agents[next(generator)]
    others = [agent for agent in agent_list if agent != FirstAgent]
    interactions = 0
    n_left = max_interactions - interactions
    
    statement = FirstAgent.make_public_statement(others, scenario, interactions, n_left, conversation_history_simple)
    print("No.0", statement, FirstAgent.name)
        
    conversation_history.append({FirstAgent.name: statement})
    conversation_history_simple.append({FirstAgent.name: statement['statement']})
    if statement['is_rational'] is False:
        print("Error! The response is not rational")
        return conversation_history

    while True:
        interactions += 1
        n_left -= 1
        others = [agent for agent in agent_list if agent != SecondAgent]
        newstatement = SecondAgent.make_public_statement(others, scenario,interactions, n_left, conversation_history_simple)
        name = SecondAgent.name
        FirstAgent = SecondAgent
        SecondAgent = agents[next(generator)]
        statement = newstatement
        print(f"No.{interactions}", statement, FirstAgent.name)
        
        conversation_history.append({name: newstatement})
        conversation_history_simple.append({name: statement['statement']})
        if statement['is_rational'] is False:
            print("Error! The response is not rational")
            break
        
        if interactions > max_interactions:
            break  
        
    return [conversation_history, conversation_history_simple]


## Roll call voting
SYSTEMMESSAGE = 'You are a voter attending a roll call voting.'

## preference menu:
#option A, B, C.
option = ["A", "B", "C"]
order = [
    f'Your preference order is {option[0]}, then {option[1]}, followed by {option[2]}',
    f'Your preference order is {option[1]}, then {option[2]}, followed by {option[0]}',
    f'Your preference order is {option[2]}, then {option[0]}, followed by {option[1]}'     
]

## Condition
RULE = 'The rules of game are: '
Nondictatorship = 'The wishes of all voters will be taken into consideration. '
ParetoEfficiency = 'The option with the highest number of votes wins. '
Independence = "You don't know other voters' order. "
UnrestrictedDomain = 'Voting will account for all individual preferences. '
SocialOrdering = 'You should give your most prefered option. Give your output as a single letter, A, B or C' ## I made a few changed to the original assumption

RULE += Nondictatorship+ ParetoEfficiency +Independence + UnrestrictedDomain

number = 6
## Additional information.
NPeople = f'There are {number} voters in this voting.'

scenario = f'{number} voters are participating a roll call voting.'
max_interactions = number -2

# Initialize a dictionary to hold agent configurations
agent_list = {}

# Populate the agents_config with each agent's details
for i in range(number):
    preference = order[i // 2]  # Integer division to repeat each order twice
    full_system_message = f"{SYSTEMMESSAGE} {NPeople}\n{preference}"

    # Use the voter index directly as the key for more direct access
    agent_list[f"Voter{i}"] = {
        "your role is": "voter",
        "your name": f"Voter{i}",
        "your preference menu is": full_system_message,
        "_goal": 'Make the final majority voting favorable based on your personable preference menu',
        "_constraint": RULE + SocialOrdering,
    }
    print(i, preference)

# agent_list["moderator"] = {
#         "your role is": "moderator",
#         "your name": "moderator",
#         "_goal": 'You should ask people about their options in a voting over A, B and C',
#         "_constraint": '',
#     }
   
## Benchmark: random ordering
# Shuffle the chat configurations to ensure random order
voter_ids = [f"Voter{i}" for i in range(number)]

# Shuffle the list to randomize the order of interaction
random.seed(1234)
random.shuffle(voter_ids)

# If you need to specify a central agent or some kind of moderator, assign it here
# central_agent = "Voter0"  # Example: making Voter0 the central agent

interaction_type = "ordered"
# Define the interaction order after shuffling
order_dict= {
    "order": voter_ids,
    "central agent": ''
}
print(order_dict)

# # Pick up a random person to start the interaction
# first = random.randint(0, number - 1)
conversation_history, conversation_history_simple = call_openai(agent_list, order_dict, scenario,  interaction_type, max_interactions) 


# Converting the conversation history into JSON format
json_data = json.dumps(conversation_history_simple, indent=4)
json_file_path = './VotingData.json'
with open(json_file_path, 'w') as file:
    file.write(json_data)



