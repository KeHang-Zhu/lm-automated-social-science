import json
import re

# Load the JSON file
file_path = 'survey_number_rounds.json'

with open(file_path, 'r') as file:
    data = json.load(file)
    
name_list = {'oliver': 1, "emma": 2, 'liam': 3}
highest_bidder_data = {}

# Extract the relevant information
data_section = data.get("data", [])
# survey_data = data_section.get("0", {}).get("survey", [])
all_survey_data = {}
for key in data_section.keys():
    if key.isdigit():  # Checking if the key is a digit
        survey_data = data_section[key].get("survey", [])
        
        print(key)
        # print(who_made_highest_bid_section)


        # Loop through the survey data to find the highest bidder
        # for item in survey_data:
        who_made_highest_bid_section = survey_data.get("number rounds", {})
        auctioneer_section = who_made_highest_bid_section.get("oracle", {})
        highest_bid_answer = auctioneer_section.get("Read the transcript, count how many number of bidding rounds are in the auction process. If the auctioneer asked a price and the bidder accepted or dropped out, this is counted as one round. If the bidder had already dropped out and auctioneer asked him again, it is not counted as one round. Give a number as answer", {})
        # print(json.loads(highest_bid_answer)["answer"])
        # Extract the name from the answer
        answer_content = json.loads(highest_bid_answer)["answer"]
        highest_bidder_data[key] = int(answer_content)
        # if answer_content.lower() == "yes":
            
        #     # highest_bidder_name = answer_content.split(' ')[0]
        #     # highest_bidder_name = highest_bidder_name.lower()
        #     # Convert the name to the corresponding number
        #     # if highest_bidder_name in name_list:
        #         # print(highest_bidder_name)
        #     highest_bidder_data[key] = 1
        # else: 
        #     highest_bidder_data[key] = 0

print(highest_bidder_data)




# import json
# import re

# file_path = "result_two people bargaining over a mug.json"
# # 'result_3 bidders participating in an auction for a piece of art starting at $100.json'
# with open(file_path, 'r') as file:
#     data = json.load(file)
    
# highest_bidder_data = {}


# outcome = "buyer satisfaction"
# agent = "buyer"
# question = "on a scale from 'very dissatisfied', 'dissatisfied', 'neutral', 'satisfied' to 'very satisfied', how would you rate your satisfaction with bargain process?"
# convertion_table = {'very dissatisfied': 0, 'dissatisfied':1, 'neutral':2, 'satisfied':3, 'very satisfied':4}

# # Extract the relevant information
# data_section = data.get("data", [])
# # survey_data = data_section.get("0", {}).get("survey", [])
# all_survey_data = {}
# for key in data_section.keys():
#     if key.isdigit():  # Checking if the key is a digit
#         survey_data = data_section[key].get("survey", [])
        
#         print(key)
#         who_made_highest_bid_section = survey_data.get(outcome, {})
#         # print(who_made_highest_bid_section)
#         # Loop through the survey data to find the highest bidder
#         # for item in survey_data:
#         # who_made_highest_bid_section = survey_data.get("who made the highest bid", {})
#         auctioneer_section = who_made_highest_bid_section.get(agent, {})
#         highest_bid_answer = auctioneer_section.get(question, {})
#         # Extract the name from the answer
#         answer_content = json.loads(highest_bid_answer)["answer"]
#         print(answer_content)
#         highest_bidder_data[f"{key}"] = answer_content
#         all_survey_data[f"{key}"]= convertion_table[f"{answer_content}"]
        
        
#         # if answer_content:
#         #     numbers = re.findall(r'\d+', answer_content)

#         #     # Converting extracted numbers to integer
#         #     for num in numbers:
#         #         numbers = int(num)
#         #     if numbers:
#         #         print(numbers)
#         #         highest_bidder_data[f"{key}"] = numbers
#         #     else:
#         #         highest_bidder_data[f"{key}"] = 0
#         #         print("missing data")
                
# # json_str1 = json.dumps(highest_bidder_data, indent=4)          
# print(highest_bidder_data)
# print(all_survey_data)