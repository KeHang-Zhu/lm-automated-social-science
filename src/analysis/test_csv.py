import json
import csv

# Load the JSON file
file_path = 'survey_price.json'

with open(file_path, 'r') as file:
    data = json.load(file)

# Prepare headers for the CSV file
headers = ['Key', 'bid1_max_budget','bid2_max_budg', 'bid3_max_budg', 'final_art_price']

# Initialize a list to store CSV data
csv_data = []

# Extract the relevant sections from the loaded JSON data
attribute_value_mapping = data.get("attribute_value_mapping", {})
survey_data_section = data.get("data", {})

for key in attribute_value_mapping.keys():
    if key.isdigit():  # Ensuring the key is a digit
        # Extracting budget data
        budgets = attribute_value_mapping[key]
        bid1_max_budget = int(budgets.get("bidder 1's maximum budget for the piece of art", [])[1:])

        bid2_max_budg = int(budgets.get("bidder 2's maximum budget for the piece of art", "0")[1:])
        bid3_max_budg = int(budgets.get("bidder 3's maximum budget for the piece of art", "0")[1:])
        
        # Extracting survey data for the same key if available
        survey_data = survey_data_section.get(key, {}).get("survey", {})
        who_made_highest_bid_section = survey_data.get("final bid price", {})
        auctioneer_section = who_made_highest_bid_section.get("oracle", {})
        highest_bid_answer = auctioneer_section.get("Read the transcript, what is the final bid of the piece of art that at the end of the auction? Outout a number", {})
        # Directly using the dictionary without json.loads()
        answer_content = json.loads(highest_bid_answer).get("answer", "0")
        
        final_price = int(answer_content)
        
        # Append combined data for this key to the CSV data list
        csv_data.append([key, bid1_max_budget, bid2_max_budg, bid3_max_budg, final_price])

# Writing the compiled data to a CSV file
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(csv_data)