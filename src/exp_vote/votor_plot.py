import matplotlib.pyplot as plt

with open(json_file_path, 'r') as file:
    loaded_data = json.load(file)

# Parse the loaded JSON data
votes_from_json = []
for entry in loaded_data:
    voter, vote = list(entry.items())[0]
    votes_from_json.append({"voter": voter, "statement": vote})

# Extract data for plotting
x_from_json = [updated_voter_ids[vote["voter"]] for vote in votes_from_json]
y_from_json = [option_mapping[vote["statement"][-2]] for vote in votes_from_json]

# Mapping voter names to IDs and options to numeric values
voter_ids = {'Voter0': 0, 'Voter1': 1, 'Voter2': 2, 'Voter3': 3, 'Voter4': 4, 'Voter5': 5}
option_mapping = {'A': 0, 'B': 1, 'C': 2}

# Extracting data for plotting
x = [voter_ids[v['voter']] for v in votes]
y = [option_mapping[v['statement'][-2]] for v in votes]

# Preference data to display
preferences = [
    " A > B > C",
    " A > B > C",
    " B > C > A",
    " B > C > A",
    " C > A > B",
    " C > A > B"
]

# Creating the plot
plt.figure(figsize=(10, 5))
# Sorting the data by voter ID for proper line plotting
sequence_x = list(range(len(votes)))

plt.plot(sequence_x, y, 'o--', color='blue')  # Dashed line
# for i in range(len(x)):
#     plt.text(x[i], y[i] + 0.1, f'{preferences[i]}', ha='center')
# for idx, pref in enumerate(preferences):
#     plt.text(idx, -0.5, pref, ha='center', fontsize=8)

for i in range(len(sequence_x)):
    voter_preference_text = f'{votes[i]["voter"]}: {preferences[voter_ids[votes[i]["voter"]]]}'
    plt.text(sequence_x[i], y[i] + 0.1, voter_preference_text, ha='center', fontsize=8)
# Displaying preferences below each voter ID, adjusting x-axis labels
plt.xticks(sequence_x, [f'No.{i}' for i in sequence_x])
plt.yticks(range(len(option_mapping)), list(option_mapping.keys()))
plt.xlabel('Vote Order')
plt.ylabel('Voted Option')
plt.title('Voting Order and Options')
plt.grid(True)
plt.show()