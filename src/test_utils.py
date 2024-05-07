import sys

sys.path.append('./src')
from .utils import generate_all_combinations_with_mapping


exp_role = {
        "brother": {
            "your role is": "brother",
            "your name": "jacob",
            "your communication style": "",
            "your current inclination": "",
            "_goal": "express your disinterest in watching the movie and try to convince your sister to do something else",
            "_constraint": "you should not agree to watch the movie under any circumstances"
        },
        "sister": {
            "your role is": "sister",
            "your name": "emily",
            "brother's communication style": "",
            "_goal": "try to persuade your brother to watch the movie with you by explaining why it would be enjoyable.",
            "_constraint": "you should not resort to irrational methods such as force or blackmail to persuade your brother."
        }
    }
variations =  {
        "bro's communication style": {
            "brother": {
                "your communication style": [
                    "passive",
                    "indirect",
                    "neutral",
                    "assertive",
                    "aggressive"
                ]
            },
            "sister": {
                "brother's communication style": [
                    "passive",
                    "indirect",
                    "neutral",
                    "assertive",
                    "aggressive"
                ]
            }
        },
        "bro's current inclination": {
            "brother": {
                "your current inclination": [
                    "desire to engage in a different activity",
                    "dislike for the genre of the movie",
                    "lack of interest in the movie",
                    "limited time availability",
                    "no specific reason"
                ]
            },
            "sister": {}
        }
    }

combined_dicts, attribute_value_mapping = generate_all_combinations_with_mapping(exp_role, variations)

print(attribute_value_mapping)