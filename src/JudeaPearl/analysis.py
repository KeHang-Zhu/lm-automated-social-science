import os
import json
import pandas as pd
import sys

sys.path.append("../LLM")
from DataParser import DataParser
from DataCleaner import DataCleaner
from DataAnalyst import DataAnalyst
from LLM import LanguageModel

### setup
if __name__ == "__main__":
    # YOU SHOULD KEEP TERMERATURE TO 0.1 HERE, WE WANT HIGH PRECISION ON ANSWERS
    LLM = LanguageModel(family="openai", model="gpt-4", temperature=0.1)
    directory_path = "/Users/benjaminmanning/Desktop/test2/"
    file_name = "result.json"
    file_path = os.path.join(directory_path, file_name)
    with open(file_path, "r") as f:
        interaction_data = json.load(f)

    # data parser

    data_parser = DataParser(interaction_data)
    data_parser.add_LLM(LLM)
    data_parser.backend_clean_data(directory_path)

    # data cleaner
    raw_data = pd.read_csv(directory_path + "raw_data.csv")
    with open(directory_path + "meta_data.json", "r") as f:
        meta_data = json.load(f)

    data_cleaner = DataCleaner(interaction_data, raw_data, meta_data)
    data_cleaner.generate_final_df()
    data_cleaner.save_data(directory_path)

    # data analyst
    data = pd.read_csv(directory_path + "data.csv")
    with open(directory_path + "meta_data.json", "r") as f:
        meta_data = json.load(f)
    with open(directory_path + "result.json", "r") as f:
        interaction_data = json.load(f)
    with open(directory_path + "final_mapping.json", "r") as f:
        final_mapping = json.load(f)
    with open(directory_path + "final_edge_dict.json", "r") as f:
        final_edge_dict = json.load(f)
    with open(directory_path + "scm_simple.json", "r") as f:
        scm_simple = json.load(f)

    data_analyst = DataAnalyst(
        data,
        meta_data,
        final_mapping,
        final_edge_dict,
        interaction_data,
        scm_simple,
    )

    data_analyst.analyze_data(
        directory_path,
        final_output_dir=directory_path,
        interaction=False,
        std_estimates=False,
    )
