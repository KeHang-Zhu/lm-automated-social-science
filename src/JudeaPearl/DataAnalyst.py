import json
import os
from typing import List, Dict, Union, Any
import pandas as pd
import numpy as np
import subprocess


class DataAnalyst:
    def __init__(
        self,
        final_df: pd.DataFrame,
        meta_data: Dict[str, Any],
        variable_mapping: Dict[str, str],
        edge_dict: Dict[str, List[str]],
        interaction_data: Dict[str, Dict],
        scm_simple: Dict[str, Dict],
    ):
        """
        Args:
            final_df (pd.DataFrame): The final dataframe to use with all normal + interaction colums
            variable_mapping (Dict[str, str]): The mapping of the variables short names to their full names
            edge_dict (Dict[str, List[str]]): The edge dictionary
            interaction_data (Dict[str, Dict]): The interaction data from the full results
            scm_simple (Dict[str, Dict]): A simple dictionary version of StructuralCausalModelBuilder made fromt the scm_to_json function in that class
                (no need to understand this class, just explaining format for future self)
        """

        # making sure there's no unnamed column
        self.final_df = final_df.drop("Unnamed: 0", axis=1, errors="ignore")
        self.final_mapped_df = self.final_df.rename(columns=variable_mapping)
        self.meta_data = meta_data
        self.variable_mapping = variable_mapping
        self.reversed_variable_mapping = {
            value: key for key, value in self.variable_mapping.items()
        }
        # final edge dict takes into account nominal variables (not working yet though)
        self.final_edge_dict = edge_dict
        self.scm_dict = self._initialize_scm(scm_simple)
        self.scenario_description = scm_simple["Variable1"]["scenario_description"]

    def _initialize_scm(self, scm_simple: Dict) -> Dict[str, Dict]:
        """
        Initialize the SCM based on the provided interaction data.

        Args:
            interaction_data (Dict[str, Dict]): Data related to SCM interactions.
        """

        scm_dict = {}
        for key, value in scm_simple.items():
            scm_dict[scm_simple[key]["name"]] = scm_simple[key]

        self.agents_in_scenario = scm_dict[scm_simple[key]["name"]][
            "agents_in_scenario"
        ]
        self.scenario_description = (
            scm_dict[scm_simple[key]["name"]]["scenario_description"]
            .replace("$100", "100 dollars")
            .lower()
        )

        return scm_dict

    def generate_sem_syntax(self, interaction=False):
        """
        Creates a string of the semopy syntax for the data analyst to use.
        Only handles depth 2 graphs with one endogenous variable.

        Args:
            interaction (bool): Whether to include interaction terms in the syntax.
        """
        rhs = ""
        mapped_outcome = self.variable_mapping[self.final_df.columns.values[0]]
        for cause in self.final_df.columns.values[1:]:
            mapped_cause = self.variable_mapping[cause]

            # Handle interaction terms
            if interaction:
                rhs += f"{mapped_cause} +"
            elif "_x_" not in cause:
                rhs += f"{mapped_cause} +"

        lhs = f"{mapped_outcome} ~"
        syntax = lhs + rhs + "1"
        return syntax

    def estimate_sem(
        self,
        data_path: str,
        sem_syntax: str,
        std_estimates: bool = False,
        interaction: bool = False,
    ):
        """
        Runs the R script to estimate the sem model using lavaan.
        Must be in the same directory as the R script.

        Args:
            data_path (str): The path to the data file
            sem_syntax (str): The sem syntax to use
            std_estimates (bool): Whether to standardize the estimates
        """
        # Get the directory of the current file
        # Get the path of the current file
        current_file_path = os.path.abspath(__file__)
        script_path = os.path.join(os.path.dirname(current_file_path), "estimate_sem.R")
        # pass the boolean as a string
        std_estimates = str(std_estimates).lower()
        interaction = str(interaction).lower()
        command = [
            "Rscript",
            script_path,
            data_path,
            sem_syntax,
            std_estimates,
            interaction,
        ]
        subprocess.run(command)

    def format_summary_stats(self, variable, dataframe, decimal_places=2):
        """
        Formats the summary statistics for a variable in the dataframe.
        Args:
            variable (str): The variable to format
            dataframe (pd.DataFrame): The dataframe to use
            decimal_places (int): The number of decimal places to round to
        """
        # Format the describe() output
        stats = dataframe[variable].describe().round(decimal_places)
        # Drop the count row, unnecessary
        stats = stats.drop("count")
        formatted_stats = ", ".join(
            [f"{stat}: {value}" for stat, value in stats.items()]
        )

        return formatted_stats

    def var_info_to_latex(self):
        """
        Generates a string of the variable information for the tables that accompany the SCM.
        """
        figure_latex = ""
        #   figure_latex += f' summary statistics: {self.final_df[variable].describe()}\\\\\n'
        for variable in self.scm_dict.keys():
            mapping = self.variable_mapping[variable]
            if self.scm_dict[variable]["__class__"] == "EndogenousVariable":
                if self.scm_dict[variable]["variable_type"] == "ordinal":
                    units = f'The units of this variable are an index from 1 to {len(self.scm_dict[variable]["levels"])}'
                else:
                    units = self.scm_dict[variable]["units"]
                formatted_stats = self.format_summary_stats(variable, self.final_df)
                figure_latex += f"""
\\begin{{tabularx}}{{\\textwidth}}{{|c|c|X|}}
\hline

\multirow{{4}}{{*}}{{\parbox{{5cm}} {{ \centering {variable} ({mapping})}}}} & Variable Type & {self.scm_dict[variable]['variable_type']}  \\\\ \cline{{2-3}}
                        &  Units   &  {units}   \\\\ \cline{{2-3}}
                        & Levels & {self.scm_dict[variable]["levels"]} \\\\ \cline{{2-3}}
                        & Summary Statistics   & {formatted_stats}    \\\\ \hline
\end{{tabularx}}\\\\
                """

            else:
                if self.scm_dict[variable]["variable_type"] == "ordinal":
                    units = f'The units of this variable are an index from 1 to {len(self.scm_dict[variable]["levels"])}'
                else:
                    units = self.scm_dict[variable]["units"]
                figure_latex += f"""

\\begin{{tabularx}}{{\\textwidth}}{{|c|c|X|}}
\hline
\multirow{{5}}{{*}}{{\parbox{{5cm}} {{ \centering {variable} \\\\({mapping})}}}} & Variable Type & {self.scm_dict[variable]['variable_type']}  \\\\ \cline{{2-3}}
                        & Proxy Attribute Name   & {self.scm_dict[variable]['attribute_variation']['attribute_name']}   \\\\ \cline{{2-3}}
                        & Varied Attribute Levels ({len(self.scm_dict[variable]['attribute_variation']['attribute_values'])})  & {self.scm_dict[variable]['attribute_variation']['attribute_values']}   \\\\ \cline{{2-3}}
                        & Units  & {units}   \\\\ \cline{{2-3}}
                        & Relevant Agent   & {self.scm_dict[variable]['attribute_variation']['varied_agent']}   \\\\ \hline
\end{{tabularx}}\\\\"""

        return (
            figure_latex.replace("$", "\$")
            .replace("\\begin", "\n\\begin")
            .replace("%", "\%")
        )

    def lavaan_to_tikz(self, file_path: str):
        """
        Generates a TikZ picture for a semopy model, with the outcome above and predictors spread out horizontally.
        Updated to use the new TikZ style with fading lines behind the labels.

        :param df: DataFrame with columns ['lhs', 'op', 'rhs', 'est', 'se', ...]
        :return: A string containing the TikZ code.
        """
        # Header with the required TikZ libraries and settings
        tikz_output = """\\begin{tikzpicture}
    \\pgfdeclarelayer{background}
    \\pgfdeclarelayer{foreground}
    \\pgfsetlayers{background,main,foreground}
    """
        df = pd.read_csv(file_path)

        # Identify the main outcome variable (lhs)
        main_var = df["lhs"].iloc[0]
        tikz_output += f"\\node[align=center] ({main_var}) at (0,0) {{{main_var}}};\n"

        predictors = set(df[df["op"] == "~"]["rhs"]) - {"1"}
        angle_increment = 360 / len(predictors)
        angle = 0
        radius = 6  # Adjust radius to your preference

        for var in predictors:
            var_vame = var
            # add a line break \\ to main variable if the symbol -x- is in the variable name
            if "_x_" in var:
                var_vame = var.split("_x_")
                var_vame = var_vame[0] + "\\\\-x-\\\\" + var_vame[1]

            # adding mean and variance to the variable name
            full_var_name = self.reversed_variable_mapping[var]
            mean = np.round(np.mean(self.final_df[full_var_name]), 3)
            variance = np.round(np.var(self.final_df[full_var_name]), 3)
            var_vame += f"\\\\ $\mu = {mean}$ \\\\ $\sigma^2 = {variance}$"

            tikz_output += f"\\node[align=center] ({var}) at ({angle}:{radius}cm) {{{var_vame}}};\n"
            angle += angle_increment

        # Draw edges on the background layer
        tikz_output += "\\begin{pgfonlayer}{background}\n"
        for _, row in df.iterrows():
            if row["op"] == "~" and row["rhs"] != "1":
                tikz_output += (
                    f"\\draw[->,red,very thick] ({row['rhs']}) -- ({row['lhs']});\n"
                )
        tikz_output += "\\end{pgfonlayer}\n"

        # Draw labels on the foreground layer
        tikz_output += "\\begin{pgfonlayer}{foreground}\n"
        for _, row in df.iterrows():
            if row["op"] == "~" and row["rhs"] != "1":
                pval = f"p={row['pvalue']:.3f}"
                # if p-value is below 0 to 3 decimal places, make it p<0.001
                if row["pvalue"] < 0.001:
                    pval = "$p<0.001$"
                # move standard error back if the estimate is negative
                estimate = row["est"]
                if estimate < 0:
                    label = f"{estimate:.3f}\\\\\\phantom{{-}}({row['se']:.3f})\\\\\\phantom{{-}} {pval} "
                else:
                    label = f"{estimate:.3f}\\\\({row['se']:.3f})\\\\ {pval}"
                tikz_output += f"\\path ({row['rhs']}) -- ({row['lhs']}) node[midway, fill=white, font=\\sffamily\\small, align=center] {{{label}}};\n"
        tikz_output += "\\end{pgfonlayer}\n"

        # End of TikZ picture and document
        tikz_output += "\\end{tikzpicture}"

        return tikz_output

    def generate_latex_figure(
        self, var_info: str, tikz_pic: str, scenario_name: str, interaction=False
    ):
        """
        Generates the final latex string for the figure

        Args:
            var_info (str): The variable information string
            tikz_pic (str): The tikz picture string
        """
        figure_latex = (
            f"""
\\documentclass[12pt]{{article}}
\\usepackage{{geometry}}
\\usepackage{{pgfplots}}
\\usepackage{{tabularx}}
\\usepackage{{array}}
\\usepackage{{multirow}}
\\usetikzlibrary{{arrows,positioning,automata,arrows.meta, shapes.geometric,decorations.pathreplacing, calc}}
\\usepackage{{tikz}}
\\geometry{{top=1in, bottom=1in, left=1in, right=1in}}
\\setlength{{\\parindent}}{{0pt}}

\\definecolor{{headercolor}}{{RGB}}{{142, 180, 227}} 
\\definecolor{{rowcolor}}{{RGB}}{{230, 240, 255}}  

\\newcolumntype{{L}}{{>{{\\raggedright\\arraybackslash}}p{{0.5\\textwidth}}}}
\\newcolumntype{{R}}{{>{{\\raggedleft\\arraybackslash}}p{{0.5\\textwidth}}}}

\\begin{{document}}
\\begin{{figure}}[ht]
\caption{{Fitted SEM for {self.scenario_description}.}}
\\centering
\\textbf{{Simulations Run}}: {self.final_df.shape[0]}\\\\
\\textbf{{Agents}}: {[agent for agent in self.agents_in_scenario]}\\\\
""".replace(
                "_", "-"
            )
            .replace("$", "\$")
            .replace("\\begin", "\n\\begin")
            .replace("%", "\%")
        )

        if not interaction:
            figure_latex += f"""
{tikz_pic}\\
  \\begin{{minipage}}{{\\textwidth}}
    \\begin{{footnotesize}}
      \\emph{{Notes:}} Each exogenous cause is given with its mean and variance. 
       The edges are labeled with their unstandardized path estimate, standard error, and p-value. 
    \\end{{footnotesize}}
    \\end{{minipage}}
\\end{{figure}}
\end{{document}}
""".replace(
                "_", "-"
            )
        else:
            figure_latex += f"""
{tikz_pic}\\
  \\begin{{minipage}}{{\\textwidth}}
    \\begin{{footnotesize}}
      \\emph{{Notes:}} The proxy attribute and one of its values are the directly provided to the relevant agent in each simulations.
      The number of simulations is all possible combinations of the varied attribute values for the causal variables.
    \\end{{footnotesize}}
    \\end{{minipage}}
\end{{figure}}
\end{{document}}
""".replace(
                "_", "-"
            )

        figure_latex = self.fix_latex_quotes(figure_latex)

        return figure_latex

    def generate_latex_table(self, var_info: str, tikz_pic: str, scenario_name: str):
        """
        Generates the final latex string for the figure

        Args:
            var_info (str): The variable information string
            tikz_pic (str): The tikz picture string
        """
        table_latex = f"""
\\documentclass[12pt]{{article}}
\\usepackage{{geometry}}
\\usepackage{{pgfplots}}
\\usepackage{{tabularx}}
\\usepackage{{array}}
\\usepackage{{multirow}}
\\usetikzlibrary{{arrows,positioning,automata,arrows.meta, shapes.geometric,decorations.pathreplacing, calc}}
\\usepackage{{tikz}}
\\geometry{{top=1in, bottom=1in, left=1in, right=1in}}
\\setlength{{\\parindent}}{{0pt}}

\\definecolor{{headercolor}}{{RGB}}{{142, 180, 227}} 
\\definecolor{{rowcolor}}{{RGB}}{{230, 240, 255}} 

\\newcolumntype{{L}}{{>{{\\raggedright\\arraybackslash}}p{{0.5\\textwidth}}}}
\\newcolumntype{{R}}{{>{{\\raggedleft\\arraybackslash}}p{{0.5\\textwidth}}}}

\\begin{{document}}
\\begin{{table}}[ht]
\\centering
\caption{{Operationalizations of the variables for the experiment simulated to generate the SCM for {self.scenario_description}.}}
\\{var_info}
    \\begin{{minipage}}{{\\textwidth}}
    \\begin{{footnotesize}}
      \\emph{{Notes:}} The proxy attribute and one of its values are directly provided to the relevant agent in each simulation.
      The number of simulations is all possible combinations of the varied attribute values for the causal variables.
    \\end{{footnotesize}}
    \\end{{minipage}}
\\end{{table}}
\end{{document}}
""".replace(
            "_", "-"
        )

        table_latex = self.fix_latex_quotes(table_latex)

        return table_latex

    def fix_latex_quotes(self, latex_string):
        """
        Fixes the quotes in the latex string to be the correct type.

        Args:
            latex_string (str): The latex string to fix
        """
        fixed_latex = ""
        n = len(latex_string)

        for i, char in enumerate(latex_string):
            if char == "'":
                prev_char = latex_string[i - 1] if i > 0 else None
                next_char = latex_string[i + 1] if i < n - 1 else None

                # Check for apostrophe (e.g., in "don't")
                if (
                    prev_char
                    and next_char
                    and prev_char.isalnum()
                    and next_char.isalnum()
                ):
                    fixed_latex += "'"
                # Check for opening quote
                elif not prev_char or not prev_char.isalnum():
                    fixed_latex += "`"
                # Otherwise, treat as closing quote
                else:
                    fixed_latex += "'"
            else:
                fixed_latex += char

        return fixed_latex

    def analyze_data(
        self,
        data_dir: str,
        final_output_dir: str,
        interaction=False,
        std_estimates=False,
    ):
        """
        Runs the analysis on the data and outputs the latex file for the plots

        Args:
            data_dir (str): The directory of the data
            final_output_dir (str): The directory to writeup
            interaction (bool): Whether to include interaction terms in the syntax
            std_estimates (bool): Whether to standardize the estimates
        """
        syntax = self.generate_sem_syntax(interaction=interaction)
        self.estimate_sem(
            data_dir, syntax, std_estimates=std_estimates, interaction=interaction
        )
        var_info = self.var_info_to_latex()
        # read in the estimates_df.csv as it's created by the R script, saved in the folder with the data/
        tag = ""
        if interaction:
            tag += "_interaction"
        if std_estimates:
            tag += "_std"
        tikz_pic = self.lavaan_to_tikz(os.path.join(data_dir, f"estimates{tag}_df.csv"))
        scenario_name = self.scenario_description
        if interaction:
            fig_file_name = scenario_name + "_scm_interaction.tex"
            latex_string = self.generate_latex_figure(
                var_info,
                tikz_pic,
                scenario_name + "_interaction",
                interaction=interaction,
            )
        else:
            fig_file_name = scenario_name + "_scm.tex"
            table_file_name = scenario_name + "_table.tex"
            latex_string = self.generate_latex_figure(var_info, tikz_pic, scenario_name)
            table_latex = self.generate_latex_table(var_info, tikz_pic, scenario_name)
            with open(os.path.join(final_output_dir, table_file_name), "w") as file:
                file.write(table_latex)
        with open(os.path.join(final_output_dir, fig_file_name), "w") as file:
            file.write(latex_string)


# NOTE THIS WILL FAIL IF RUN HERE!!!!!!
if __name__ == "__main__":
    # others that work but are less interesting/small sample 'mug_found''mug_1', auction_art_2vars
    dir_list = ["auction_art_3vars", "mug_love", "lawyer_interview_3var", "tax_fraud"]
    dir_list = ["mug_love"]
    path = "/Users/benjaminmanning/Desktop/rs/data/"
    for dir in dir_list:
        directory_path = path + dir + "/"
        data = pd.read_csv(directory_path + "data.csv")
        with open(os.path.join(directory_path, "meta_data.json"), "r") as f:
            meta_data = json.load(f)
        with open(os.path.join(directory_path, "result.json"), "r") as f:
            interaction_data = json.load(f)
        with open(os.path.join(directory_path, "final_mapping.json"), "r") as f:
            final_mapping = json.load(f)
        with open(os.path.join(directory_path, "final_edge_dict.json"), "r") as f:
            final_edge_dict = json.load(f)
        with open(os.path.join(directory_path, "scm_simple.json"), "r") as f:
            scm_simple = json.load(f)

        data_analyst = DataAnalyst(
            data,
            meta_data,
            final_mapping,
            final_edge_dict,
            interaction_data,
            scm_simple,
        )
        print("###########" + dir + "###########\n")

        if True:
            data_analyst.analyze_data(
                directory_path,
                final_output_dir=directory_path,
                interaction=False,
                std_estimates=False,
            )

        if False:
            syntax = data_analyst.generate_sem_syntax(interaction=False)
            data_analyst.estimate_sem(directory_path, syntax, std_estimates=False)
            var_info = data_analyst.var_info_to_latex()
            tikz_pic = data_analyst.lavaan_to_tikz(directory_path + "estimates_df.csv")
            latex_string = data_analyst.generate_latex_figure(var_info, tikz_pic)
            with open(directory_path + "output.tex", "w") as file:
                file.write(latex_string)

        # data_analyst.analyze_data('/Users/benjaminmanning/Desktop/rs/writeup/plots/', dir, interaction = False)
        # data_analyst.analyze_data('/Users/benjaminmanning/Desktop/rs/writeup/plots/', dir, interaction = True)
