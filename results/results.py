# the necessary results.csv file is downloaded from the gsheet

import pandas as pd
import matplotlib.pyplot as plt
import os

DIR = "results"
plt.rcParams["figure.figsize"] = [15, 10]
plt.rcParams["font.size"] = 16
plt.color_sequences = "tab10"

morphology_description = {
    10: "Little or no inflectional morphology",
    2: "Predominantly suffixing",
    1: "Moderate preference for suffixing",
    0: "Approximately equal amounts of suffixing and prefixing",
    -1: "Moderate preference for prefixing",
    -2: "Predominantly prefixing",
}


def add_language_code(results):
    "Add a column with the language code"
    code = {
        "Finnish": "fi",
        "English": "en",
        "Swahili": "sw",
        "Chinese": "zh",
    }
    results["Language Code"] = results["Language"].map(code)
    return results


def add_morphology(results):
    "A language is either prefixing or suffixing"
    morphology = {
        "fi": 2,
        "en": 2,
        "sw": -1,
        "zh": 0,
    }
    results["Morphology"] = results["Language Code"].map(morphology)
    return results


def load_results():
    # get root directory
    dr = os.path.dirname(os.path.realpath(__file__))
    results = pd.read_csv(f"{dr}/results.csv")
    results = results.dropna(subset=[results.columns[0]])
    # Language,Tokenizer,Model Size,Seed,Eval Loss,Accuracy,F1,Perplexity,Top5 Accuracy,Top10 Accuracy,TTR,Avg Token Length
    # Finnish,Trailing,14m,9336,4.765193939,0.001906078923,0.001199148164,117.3538746,0.00589216469,0.01230434985,0.599566981,5.6199375
    # Finnish,Trailing,14m,42,4.223970413,0.001371014214,0.0009371384652,68.30414229,,,,
    results = add_language_code(results)
    results = add_morphology(results)
    return results


def avg_and_var_across_seeds(results, metric, groupby=None):
    "Returs a DataFrame with MultiIndex mean and standard deviation"
    if groupby is None:
        groupby = ["Language", "Tokenizer", "Model Size"]
    results = results.dropna(subset=[metric])
    grouped = results.groupby(groupby).agg({metric: ["mean", "std"]})
    grouped = grouped.reset_index()
    return grouped


def boxplots_14m_perplexity(results):
    fig, ax = plt.subplots()
    data = results[results["Model Size"] == "14m"]
    data = data.dropna(subset=["Perplexity"])
    data = data.sort_values("Language")
    data.boxplot(column="Perplexity", by="Language", ax=ax)
    plt.title("Perplexity for 14m model")
    plt.suptitle("")  # Remove the automatic 'Boxplot grouped by Language' title
    plt.ylabel("Perplexity")
    plt.xlabel("Language")
    # remove gridlines
    ax.grid(False)
    plt.savefig(f"{DIR}/boxplot_perplexity_14m.png")


if __name__ == "__main__":
    results = load_results()
    grouped = avg_and_var_across_seeds(results, "Eval Loss")
    boxplots_14m_perplexity(results)
