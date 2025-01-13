# the necessary results.csv file is downloaded from the gsheet

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

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
        "en": 2,
        "fi": 2,
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


def original_boxplots_14m_perplexity(results):
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


def mophology_vs_top10accuracy(results):
    model = "60m"
    data = results.dropna(subset=["Top10 Accuracy"])[results["Model Size"] == model]
    color_map = {"Leading": "tab:blue", "Trailing": "tab:orange"}
    marker_map = {"Finnish": "v", "English": "o", "Swahili": "s", "Chinese": "d"}
    fig, ax = plt.subplots()
    for (lang, tokenizer), group in data.groupby(["Language", "Tokenizer"]):
        y = group["Top10 Accuracy"].mean()
        yerr = group["Top10 Accuracy"].std()
        x = group["Morphology"].values[0]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=marker_map[lang],
            color=color_map[tokenizer],
            label=f"{lang} ({tokenizer})",
            markersize=10,
        )

    ax.set_xticks([-1, 0, 1, 2])
    ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
    # Split the legend into two: one for languages and one for tokenizers
    language_handles = [
        plt.Line2D(
            [0], [0], marker=marker, color="w", markerfacecolor="k", markersize=10
        )
        for marker in marker_map.values()
    ]
    language_labels = list(marker_map.keys())
    tokenizer_handles = [
        plt.Line2D([0], [0], color=color) for color in color_map.values()
    ]
    tokenizer_labels = list(color_map.keys())

    first_legend = ax.legend(
        language_handles,
        language_labels,
        title="Language",
        loc="upper right",
        bbox_to_anchor=(1, 1),
    )
    ax.add_artist(first_legend)
    ax.legend(
        tokenizer_handles,
        tokenizer_labels,
        title="Tokenizer",
        loc="upper right",
        bbox_to_anchor=(0.85, 1),
    )

    plt.xlabel("← Prefixing                                      Suffixing →")
    plt.ylabel("Top10 Accuracy")
    plt.title(f"Top10 Accuracy vs Morphology for {model} model")
    plt.grid(False)
    plt.savefig(f"{DIR}/morphology_vs_top10accuracy_{model}.png")


def boxplots_14m_perplexity(results):
    # Filter for 14m model size and drop missing values
    data = results[results["Model Size"] == "14m"].dropna(subset=["Perplexity"])

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Create a boxplot with Seaborn for better flexibility
    sns.boxplot(
        data=data,
        x="Language",
        y="Perplexity",
        hue="Tokenizer",
        dodge=True,  # Separate bars for each tokenizer
        medianprops=dict(color="black", linewidth=1.5),
    )

    # Rotate the tokenizer labels for readability
    plt.xticks(rotation=0)
    plt.title("")
    plt.ylabel("Perplexity")
    plt.xlabel("Language")

    # Format legend for better readability
    plt.legend(title="Tokenizer", loc="upper left")

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig("boxplot_perplexity_14m.png")
    plt.show()


def boxplots_60m_perplexity(results):
    # Filter for 14m model size and drop missing values
    data = results[results["Model Size"] == "60m"].dropna(subset=["Perplexity"])

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Create a boxplot with Seaborn for better flexibility
    sns.boxplot(
        data=data,
        x="Language",
        y="Perplexity",
        hue="Tokenizer",
        dodge=True,  # Separate bars for each tokenizer
        medianprops=dict(color="black", linewidth=1.5),
    )

    # Rotate the tokenizer labels for readability
    plt.xticks(rotation=0)
    plt.title("")
    plt.ylabel("Perplexity")
    plt.xlabel("Language")

    # Format legend for better readability
    plt.legend(title="Tokenizer", loc="upper left")

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig("boxplot_perplexity_60m.png")
    plt.show()


if __name__ == "__main__":
    results = load_results()
    # ['Language', 'Tokenizer', 'Model Size', 'Seed', 'Eval Loss', 'Accuracy',
    #  'F1', 'Perplexity', 'Top5 Accuracy', 'Top10 Accuracy', 'TTR',
    #  'Avg Token Length', 'Language Code', 'Morphology']
    grouped = avg_and_var_across_seeds(results, "Eval Loss")
    mophology_vs_top10accuracy(results)
    boxplots_14m_perplexity(results)
    boxplots_60m_perplexity(results)
