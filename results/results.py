# the necessary results.csv file is downloaded from the gsheet

from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import wandb
import pandas as pd

api = wandb.Api()

DIR = "results"
plt.rcParams["figure.figsize"] = [12, 8]
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
COLOR_MAP = {"Leading": "tab:blue", "Trailing": "tab:orange"}
MARKER_MAP = {"Finnish": "v", "English": "o", "Swahili": "s", "Chinese": "d"}
MAP_TOKENIZER = {"lead": "Leading", "trail": "Trailing"}
MAP_LANGUAGE_CODE = {
    "fi": "Finnish",
    "en": "English",
    "sw": "Swahili",
    "zh": "Chinese",
}
MORPHOLOGY_OF_LANGUAGES = {
    "en": 2,
    "fi": 2,
    "sw": -1,
    "zh": 0,
}


def load_wandb_results():
    runs = api.runs("ykmemara/deeplearning")
    results = []
    for run in runs:
        config = run.config
        metrics = run.summary_metrics
        # get loss curve
        loss = run.history()
        for _, row in loss.iterrows():
            results.append({**config, **metrics, **row})
    results = pd.DataFrame(results)
    results.to_csv("results/wandb_results.csv")


def get_wandb_results() -> pd.DataFrame:
    "access the wandb results from csv"
    results = pd.read_csv("results/wandb_results.csv")
    return results


def make_wandb_loss_curve(results, model_size="14m"):
    fig, ax = plt.subplots()
    step = "train/global_step"
    loss = "train/loss"  # train/loss or eval/loss
    # first agg over seeds
    data = (
        results[results["model_size"] == model_size]
        .groupby(["language", "tokenizer", "model_size", step])
        .agg({loss: ["mean", "std"]})
    )
    data = data.reset_index()
    # now plot
    for (lang, tokenizer, model), group in data.groupby(
        ["language", "tokenizer", "model_size"]
    ):
        x = group[step]
        y = group[(loss, "mean")]
        yerr = group[(loss, "std")]
        # get keys for language and tokenizer from MAP_LANGUAGE_CODE and MAP_TOKENIZER
        label = f"{MAP_LANGUAGE_CODE[lang]} ({MAP_TOKENIZER[tokenizer]})"
        ax.plot(x, y, label=label)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_xlim(0, 6000)
    ax.legend()
    # tight layout
    plt.tight_layout()
    plt.savefig(f"{DIR}/wandb_loss_curve_{model_size}.png")


def add_language_code(results):
    "Add a column with the language code"
    results["Language Code"] = results["Language"].map(MAP_LANGUAGE_CODE)
    return results


def add_morphology(results):
    "A language is either prefixing or suffixing"
    results["Morphology"] = results["Language Code"].map(MORPHOLOGY_OF_LANGUAGES)
    return results


def load_results():
    # get root directory
    dr = os.path.dirname(os.path.realpath(__file__))
    results = pd.read_csv(f"{dr}/results.csv")
    results = results.dropna(subset=[results.columns[0]])
    # Language,Tokenizer,Model Size,Seed,Eval Loss,Accuracy,F1,Perplexity,Top5 Accuracy,Top10 Accuracy,TTR,Avg Token Length
    results = add_language_code(results)
    results = add_morphology(results)
    return results


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


def mophology_vs_top10accuracy(results, model="14m"):
    data = results.dropna(subset=["Top10 Accuracy"])[results["Model Size"] == model]

    fig, ax = plt.subplots()

    for (lang, tokenizer), group in data.groupby(["Language", "Tokenizer"]):
        x = group["Top10 Accuracy"].mean()
        xerr = group["Top10 Accuracy"].std()
        y = group["Morphology"].values[0]
        sns.scatterplot(
            x=[x],
            y=[y],
            hue=[tokenizer],
            style=[lang],
            markers=MARKER_MAP,
            palette=COLOR_MAP,
            s=100,
            ax=ax,
            legend=False,
        )
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt=MARKER_MAP[lang],
            color=COLOR_MAP[tokenizer],
            markersize=10,
        )

    ax.set_yticks([-1, 0, 1, 2])
    ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])

    # Split the legend into two: one for languages and one for tokenizers
    language_handles = [
        plt.Line2D(
            [0], [0], marker=marker, color="w", markerfacecolor="k", markersize=10
        )
        for marker in MARKER_MAP.values()
    ]
    language_labels = list(MARKER_MAP.keys())
    tokenizer_handles = [
        plt.Line2D([0], [0], color=color) for color in COLOR_MAP.values()
    ]
    tokenizer_labels = list(COLOR_MAP.keys())

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
        bbox_to_anchor=(0.8, 1),
    )

    plt.ylabel("← Prefixing                                      Suffixing →")
    plt.xlabel("Top10 Accuracy")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{DIR}/morphology_vs_top10accuracy_{model}.png")


def compare_leading_vs_trailing(results, model="14m", metric="Top10 Accuracy"):
    data = results.dropna(subset=[metric])[results["Model Size"] == model]

    summary = []
    for lang, group in data.groupby("Language"):
        leading = group[group["Tokenizer"] == "Leading"][metric].mean()
        trailing = group[group["Tokenizer"] == "Trailing"][metric].mean()
        if pd.notna(leading) and pd.notna(trailing) and trailing != 0:
            diff_percent = ((leading - trailing) / trailing) * 100
            morphology = group["Morphology"].values[0]
            summary.append((lang, morphology, diff_percent))

    fig, ax = plt.subplots()
    langs, morphologies, diffs = zip(*summary)

    for lang, morphology, diff in summary:
        sns.scatterplot(
            x=[morphology],
            y=[diff],
            style=[lang],
            markers=MARKER_MAP,
            color="black",
            s=100,
            ax=ax,
            legend=False,
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("← Prefixing                                      Suffixing →")
    ax.set_ylabel("% Difference (Leading vs. Trailing)")
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_yticks([-30, -20, -10, 0, 10, 20])

    # Create a legend for languages
    language_handles = [
        plt.Line2D(
            [0], [0], marker=marker, color="w", markerfacecolor="k", markersize=10
        )
        for marker in MARKER_MAP.values()
    ]
    language_labels = list(MARKER_MAP.keys())

    ax.legend(
        language_handles,
        language_labels,
        title="Language",
        loc="upper right",
        bbox_to_anchor=(1, 1),
    )

    plt.tight_layout()
    suffix = metric.replace(" ", "_") if metric != "Top10 Accuracy" else ""
    plt.savefig(f"{DIR}/lead_vs_trail_{model}{suffix}.png")


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
    plt.savefig(f"{DIR}/boxplot_perplexity_14m.png")


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
    plt.savefig(f"{DIR}/boxplot_perplexity_60m.png")


def main():
    results = load_results()
    df = get_wandb_results()
    mophology_vs_top10accuracy(results, model="14m")
    mophology_vs_top10accuracy(results, model="60m")
    boxplots_14m_perplexity(results)
    boxplots_60m_perplexity(results)
    compare_leading_vs_trailing(results, model="14m")
    compare_leading_vs_trailing(results, model="60m")
    compare_leading_vs_trailing(results, model="14m", metric="Perplexity")
    compare_leading_vs_trailing(results, model="60m", metric="Perplexity")
    make_wandb_loss_curve(df, model_size="14m")
    make_wandb_loss_curve(df, model_size="60m")


if __name__ == "__main__":
    main()
