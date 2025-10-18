import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_scree_plot(pipeline):

    fig, ax = plt.subplots()

    components = np.arange(1, pipeline["pca"].n_components_ + 1)

    ax.plot(
        components,
        np.cumsum(pipeline["pca"].explained_variance_ratio_)
    )

    ax.bar(
        components,
        pipeline["pca"].explained_variance_ratio_,
    )

    ax.axhline(0.80, linestyle="--", label="80% da variância explicada", color="red")
    ax.axhline(0.95, linestyle="--", label="95% da variância explicada", color="green")
    ax.axhline(0.99, linestyle="--", label="99% da variância explicada", color="purple")

    ax.set_xlabel("Componentes Principais")
    ax.set_ylabel("Variância Explicada")
    ax.set_title("Scree Plot")

    plt.show()

def plot_comparing_metrics(df_results):

    fig, axs = plt.subplots(4, 2, figsize=(9, 9), sharex=True)

    comparing_metrics = [
        "time_seconds",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_roc_auc",
        "test_average_precision",
    ]

    metric_names = [
        "Tempo (s)",
        "Acurácia",
        "Acurácia balanceada",
        "F1",
        "Precisão",
        "Recall",
        "AUROC",
        "AUPRC",
    ]

    for ax, metric, name in zip(axs.flatten(), comparing_metrics, metric_names):
        
        sns.boxplot(
            df_results,
            x="model",
            y=metric,
            ax=ax,
            showmeans=True,
        )

        ax.set_title(name)
        ax.set_ylabel(name)
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.show()