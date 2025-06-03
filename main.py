import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from palmerpenguins import load_penguins
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from typing import Tuple, Dict, List

# -- Loading the penguins' dataset,
# -- Basic normalisation via dropping NaN rows,
# -- Pseudo-randomisation of dataset
penguins = load_penguins()
penguins.dropna(inplace=True)
penguins = penguins.sample(frac=1)


def island(n: str) -> int:
    """
    Takes an island from the penguins dataset and assigns it
    an integer in-place for identity.
    :param n: str -> island
    :return: int -> island_identifier
    """

    if n == "Torgersen":
        return 1
    elif n == "Dream":
        return 2
    else:
        return 3


def species(n: str) -> int:
    """
    Takes a species from the penguins dataset and assigns it
    an integer in-place for identity.
    :param n: str -> species
    :return: int -> species_identifier
    """

    if n == "Adelie":
        return 1
    elif n == "Gentoo":
        return 2
    else:
        return 3


# -- Application of identity conversion functions and an inline lambda for sex.
penguins["species"] = penguins["species"].apply(species)
penguins["island"] = penguins["island"].apply(island)
penguins["sex"] = penguins["sex"].apply(lambda n: 1 if n == "male" else 2)
training, testing = [], []


def transmute_data(table: pd.DataFrame) -> List:
    """
    Takes the penguin table and transmutes it into a more linear
    list, allowing for easier access and iteration of data.
    :param table: pd.DataFrame -> penguin_table
    :return: List -> penguin_list
    """

    return [(island(v[1][1]), v[1][2], v[1][3], v[1][4], v[1][5], 1 if v[1][6] == "male" else 2, v[1][7])
            for v in table.iterrows()], [v[1][0] for v in table.iterrows()]


def create_PCA(bar=False, scatter=False, loading=True) -> None:
    """
    Create a PCA via StandardScaler, this includes optional charting and
    normalisation options.
    :param bar: bar_graph -> creates a bar graph plot for PCA
    :param scatter: scatter_graph -> creates a scatter graph of closely / distantly related attributes
    :param loading: drops -> drop redundant values
    :return: None
    """

    scaled = StandardScaler().fit_transform(penguins.T)

    pca = PCA()
    pca.fit(scaled)
    pca_data = pca.transform(scaled)

    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = [f"PC{x}" for x in range(1, len(per_var) + 1)]

    if bar:
        plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
        plt.ylabel("Percentage of Variance")
        plt.xlabel("Principal Component")
        plt.title("PCA Values for Penguins")
        plt.show()

    pca_df = pd.DataFrame(pca_data, index=[*penguins.columns.tolist()], columns=labels)

    if scatter:
        plt.scatter(pca_df.PC1, pca_df.PC2)
        plt.title("PCA Scatter for Penguins")
        plt.xlabel(f"PC1 - {per_var[0]}%")
        plt.ylabel(f"PC2 - {per_var[1]}%")

        for s in pca_df.index:
            plt.annotate(s, (pca_df.PC1.loc[s], pca_df.PC2.loc[s]))

        plt.show()

    if loading:
        loading_scores = pd.Series(pca.components_[0], index=penguins.index.tolist())
        srted = loading_scores.abs().sort_values(ascending=False)

        top = srted[0:10].index.values

        penguins.drop(srted[0:10].index.tolist(), inplace=True)

        print(loading_scores[top])


def ten_fold_k(k: int = 5, n: int = 1) -> Tuple[float, float, Tuple]:
    """
    Creates training splits and data for a kNN classifier at specified folds
    and neighbours, a summation of this data is then outputted to console.
    :param k: neighbours -> amount of neighbours for kNN
    :param n: fold -> determines train-test split
    :return: errors, model -> split errors and a model containing all relevant information
    """

    x_train, x_test, y_train, y_test = train_test_split(tbl, names, test_size=0.1 * n, random_state=0)

    kNN = KNeighborsClassifier(n_neighbors=k)
    kNN.fit(x_train, y_train)

    prediction = kNN.predict(x_test)
    probability = kNN.predict_proba(x_test)

    fpr, tpr, thresholds = roc_curve(y_test, probability[:, 1], pos_label=2)
    r_auc = roc_auc_score(y_test, probability, multi_class="ovr")

    test_err, train_err = kNN.score(x_test, y_test), kNN.score(x_train, y_train)
    p_score = precision_score(y_test, prediction, average='macro')
    f_score = f1_score(y_test, prediction, average='macro')
    r_score = recall_score(y_test, prediction, average='macro')

    print(f"k={k}, n={n}\nTraining: {len(x_train)}, Testing: {len(x_test)} -> "
          f"Incorrect: {(r := (y_test != prediction).sum())}\n"
          f"Accuracy ( Training Errors ): {train_err}\n"
          f"Accuracy ( Testing Errors ): {test_err}\n"
          f"Precision: {p_score}\n"
          f"F1: {f_score}\n"
          f"Recall: {r_score}\n"
          f"FPR: {fpr}\n"
          f"TPR: {tpr}\n"
          f"ROC_AUC: {r_auc}\n")

    return test_err, train_err, (
    test_err, train_err, p_score, f_score, r_score, r_auc, fpr, tpr, y_test, prediction, k, n)


def ten_fold_g(k: int = 1) -> Tuple[float, float, Tuple]:
    """
    Creates training splits and data for a gNB classifier at specified folds,
    a summation of this data is then outputted to console.
    :param k: fold -> determines train-test split
    :return: errors, model -> split errors and a model containing all relevant information
    """

    x_train, x_test, y_train, y_test = train_test_split(tbl, names, test_size=0.1 * k, random_state=0)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    prediction = gnb.predict(x_test)
    probability = gnb.predict_proba(x_test)
    rp_probs = np.array([np.array([0, 1, 0]) for _ in range(len(y_test))])

    fpr, tpr, thresholds = roc_curve(y_test, probability[:, 1], pos_label=2)
    r_fpr, r_tpr, thresholds = roc_curve(y_test, rp_probs[:, 1], pos_label=2)
    r_auc = roc_auc_score(y_test, probability, multi_class="ovr")
    rp_auc = roc_auc_score(y_test, rp_probs, multi_class="ovr")

    test_err, train_err = gnb.score(x_test, y_test), gnb.score(x_train, y_train)
    p_score = precision_score(y_test, prediction, average='macro')
    f_score = f1_score(y_test, prediction, average='macro')
    r_score = recall_score(y_test, prediction, average='macro')

    print(f"Training: {len(x_train)}, Testing {len(x_test)} -> Incorrect: {(r := (y_test != prediction).sum())}\n"
          f"Accuracy ( Training Errors ): {train_err}\n"
          f"Accuracy ( Testing Errors ): {test_err}\n"
          f"Precision: {p_score}\n"
          f"F1: {f_score}\n"
          f"Recall: {recall_score(y_test, prediction, average='macro')}\n"
          f"FPR: {fpr}\n"
          f"TPR: {tpr}\n"
          f"ROC_AUC: {r_auc}\n")

    return test_err, train_err, (test_err, train_err, p_score, f_score, r_score, r_auc, fpr, tpr, y_test, prediction, k)


def obtain_knn_errors() -> Tuple[np.ndarray, List, List, List, Tuple]:
    test_dict, train_dict, models = {}, {}, []

    for i in range(1, 10):
        test_dict[i], train_dict[i] = {}, {}
        for v in range(1, 10):
            result = ten_fold_k(v, i)
            test_dict[i][v] = result[0]
            train_dict[i][v] = result[1]
            models.append(result[2])

    models.sort(reverse=True)

    return np.arange(1.0, 10.0, 1.0), \
           [list(test_dict[t].values()) for t in test_dict], \
           [list(train_dict[t].values()) for t in train_dict], \
           models, \
           sorted([(sum(test_dict[j].values()), j) for j in test_dict])[~0]


def plot_knn_errors(bot: np.ndarray, test: List, train: List) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(9):
        ax[0].plot(bot, test[i], marker='.', label=f"k={i + 1}")
        ax[1].plot(bot, train[i], marker='.', label=f"k={i + 1}")

    ax[0].legend()
    ax[1].legend()

    fig.supxlabel("Test sizes (%)")
    fig.supylabel("Accuracy")

    fig.suptitle("kNN Training & Testing Accuracy")
    ax[0].set_title("Training accuracy")
    ax[1].set_title("Testing accuracy")
    plt.show()


def multi_plot_knn(table: List, rev=False) -> None:
    """
    Plots CM, CV and ROC for most promising kNN model unless rev flagged is raised,
    in which case plot the least promising model.
    :param table: data -> a list containing all kNN models generated from obtain_knn_errors
    :param rev: worst -> if True plot the least promising model
    :return: None
    """

    for i in table:
        if i[~1] > 2:
            best = i
            break

    cv_result = cross_validate(KNeighborsClassifier(n_neighbors=best[~1]), tbl, names, cv=10, return_train_score=True)
    rp_probs = np.array([np.array([0, 1, 0]) for _ in range(len(best[~3]))])
    r_fpr, r_tpr, thresholds = roc_curve(best[~3], rp_probs[:, 1], pos_label=2)
    rp_auc = roc_auc_score(best[~3], rp_probs, multi_class="ovr")

    cm = confusion_matrix(best[~3], best[~2])
    df_cm = pd.DataFrame(cm, ["Adelie", "Gentoo", "Chinstrap"], ["Adelie", "Gentoo", "Chinstrap"])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.show()

    bot = np.arange(0.0, 10.0, 1.0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(bot, cv_result['fit_time'], marker='.', label="Fit time")
    ax[0].plot(bot, cv_result['score_time'], marker='.', label="Score time")
    ax[1].plot(bot, cv_result['train_score'], marker='.', label="Training score")
    ax[1].plot(bot, cv_result['test_score'], marker='.', label="Testing score")
    fig.supxlabel("Test sizes (%)")

    ax[0].legend()
    ax[1].legend()

    ax[0].set_title("Timings")
    ax[1].set_title("Accuracy")

    fig.suptitle(f"kNN (n={best[~0]}, k={best[~1]})")
    plt.show()

    plt.plot(r_fpr, r_tpr, linestyle='--', label=f"Random prediction, AUROC={rp_auc}")
    plt.plot(best[~5], best[~4], marker='.', label=f"kNN (n={best[~0]}, k={best[~1]}), AUROC={best[~6]}")
    plt.title("kNN ROC Plot")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


def obtain_gnb_errors() -> Tuple[np.ndarray, List, List, List]:
    test_dict, train_dict, models = {}, {}, []

    for i in range(1, 10):
        result = ten_fold_g(i)
        test_dict[i] = result[0]
        train_dict[i] = result[1]
        models.append(result[2])

    return np.arange(1.0, 10.0, 1.0), \
           list(test_dict.values()), \
           list(train_dict.values()), \
           models


def plot_gnb_errors(bot: np.ndarray, test: List, train: List) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].bar(bot, test)
    ax[1].bar(bot, train)

    fig.supxlabel("Test sizes (%)")
    fig.supylabel("Accuracy")

    ax[0].set_title("Training accuracy")
    ax[1].set_title("Testing accuracy")

    fig.suptitle("gNB Training & Testing Accuracy")
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(bot, test, marker='.')
    ax[1].plot(bot, train, marker='.')

    fig.supxlabel("Test sizes (%)")
    fig.supylabel("Accuracy")

    ax[0].set_title("Training accuracy")
    ax[1].set_title("Testing accuracy")

    fig.suptitle("gNB Training & Testing Accuracy")
    plt.show()


def plot_comparison(k_test: Dict, g_test: List, bot: np.ndarray, pr: List) -> None:
    fig, ax = plt.subplots()
    ax.plot(bot, k_test[pr[1]], label="kNN")
    ax.plot(bot, g_test, label="gNB")
    ax.legend()

    ax.set(xlabel="K-Cross fold", ylabel="Accuracy", title="gNB vs kNN")

    plt.show()


def multi_plot_gnb(table: List, rev=False) -> None:
    """
    Plots CM, CV and ROC for most promising gNB model unless rev flagged is raised,
    in which case plot the least promising model.
    :param table: data -> a list containing all gNB models generated from obtain_gnb_errors
    :param rev: worst -> if True plot the least promising model
    :return: None
    """

    best = table[0] if not rev else table[~0]
    cv_result = cross_validate(GaussianNB(), tbl, names, cv=10, return_train_score=True)
    rp_probs = np.array([np.array([0, 1, 0]) for _ in range(len(best[~2]))])
    r_fpr, r_tpr, thresholds = roc_curve(best[~2], rp_probs[:, 1], pos_label=2)
    rp_auc = roc_auc_score(best[~2], rp_probs, multi_class="ovr")

    cm = confusion_matrix(best[~2], best[~1])
    df_cm = pd.DataFrame(cm, ["Adelie", "Gentoo", "Chinstrap"], ["Adelie", "Gentoo", "Chinstrap"])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.show()

    bot = np.arange(0.0, 10.0, 1.0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(bot, cv_result['fit_time'], marker='.', label="Fit time")
    ax[0].plot(bot, cv_result['score_time'], marker='.', label="Score time")
    ax[1].plot(bot, cv_result['train_score'], marker='.', label="Training score")
    ax[1].plot(bot, cv_result['test_score'], marker='.', label="Testing score")
    fig.supxlabel("Test sizes (%)")

    ax[0].legend()
    ax[1].legend()

    ax[0].set_title("Timings")
    ax[1].set_title("Accuracy")

    fig.suptitle(f"gNB (k={best[~0]})")
    plt.show()

    plt.plot(r_fpr, r_tpr, linestyle='--', label=f"Random prediction, AUROC={rp_auc}")
    plt.plot(best[~4], best[~3], marker='.', label=f"gNB, k=({best[~0]}), AUROC={best[~5]}")
    plt.title("gNB ROC Plot")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


tbl, names = transmute_data(penguins)

create_PCA()

bottom, test_errors, train_errors, k_table, promising = obtain_knn_errors()
gnb_bottom, gnb_test_errors, gnb_train_errors, g_table = obtain_gnb_errors()
plot_knn_errors(bottom, test_errors, train_errors)
plot_gnb_errors(gnb_bottom, gnb_test_errors, gnb_train_errors)
multi_plot_knn(k_table, rev=False)
multi_plot_gnb(g_table, rev=False)
