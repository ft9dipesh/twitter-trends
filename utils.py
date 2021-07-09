import pandas as pd


def percentage(part, whole):
    return round(100 * float(part) / float(whole), 2)


def count_values_in_column(data, feature):
    total = data.loc[:,feature].value_counts(dropna=False)

    percentage = round(data.loc[:, feature].value_counts(dropna=False, normalize=True) * 100,2)

    return pd.concat(
        [total, percentage], 
        axis=1, 
        keys=["Total", "Percentage"]
    )
