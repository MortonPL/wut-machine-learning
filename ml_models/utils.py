def get_x(dataset, y_column):
    cols = [col for col in dataset.columns if col != y_column]
    if len(cols) == 1:
        return dataset.loc[:, cols].values.reshape(-1, 1)
    return dataset.loc[:, cols]
