import pandas as pd
from ml_explore import eval_ratios


def payment_grid(df, focus_cols, group_col):
    df_list = []
    for col in focus_cols:
        new_df = eval_ratios(df, include_cols = [group_col, col], 
                             category_cols = [group_col], method = "sum", 
                             pct = True)
        df_list.append(new_df)
    full_df = pd.concat(df_list, axis = 1)
    return full_df



def standardized_comparison(df, primary_cols, group_cols, insert_col = None):
    df_list = []
    if insert_col:
        focus_cols = primary_cols + [insert_col]
        final_group = group_cols + [insert_col]
        return df[focus_cols].groupby(final_group).mean()

    return df[primary_cols].groupby(group_cols).mean()