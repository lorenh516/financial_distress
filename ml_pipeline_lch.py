# Name: Loren Hinkson

# read in data
def get_data(filename, headers = False, ind = None):
    '''
    Read in data from CSV to a pandas dataframe
    '''
    if headers and ind:
        data_df = pd.read_csv(filename, header = 0, index_col = ind)
    elif headers and not ind:
        data_df = pd.read_csv(filename, header = 0)
    else:
         data_df = pd.read_csv(filename)
    return data_df



def print_null_freq(df):
    '''
    For all columns in given dataframe, calculate and print number of null and non-null values

    Attribution: https://github.com/yhat/DataGotham2013/blob/master/analysis/main.py
    '''
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    print(pd.crosstab(df_lng.variable, null_variables))


def create_col_ref(df):
    '''
    Develop quick check of column position via dictionary
    '''
    col_list = df.columns
    col_dict = {}
    for list_position, col_name in enumerate(col_list):
        col_dict[col_name] = list_position
    return col_dict