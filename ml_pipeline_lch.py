# Name: Loren Hinkson

# read in data
def get_data(filename, headers = False, ind = None):
    '''
    Read in data from CSV to a pandas dataframe
    '''
    if headers and isinstance(set_ind, int):
        data_df = pd.read_csv(filename, header = 0, index_col = set_ind)
    elif headers and not set_ind:
        data_df = pd.read_csv(filename, header = 0)
    else:
        data_df = pd.read_csv(filename)
    return data_df



def print_null_freq(df):
    '''
    For all columns in a given dataframe, calculate and print number of null and non-null values

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


def basic_fill_vals(df, col_name, method = None):
    '''
    For columns with more easily predicatable null values, fill with mean, median, or zero
    '''
    if method == "zeros":
        df[col_name] = df[col_name].fillna(0)
    elif method == "mean":
        replacement_val = df[col_name].median()
        df[col_name] = df[col_name].fillna(replacement_val)
    elif method == "mean":
        replacement_val = df[col_name].mean()
        df[col_name] = df[col_name].fillna(replacement_val)



def view_dist(df, categoricals_list):
    '''
    Plot distributions of non-categorical columns in a given dataframe

    Inputs:
        df: pandas dataframe
        categoricals_list: list of strings corresponding to categorical columns (ex: zip codes)
    '''
    print("Feature Frequencies")
    df.drop(categoricals_list, axis = 1).hist(bins = 50, figsize=(20,15), color = 'blue')
    plt.annotate('Source: City of Chicago Open Data Portal', xy=(0.7,-0.2), xycoords="axes fraction")
    plt.show()


def check_corr(df, categorical_list):
    '''
    Check linear correlation between non-categorical columns in a given dataframe

    Inputs:
        df: pandas dataframe
        categoricals_list: list of strings corresponding to categorical columns (ex: zip codes)

    Attribution: https://www.datascience.com/blog/introduction-to-correlation-learn-data-science-tutorials
    '''
    df.drop(categorical_list, axis = 1).corr(method="pearson").style.format("{:.2}").background_gradient(cmap=plt.get_cmap("coolwarm"), axis = 1)



def plot_corr(df, categoricals_list):
    '''
    Observe distributions and correlations of features for non-categorical 

    Inputs:
        df: pandas dataframe
        categoricals_list: list of strings corresponding to categorical columns (ex: zip codes)
    '''
    corr = sns.pairplot(df.drop(categoricals_list, axis = 1))


def change_col_name(df, current_name, new_name):
    '''
    Change name of a single column in a given data frame
    '''
    df.columns = [new_name if col == current_name else col for col in df.columns]


def feature_by_geo(df, geo, expl_var, num_var, method = "median"):
    '''
    Evaluate specific features by geography (ex: zip code)
    
    Inputs:
        df: (dataframe) pandas dataframe
        geo: (string) column name corresponding to geography used for grouping
        expl_var: (string) column name for exploratory variable used for 
            grouping
        num_var: (string) column name for numeric variable/ feature to be 
            aggregated
        method: (string) groupby aggregation method for column values

    Output:
        geo_features: (dataframe)
    '''
    df_geo = df[(df[geo] != 0)]
    groupby_list = [geo] + expl_var
    if method == "median":
        geo_features = df_geo.groupby(groupby_list)[num_var].median().unstack(level = 1)
    if method == "count":
        geo_features = df_geo.groupby(groupby_list)[num_var].count().unstack(level = 1)
    geo_features.fillna(value = "", inplace = True)
    return geo_features


def abs_diff(col, factor, col_median, MAD):
    '''
    Calculate modified z-score of value in pandas data frame column, using sys.float_info.min to avoid dividing by zero

    Inputs:
        col: column name in pandas data frame
        factor: factor for calculating modified z-score (0.6745)
        col_median: median value of pandas data frame column
        MAD: mean absolute difference calculated from pandas dataframe column
    
    Attribution: workaround for MAD = 0 adapted from https://stats.stackexchange.com/questions/339932/iglewicz-and-hoaglin-outlier-test-with-modified-z-scores-what-should-i-do-if-t
    '''
    if MAD == 0:
        MAD = 2.2250738585072014e-308 
    return (x - y)/ MAD


def outliers_modified_z_score(df, col):
    '''
    Identify outliers (values falling outside 3.5 times modified z-score of median) in a column of a given data frame

    Attribution: Modified z-score method for identifying outliers adapted from from http://colingorrie.github.io/outlier-detection.html
    '''
    threshold = 3.5
    zscore_factor = 0.6745
    col_median = df[col].astype(float).median()
    median_absolute_deviation = abs(df[col] - col_median).mean()
    
    modified_zscore = df[col].apply(lambda x: abs_diff(x, zscore_factor, col_median, median_absolute_deviation))
    return modified_zscore[modified_zscore > threshold]


