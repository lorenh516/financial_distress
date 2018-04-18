# Name: Loren Hinkson

# read in data
def get_data(filename, headers = False, ind = None):
    '''
    Read in data from CSV to a pandas dataframe

    Inputs:
        filename: (string) filename of CSV
        headers: (boolean) whether or not CSV includes headers
        ind: (integer) CSV column number of values to be used as indices in 
            data frame

    Output: pandas data frame
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

    Inputs:
        df: pandas data frame
        col_name: (string) column of interest
        method: (string) desired method for filling null values in data frame. 
            Inputs can be "zeros", "median", or "mean"
    '''
    if method == "zeros":
        df[col_name] = df[col_name].fillna(0)
    elif method == "median":
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



def isolate_noncategoricals(df, ret_categoricals = False, geo_cols = None):
    '''
    Retrieve list of cateogrical or non-categorical columns from a given dataframe

    Inputs:
        df: pandas dataframe
        ret_categoricals: (boolean) True when output should be list of  
            categorical colmn names, False when output should be list of non-categorical column names

    Outputs: list of column names from data frame
    '''
    if ret_categoricals:
        categorical = [col for col in df.columns if re.search("_bin", col)]
        return categorical + geo_cols
    else:
        non_categorical = [col for col in df.columns if not re.search("_bin", col) and col not in geo_cols]
        return non_categorical


def check_corr(df, categorical_list):
    '''
   Display heatmap of linear correlation between non-categorical columns in a given dataframe

    Inputs:
        df: pandas dataframe
        geo_columns: list of column names corresponding to columns with numeric geographical information (ex: zipcodes)

    Attribution: Colormap Attribution: adapted from gradiated dataframe at https://www.datascience.com/blog/introduction-to-correlation-learn-data-science-tutorials and correlation heatmap at https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas
    '''
    fig, ax = plt.subplots(figsize=(12, 12))
    non_categoricals = isolate_noncategoricals(copy_df, ret_categoricals = False, geo_cols = geo_columns)
    
    corr = df[non_categoricals].corr(method="pearson")
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=plt.get_cmap("coolwarm"),
            square=True, ax=ax, annot=True)
    
    ax.set_xticks(range(len(non_categoricals)))
    ax.set_yticks(range(len(non_categoricals)))

    ax.tick_params(direction='inout')
    ax.set_xticklabels(non_categoricals, rotation=45, ha='right')
    ax.set_yticklabels(non_categoricals, rotation=45, va='top')
    plt.show()



def discretize_cols(df, geo_columns, num_bins):
    '''
    Add columns to discretize and classify non-categorical columns in a given data frame

    Inputs:
        df: pandas dataframe
        geo_columns:  list of column names corresponding to columns with 
            numeric geographical information (ex: zipcodes)
        num_bins: number of groups into which column values should be discretized
    '''
    for col in cols_to_bin:
        bin_col = col + "_bin"
        if col == "age":
            age_bins = math.ceil((df[col].max() - df[col].min()) / 10)
            df[bin_col] = pd.cut(df[col], bins = age_bins, right = False, precision=0)
        else:
            try:
                df[bin_col] = pd.qcut(df[col], q = num_bins, precision=0)
            except:
                df[bin_col] = pd.qcut(df[col], q = num_bins + 3, precision=0, duplicates = 'drop')



def plot_corr(df, geo_columns, color_category):
    '''
    Observe distributions and correlations of features for non-categorical 

    Inputs:
        df: pandas dataframe
        categoricals_list: list of strings corresponding to categorical columns (ex: zip codes)
    '''
    non_categoricals = isolate_noncategoricals(df, ret_categoricals = False, geo_cols = geo_columns)
    plot_list = non_categoricals + [color_category]
    corr = sns.pairplot(df[plot_list], hue = color_category, palette = "Set2")


def plot_relationship(df, feature_x, xlabel,feature_y, ylabel, xlimit = None, ylimit = None, color_cat = None):
    '''
    Plot two features in a given data frame against each other to view relationship and outliers. 
    
    Attribution: adapted from https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf
    '''
    sns.set_style("whitegrid")
    g = sns.lmplot(x = feature_x, y = feature_y, data = df, aspect = 3, hue = color_cat)
    g = (g.set_axis_labels(xlabel,ylabel)).set(xlim = xlimit , ylim = ylimit)
    plot_title = ylabel + " by " + xlabel
    plt.title(plot_title)
    plt.show(g)


def change_col_name(df, current_name, new_name):
    '''
    Change name of a single column in a given data frame
    '''
    df.columns = [new_name if col == current_name else col for col in df.columns]



def plot_relationship(df, feature_x, xlabel, feature_y, ylabel):
    '''
    Plot two features in a given data frame against each other to view relationship and outliers. 
    
    Attribution: adapted from https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf
    '''
    sns.set_style("whitegrid")
    g = sns.lmplot(x = feature_x, y = feature_y, data = df, aspect = 2)
    g = (g.set_axis_labels(xlabel,ylabel))
    plot_title = ylabel + " by " + xlabel
    plt.title(plot_title)
    plt.show(g)
    


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



def view_max_mins(df, max = True):
    '''
    View top and bottom 10% of values in each column of a given data frame

    Inputs: 
        df: pandas dataframe
        max: (boolean) indicator of whether to return to or bottom values
    '''
    if max:
        return df.quantile(q=np.arange(0.99, 1.001, 0.001))
    else: 
        return df.quantile(q=np.arange(0.0, 0.011, 0.001))


def view_likely_outliers(df, max = True):
    '''
    View percent change between percentiles in top or bottom 10% of values in  each column of a given data frame 

    Inputs: 
        df: pandas dataframe
        max: (boolean) indicator of whether to return to or bottom values
    '''
    if max:
        return df.quantile(q=np.arange(0.9, 1.001, 0.001)).pct_change()
    else: 
        return df.quantile(q=np.arange(0.0, 0.011, 0.001)).pct_change()



def remove_over_threshold(df, col, threshold):
    '''
    Remove values over given percentile in a column of a given data frame
    '''
    maxes = view_max_mins(df, max = True)
    df.loc[df[col] > maxes.loc[threshold, col], col] = None

    
def remove_dramatic_outliers(df, col, threshold, max = True):
    '''
    Remove values over certain level of percent change in a column of a given data frame
    '''
    if max:
        maxes = view_max_mins(df, max = True)
        likely_outliers_upper = view_likely_outliers(df, max = True)
        outlier_values = list(maxes.loc[likely_outliers_upper[likely_outliers_upper[col] > threshold][col].index, col])
    else: 
        mins = view_max_mins(df, max = False)
        likely_outliers_lower = view_likely_outliers(df, max = False)
        outlier_values = list(mins.loc[likely_outliers_lower[likely_outliers_lower[col] > threshold][col].index, col])
    
    df = df[~df[col].isin(outlier_values)]



def create_binary_vars(df, cols_to_dummy, keyword_list):
    '''
    Create columns of binary values corresponding to values above zero for selected columns in a given dataframe based on common keywords

    Inputs:
        df: pandas dataframe
        cols_to_dummy: (list of strings) columns in data frame to be evaluated 
            into dummy variables
        keyword_list: (list of strings) words or phrases included in columns 
            to be evaluated indicating a dummy variable should be created based on its values 
    '''
    keyword_string = ("|").join(keyword_list)
    for col in cols_to_dummy:
        colname_trunc = re.sub(keyword_string, '', col)
        binary_col_name = 'tf_' + colname_trunc
        df[binary_col_name] = df[col].apply(lambda x: x > 0)

