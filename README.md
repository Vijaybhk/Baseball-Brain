# Baseball-HomeTeam-Win-Prediction

# Setup for development:

- Set up a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
- `pip3 install --upgrade pip`
- Install pip-tools `pip3 install pip-tools`
- Update dev requirements: `pip-compile --output-file=requirements.dev.txt requirements.dev.in`
- Update requirements: `pip-compile --output-file=requirements.txt requirements.in`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- `pre-commit install`

## Update versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`
`pip-compile --output-file=requirements.txt requirements.in --upgrade`

# Run `pre-commit` locally.

`pre-commit run --all-files`

# Python-Modules

### class DfVariableProcessor

   `DfVariableProcessor(input_df: 'DataFrame | None' = None, predictors: 'list[str] | None' = None, response: 'str | None' = None)`
   
   Variable Processor class to process independent
   and dependent variables in the dataframe
   
   Methods defined here:
   
   __init__(self, input_df: 'DataFrame | None' = None, predictors: 'list[str] | None' = None, response: 'str | None' = None)
       Constructor for the Variable Processor class

       Parameters
       -------------
       
           input_df: DataFrame
               Input Pandas dataframe
           predictors: list[str]
               list of predictors
           response: str
               Response Variable Name
   
   `check_continuous_var(self, var: 'str') -> 'bool'`\
       Method to check if the variable is continuous or not
       
       Parameters
       -------------
           var: str
               Input Variable Name
       
       Returns
       -------------
           flag_cont: bool
               True if Continuous else False
   
   `get_cat_and_cont_predictors(self) -> 'tuple[list, list, dict]'`\
       Method to get the lists of Categorical and Continuous Predictors,
       and dictionary of predictor types
       
       Returns
       -------------
           cat_predictors : list
               lists of Categorical Predictors
           cont_predictors : list
               lists of Continuous Predictors
           pred_type_dict : dict
               dictionary of predictor types
   
   `get_random_forest_scores(self, cont_predictors: 'list[str]') -> 'dict'`\
       Method to execute Random Forest Classifier or Regressor based on
       response type and get the feature importance scores.
       
       Parameters
       -------------
           cont_predictors: list[str]
               list of continuous predictors
       
       Returns
       -------------
           scores_dict: dict
               dictionaries of Random Forest variable importance scores
               with predictor name as key
   
   `get_regression_scores(self, cont_predictors: 'list[str]') -> 'tuple[dict, dict]'`\
       Method to execute logistic or linear regression for each variable
       based on response type and get the p values and t scores.
       
       Parameters
       -------------
           cont_predictors: list[str]
               list of continuous predictors
       
       Returns
       -------------
           t_dict: dict
               dictionaries of t scores with predictor name as key
           p_dict: dict
               dictionaries of p values with predictor name as key
   
   `get_response_type(self) -> 'str'`\
       Method to get the response type. Converts bool(True/False) to 1 and 0 to
       support remaining code. Responses other than boolean categorical or
       continuous will be returned unsupported.
       
       Returns
       -------------
       str
           Outputs Response Type
           "Categorical", or "Continuous", or "Unsupported Categorical"


### class VariablePlotter

   `VariablePlotter(input_df: 'DataFrame | None' = None)`
   
   Custom Class to Plot Variables in a Dataframe
   
   Methods defined here:
   
   __init__(self, input_df: 'DataFrame | None' = None)
       Constructor Method for Plotter Class
       
       Parameters
       -------------
           input_df: DataFrame | None = None
               Input DataFrame
   
   `cat_response_cat_predictor(self, cat_resp: 'str', cat_pred: 'str', write_dir: 'str') -> 'tuple[float, float]'`\
       Method to Create Heat Density Plot for Categorical Response and Categorical Predictor
       and also get Difference in Mean of Response plots and Tables for the Predictor
       
       Parameters
       ------------
       
           cat_resp: str
               Categorical Response Variable
           cat_pred: str
               Categorical Predictor Variable
           write_dir: str
               Directory to Save the plots
       
       Returns
       -------------
           msd: tuple[float, float]
               tuple of Mean Square Difference Unweighted and Weighted
   
   `cat_response_cont_predictor(self, cat_resp: 'str', cont_pred: 'str', write_dir: 'str') -> 'tuple[float, float]'`\
       Method to Create Plots for Categorical Response and Continuous Predictor
       and also get Difference in Mean of Response plots and Tables for the Predictor
       
       Parameters
       -------------
       
           cat_resp: str
               Categorical Response Variable
           cont_pred: str
               Continuous Predictor Variable
           write_dir: str
               Directory to Save the plots
       
       Returns
       -------------
           msd: tuple[float, float]
               tuple of Mean Square Difference Unweighted and Weighted
   
   `cont_response_cat_predictor(self, cont_resp: 'str', cat_pred: 'str', write_dir: 'str') -> 'tuple[float, float]'`\
       Method to Create Plots for Continuous Response and Categorical Predictor
       and also get Difference in Mean of Response plots and Tables for the Predictor
       
       Parameters
       ------------
       
           cont_resp: str
               Continuous Response Variable
           cat_pred: str
               Categorical Predictor Variable
           write_dir: str
               Directory to Save the plots
       Returns
       -------------
           msd: tuple[float, float]
               tuple of Mean Square Difference Unweighted and Weighted
   
   `cont_response_cont_predictor(self, cont_resp: 'str', cont_pred: 'str', write_dir: 'str') -> 'tuple[float, float]'`\
       Method to Create Scatter Plot for Continuous Response and Continuous Predictor
       and also get Difference in Mean of Response plots and Tables for the Predictor
       
       Parameters
       -----------
           cont_resp: str
               Continuous Response Variable
           cont_pred: str
               Continuous Predictor Variable
           write_dir: str
               Directory to Save the plots
       
       Returns
       -------------
           msd: tuple[float, float]
               tuple of Mean Square Difference Unweighted and Weighted
   
   `get_all_plots(self, cont_pred: 'list[str]', cat_pred: 'list[str]', response: 'str', res_type: 'str', write_dir: 'str') -> 'tuple[dict, dict, dict, dict]'`\
       Method to get all the plots based on Continuous/Categorical Variables(Predictor/Response)
       Uses other methods in the class to generated outputs and save plots.
       
       Parameters
       ------------
           cont_pred: list[str]
               List of Continuous predictors you want plotted
           cat_pred: list[str]
               List of Categorical predictors you want plotted
           response: str
               Response Variable you want plotted
           res_type: str
               Response Type either "categorical" or "continuous"
           write_dir: str
               Path to the directory where you want all the plots stored
       
       Returns
       ------------
           diff_dict: dict
               Dictionary object with Variable name as key and combined path of
               difference in mean of response plots and tables
       
           plot_dict: dict
               Dictionary object with Variable name as key and combined path of
               Predictor vs Response plots
       
           uw_dict: dict
               Dictionary object with Variable name as key and Unweighted mean of response
       
           w_dict: dict
               Dictionary object with Variable name as key and Weighted mean of response
   
   ----------------------------------------------------------------------
   Static methods defined here:
   
   `diff_mean_response_plots(df: 'DataFrame', predictor: 'str', response: 'str', write_dir: 'str', predtype: 'str', nbins: 'int' = 10) -> 'tuple[float, float]'`\
       Creates difference with mean of response plots for numerical/continuous predictors
       and saves as html files in the write directory.
       Also, saves weighted and unweighted mean of response dataframes as html
       
       Parameters
       ---------------
       
           df: DataFrame
               Input dataframe
           predictor: str
               predictor column in the dataframe which is a numerical/continuous variable
           response: str
               response variable
           write_dir: str
               Input the write directory path where the plots are to be saved
           nbins: int = 10 as default
               Number of bins to be divided in the bar plot, default is 10.
           predtype: str = None as default
               Predictors Type either categorical or continuous
       
       Returns
       -------------
           msd: tuple[float, float]
               tuple of Mean Square Difference Unweighted and Weighted
   
   `distribution_plot(df: 'DataFrame', cat_var: 'str', cont_var: 'str', write_dir: 'str', restype: 'str')`\
       Static Method to Creates Distribution Plots for Cat vs Cont Variables
       Used Within this class
       
       Parameters
       ------------
       
           df: DataFrame
               Input Dataframe
           cat_var: str
               Categorical Variable(Response/Predictor)
           cont_var: str
               Continuous Variable(Response/Predictor)
           write_dir: str
               Directory to save plots
           restype: str
               Response Type "cat" or "cont"
       
       Returns
       -------------
           None
   
   `violin_plot(df: 'DataFrame', cat_var: 'str', cont_var: 'str', write_dir: 'str', restype: 'str')`\
       Static Method to Creates Violin Plots for Cat vs Cont Variables
       Used Within this class
       
       Parameters
       -------------
       
           df: DataFrame
               Input Dataframe
           cat_var: str
               Categorical Variable(Response/Predictor)
           cont_var: str
               Continuous Variable(Response/Predictor)
           write_dir: str
               Directory to save plots
           restype: str
               Response Type "cat" or "cont"
       
       Returns
       -------------
           None

### class CorrelationAndBruteForce(DfVariableProcessor)

   `CorrelationAndBruteForce(input_df: 'DataFrame', predictors: 'list', response: 'str', write_dir: 'str')`
   
   Custom Class for Correlation And Brute Force.
   As Sub Class to DfVariableProcessor.
   
   Method resolution order:
       CorrelationAndBruteForce
       variables.DfVariableProcessor
       builtins.object
   
   Methods defined here:
   
   __init__(self, input_df: 'DataFrame', predictors: 'list', response: 'str', write_dir: 'str')

       Parameters
       -------------
           input_df: DataFrame
               Input Dataframe, super class input
       
           predictors: list
               Input list of Predictors, super class input
       
           response: str
               Input response variable name, super class input
       
           write_dir: str
               Plots directory path to save plots and tables
   
   `brute_force_matrix(self, name: 'str') -> 'DataFrame'`

       Parameters
       -------------
           name: str
               "Cat_Cat_Cramer", "Cat_Cat_Tschuprow", Cat_Cont", "Cont_Cont"
               Type of input names, to get correlation tables and matrices
       
       Returns
       ----------------
           brute_df: DataFrame
               Brute Force Dataframe with Variable pairs (with links to response plots)
               Correlation coefficients their absolute values.
               Links to brute force matrices
   
   `corr_table_and_matrix(self, name: 'str', save: 'bool' = True) -> 'DataFrame'`

       Parameters
       -------------
           name: str
               "Cat_Cat_Cramer", "Cat_Cat_Tschuprow", Cat_Cont", "Cont_Cont"
               Type of input names, to get correlation tables and matrices
       
           save: bool = True
               whether to save the correlation matrix and table
       
       Returns
       ----------------
           corr_df: DataFrame
               Correlation Dataframe with Variable pairs and Correlation coefficients.
               And links to difference with mean of response plots
   
   `get_all_brute_force_metrics(self)`\
       Saves all brute force tables and matrices for all input name types
   
   `get_all_correlation_metrics(self)`\
       Saves all correlation tables and matrices for all input name types

### generate_report

`generate_report(df: DataFrame, predictors: list[str], response: str, plot_dir: str, data_name: str):`\
    Function to generate individual reports categorical and continuous predictors
    saves in plots directory.
    Combines all brute force and correlation reports to the predictor reports.
    And creates the final html in the root directory as data_name.html

    Parameters
    -----------
        df : DataFrame
            Input dataframe

        predictors: list[str]
            Input list of predictors

        response: str
            Input response variable name

        plot_dir: str
            Input Plots directory path to store plots and predictor reports as html

        data_name: str
            Name of the dataset, uses a header in the html report

    Returns
    -----------
        None
