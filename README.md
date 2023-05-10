# SkinCancerDetection


Just copying some code from old README to have references on how to insert links/images/repo structure

## For More Information

Please review my full analysis in [my Jupyter Notebook](./TimeseriesNotebook.ipynb) or my [presentation](./Presentation.pdf).

For any additional questions, please contact **Maria Kuzmin, marianlkuzmin@gmail.com**

## Repository Structure

Description of the structure of the repository and its contents:

```
├── .ipynb_checkpoints
├── Graphs
    └── Frecasting.png
    └── PieStates.png
    └── Predsontest.png
    └── Predsontestzoom.png
    └── TrainTestSplit.png
    └── YearStates.png
├── .gitignore
├── Presentation.pdf
├── README.md
├── SARgrid1.pkl
├── SARgs.pkl
├── TimeSeriesNotebook.ipynb   
└── organised_Gen.csv
```

### To add an image

This is what we found:

"![YearStates](./Graphs/YearStates.png)"


### Embedded Link to a website 

 you can read [this article](https://www.globalpwr.com/top-ten-industries-requiring-megawatts-of-generator-power/)). 
 
Regular link   adapted from https://www.eia.gov/electricity/, where

### Roadmap:
Here is a roadmap of the steps that we took:

The Data:
* Data Preparation:
    * Study of Energy Source
    * Natural gas, solar and wind
    * Checking for normality
* EDA of Natural gas:
    * Split Train Validation and Test Set
    * Subtracting Rolling Mean
    * Series Decomposition
    * Studying Autocorrelation: ACF PACF
* Modeling:
    * Baseline Model: Naive
    * ARMA Models
    * Grid Search for ARIMA models:
        * First Search
        * Best model from Grid Search
        * Cross Validation
        * Second Search
        * Best model from second search
        * Cross Validation
    * SARIMAX
        * First Grid Search SARIMAX
        * Best model after first search
        * Cross Validation
        * Second Grid Search SARIMAX
        * Best model after second search
        * Cross Validation
* Predicting on the test
* Forecasting in the future
* Study of Seasonality and States
* Results
* Limitations
* Recommendations