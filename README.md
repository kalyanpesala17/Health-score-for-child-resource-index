# CPAL's CRI-index-2.0
Created a child resource index health score for 1172 census tracts in Dallas, Collin, Tarrant and Denton. CRI 1.0 provides a measure for community strengths and weaknesses to policymakers, stakeholders, and residents to make data-driven decisions. IN CRI 2.0 we have added more indicators and Implement individual weightage for each indicator based on how strongly the indicator predicts health outcomes, which improves the predictive validity of the index. 

# Installation
pandas    
numpy   
matplotlib.pyplot     
seaborn     
statsmodels.api     
sklearn     

# Code Files description
 - Data Collection.ipynb : collected the data from various sources and created a master dataframe.
 - Model.py : Perform data preprocessing and transformation to execute linear regression model.
 - Healthscores.py : Calcultes health scores with the linear regression weights from the given y variables.
 - Results.ipynb : imports model.py and healthscores.py and creates a health score using our methodology for any data in the raw format.
 

# Results
![Tableau viz](https://github.com/kalyanpesala17/Health-score-for-child-resource-index/blob/master/result.jpeg) 

# Licensing, Authors, Acknowledgements
Must give credit to CDC for the public data and CPAL’s mission of mitigating the root-drivers of child poverty.. Feel free to use the code here as you would like!
