<h1><center>SLEEK HOME ASSIGNMENT - ALON RAZON</center></h1>

<br> **Instructions**
- Locate the file archive.zip in data folder.
- Run the notebook unzip_file.ipynb to unzip archive.zip.
- Run the notebook dataset_structure.ipynb.
- Run the notebook EDA.ipynb before running ML_Models.ipynb
  

<br>**Objective**

<br>This assignment aims to evaluate proficiency in:
- Learning the domain
- Recognizing potential opportunities
- Deploying advanced machine learning (ML) and deep learning (DL) techniques to extract insights that would otherwise remain obscure.
<br>The task is to analyze the CICIDS2017 dataset:
- Importing the dataset
- Conducting comprehensive exploratory data analysis (EDA)
- Deriving three compelling insights. At least two of these insights must harness the power of ML or DL techniques to extract valuable information. 

<br>CICIDS2017 Dataset

<br>Intrusion Detection Systems (IDSs) and Intrusion Prevention Systems (IPSs) are the most important defense tools against the sophisticated and ever-growing network attacks. Due to the lack of reliable test and validation datasets, anomaly-based intrusion detection approaches are suffering from consistent and accurate performance evolutions.

<br>Our evaluations of the existing eleven datasets since 1998 show that most are out of date and unreliable. Some of these datasets suffer from the lack of traffic diversity and volumes, some do not cover the variety of known attacks, while others anonymize packet payload data, which cannot reflect the current trends. Some are also lacking feature set and metadata.

<br>CICIDS2017 dataset contains benign and the most up-to-date common attacks, which resembles the true real-world data (PCAPs). It also includes the results of the network traffic analysis using CICFlowMeter with labeled flows based on the time stamp, source, and destination IPs, source and destination ports, protocols and attack (CSV files). Also available is the extracted features definition.

<br>Generating realistic background traffic was our top priority in building this dataset. We have used our proposed B-Profile system (Sharafaldin, et al. 2016) to profile the abstract behavior of human interactions and generates naturalistic benign background traffic. For this dataset, we built the abstract behavior of 25 users based on the HTTP, HTTPS, FTP, SSH, and email protocols.

<br>The data capturing period started at 9 a.m., Monday, July 3, 2017, and ended at 5 p.m. on Friday, July 7, 2017, for a total of 5 days. Monday is the normal day and only includes benign traffic. The implemented attacks include Brute Force FTP, Brute Force SSH, DoS, Heartbleed, Web Attack, Infiltration, Botnet and DDoS. They have been executed both morning and afternoon on Tuesday, Wednesday, Thursday and Friday.

<br>**Importing and Familiarization**

<br>CICIDS2017 Dataset was downloaded from Kaggle: https://www.kaggle.com/datasets/cicdataset/cicids2017
<br>I acquired the data set and loaded it into a Jupyter Notebook environment for further analysis using Python libraries. 
<br>Data file archive.zip decompress under data folder.
<br>Data includes 8 CSV files which are located in: data/MachineLearningCSV/MachineLearningCVE/ and MachineLearningCSV.md5 located under data.
<br>CSV files were loaded and concatenated to one pandas DataFrame. 
- I checked that all CSV files include the same features (DataFrame columns).
- I added the CSV file names to new column named 'source' for analysis purposes.
- Data saved as pickle file named raw_data.pkl in data folder.
<br>Raw Data shape: (2830743, 80).
<br>All the above is done in the script dataset_structure.ipynb.

<br>**Exploratory Data Analysis (EDA)**

<br>EDA was implemented using 3 distinct Python scripts:
- Visualization.ipynb
- EDA.ipynb
- ML_Models.ipynb

<br>**Visualization.ipynb:**
- Loads raw_data.pkl
- Edit titles, complete missing data (will be detailed further).
- Generated distribution histograms for each feature. Images are located in: Images/Feature_Distribution.
- Generated Boxplot (visual descriptive statistics) for each feature. Images are located in: Images/BoxPlot.
- Principal Component Analysis (**PCA**):
    - PCA was initiated to 95% of data's variance (n_components).
    - The result was 26 principal components (PCs) where the 3 mains discribed only ~40% of the variance.
        - 40% of the variance is not suffitient for visual interpetation.
        - I could use these 26 PCs as new features (which could save computation time) but PCs has no actual meaning except variance description thus, I chose to continue investigate the data. Using PCs as features is an option.
    - PCA explained_variance_ratio_ was plotted and saved in: Images/PCA/Explained_Variance_Ratio.png

<br>**EDA.ipynb:**
<br>At this script I performed preprocessing (wrangling) and statistical computations.
<br>The Script:
- Loads raw_data.pkl.
    - Raw data loading time: 2.16 seconds
    - DataFrame occupies: 1.83 GB
    - Removes leading and trailing characters (whitespace) from columns titles. I asume there is a meaning behind it, some features are sub-features of a main one.
    - Performed descriptive statistics on all data and save it in wrangling/descriptive_statistics.csv.
        - Observation: Flow Bytes/s and Flow Packets/s include numpy inf.
        - numpy inf values replaced with the value of: 10 * feature_max_value
    - I had an assumption that these inf values are a sign for attack thus, I checked the labels:
        - Samples Label, where Flow Bytes/s include inf values:
            - BENIGN         1368
            - PortScan        126
            - Bot              10
            - FTP-Patator       3
            - DDoS              2
        - Samples Label, where Flow Packets/s include inf values:
            - BENIGN         1777
            - DoS Hulk        949
            - PortScan        126
            - Bot              10
            - FTP-Patator       3
            - DDoS              2
        - Samples Label, where Flow Bytes/s and Flow Packets/s include inf values:
            - BENIGN         1368
            - PortScan        126
            - Bot              10
            - FTP-Patator       3
            - DDoS              2
        - Conclusion: inf values are not an indication for attack because they are common at benign samples.
     - Flow Bytes/s include NA values. I checked the labels:
         - DoS Hulk    949
         - BENIGN      409
         - **Conclusion**: NA values could indicate on DoS Hulk attack but further investigation on benign samples is required (False Positive).
     - Missing data (NA values) were completed using features modes. Modes dictionary was saved at: utils/modes.json
     - I Generated two additional files and saved them at:
         - wrangling/raw_data_type.csv - data type
             - data types are: int64 and float64 -> should be converted to int32 and float32
             - DataFrame occupies: 0.27 GB
             - FEATURES WERE NOT CONVERTED DUE TO ISSUE WITH VALUES: Input contains infinity or a value too large for dtype('float32')
         - wrangling/raw_data_n_uniques.csv - number of unique values in each feature (column)
             - Observation: some of the features include only one unique value -> No contribution thus, can be discarded.
             - Data new shape (after dropping features with one value): (2830743, 72)
    
     - **Correlation between different features:**
        <br> Dropping correlated features from data isn't always mandatory, but it's generally recommended for several reasons:
        - Redundancy: Correlated features contain overlapping information. Keeping both provides no additional insights and can even be detrimental.
        - Multicollinearity: Highly correlated features can lead to multicollinearity, where model coefficients become unreliable and interpretation difficult.
        - Model Simplicity: Fewer features often lead to simpler models that are easier to interpret and potentially less prone to overfitting (memorizing the training data instead of learning general patterns).
        - Computational Efficiency: Training models with fewer features can be faster, especially for complex algorithms.
        <br> Correlation threshold chosen to be **0.9**
        <br> Correlation matrix is documented in wrangling/raw_data_correlation.csv
        <br> Correlated features names are documented in: utils/correlated_features_names.pkl
        <br> New data shape (after dropping correlated features): (2830743, 43)
        <br> **Note:**
        <br>There are situations where dropping correlated features might NOT be crucial:
        - Domain Knowledge: If you have strong domain knowledge and understand the meaning of both features, keeping them might be beneficial.
        - Ensemble Methods: Some ensemble methods (like Random Forests) can handle correlated features better.
        - Feature Importance: If a correlated feature still has independent predictive power, it might be worth keeping.
     
     - **Drop Duplications:**
        <br> Duplicate data points in K-Means (ML in general) can unfairly influence cluster centers, giving more weight to redundant information and potentially drowning out unique data points.
        <br> New data shape (after drop_duplicates): (2522256, 43)
        
     - **Ouliers Suspects using IQR method**
        <br>IQR stands for Interquartile Range. It's a measure of spread or variability used in statistics, specifically focusing on the middle 50% of data set. IQR is a valuable tool for understanding the distribution of the data, especially when outliers might be present.
        <br> The data is composed of benign traffic (Monday) and attacks occured during the next 4 days. Seemingly, reducing ouliers at benign samples will reduce data overlapping with the attacks. There are issues with that statment, and the process should be done very carfully but for the purpose of this assignment, IQR method will be presented to identify noisy features. Those features will NOT be dropped. 
        <br> IQR analysis results are documented in: wrangling/raw_data_IQR_analysis.csv
        <br> Observation: most of the data composed out of BENIGN and distributed along all data sources (5 days).
        <br> Conclusion: most of the features include more 100k outliers. A new way will be considered.
    - **Clustering using K-Means - Unsupervised**
        <br> The goal is to idetify benign samples overlapped with attacks by clustering
        - **Standard Scaling**
            <br> K-means clustering relies on distance calculations to group data points. When features have different scales, features with larger scales dominate the distance calculation, even if they might not be as relevant for grouping. Scaling features ensures all features contribute equally based on their relative values, not just their magnitude, leading to more meaningful clusters.
            <br> StandardScaler Standardize features by removing the mean and scaling to unit variance.
            <br> The standard score of a sample x is calculated as:
            <br> z = (x - u) / s
            <br> where u is the mean of the training samples and s is the standard deviation of the training samples.
        - **Elbow method**
            <br> The elbow method is a visual technique used to determine the optimal number of clusters (k) for K-means clustering.
            <br> Elbow method graph is located in: Images/Elbow_Method.png
            <br> Conclusion: Elbow method showed that K = 9 is a reasonable choice.
        - K-Means Clustering Conclusion: ~96% of the data belong to clusters: 8, 6, 4, 0, 7
            <br> The other 3% belong to clusters: 1, 2, 3, 5
            <br> All samples belong to clusters  1, 2, 3, 5 will be discarded.
            <br> Importent NOTE: those samples most be reported on and discussed with Sleek Data teams.
            <br> Importent NOTE: those samples most be reported on and discussed with Sleek Data teams.
     -  Data for ML model saved at: data/processed_data.pkl
            <br> Data Final Shape: (2386579, 43)

<br> **Two interesting insights:**
- Feature Reduction Potential: A significant portion of the features can be eliminated due to high correlation with other features or a lack of informative value (e.g., features with only one unique value). In the industry, reducing the number of features offers several benefits, including:
    - Reduced data storage requirements
    - Improved computational efficiency (both human and machine)
    - Potential cost savings
- Data Quality Issues: The presence of numpy.inf and NA values in the data warrants further investigation. While the cause is currently unknown, the NA values could potentially indicate a DoS Hulk attack. However, further analysis of benign samples is necessary to differentiate true DoS attempts from false positives.

<br>**ML_Models.ipynb**
<br>Train and test Random Forest and GradientBoosting Classifiers
<br>The Script:
- Loads processed_data.pkl (located in data/processed_data.pkl).
    - Raw data loading time: 1.04 seconds
    - DataFrame occupies: 0.87 GB
- Present samples distribution (already done in earlier stages in the project):
    - distribution plot is located in: Images/Samples_Distribution.png
    - Label Value Counts:
        - BENIGN                       2208187
        - PortScan                      158631
        - DDoS                          106086
        - DoS Hulk                       89156
        - FTP-Patator                     7935
        - DoS GoldenEye                   7868
        - SSH-Patator                     5897
        - DoS Slowhttptest                4514
        - DoS slowloris                   4417
        - Bot                             1912
        - Web Attack  Brute Force         1507
        - Web Attack  XSS                  652
        - Infiltration                      29
        - Web Attack  Sql Injection         21
        - Heartbleed                         6
     - Conclusion: **Unbalnced data**
     - Solution considered: weighting each sample according to it's label for example, benign gets the lowest weight because it is the most frequent and heartbleed gets the highest.
     
     - Seperate data to X - features and y - labels. features Shape: (2420904, 41). target Shape: (2420904,).
     - Encoding labels (can be performed in a lot of ways, here I chose to it by myself by mapping).
       - **Label2ID**: {'BENIGN': 0, 'DDoS': 1, 'PortScan': 2, 'Bot': 3, 'Infiltration': 4, 'Web Attack  Brute Force': 5, 'Web Attack  XSS': 6, 'Web Attack  Sql Injection': 7, 'FTP-Patator': 8, 'SSH-Patator': 9, 'DoS slowloris': 10, 'DoS Slowhttptest': 11, 'DoS Hulk': 12, 'DoS GoldenEye': 13, 'Heartbleed': 14}
        - **ID2Label**: {0: 'BENIGN', 1: 'DDoS', 2: 'PortScan', 3: 'Bot', 4: 'Infiltration', 5: 'Web Attack  Brute Force', 6: 'Web Attack  XSS', 7: 'Web Attack  Sql Injection', 8: 'FTP-Patator', 9: 'SSH-Patator', 10: 'DoS slowloris', 11: 'DoS Slowhttptest', 12: 'DoS Hulk', 13: 'DoS GoldenEye', 14: 'Heartbleed'}
     - Split to Train and Test (test size = 0.5 due to volume of data)
        <br>NOTE: At the industry, data will be split to train, test, validation sets
        <br>I used stratify arg to keep same labels propotion in train and test set.
        - X Train shape: (1210452, 41)
        - y Train shape: (1210452,)
        - X Test shape: (1210452, 41)
        - y Test shape: (1210452,)
     - Scaled features
         - While feature scaling is generally not required for decision-tree based algorithms like random forests, I opted to standardize the features in this instance. This decision was made to ensure consistency with the planned inclusion of other algorithms in the future that may necessitate scaled features. Due to resource constraints, this project focused solely on Random Forest and GradientBoosting. However, it is important to acknowledge that scaling can be a crucial step when employing a broader range of machine learning models.
     - **Train Random Forest**:
        - Training run time: 2262.0181295871735 seconds
        - Prediction run time: 19.09727168083191 seconds
- Trining Results:
<br>| **Precision** | **Recall** | **F1-score** | **Support** |
<br>|---|---|---|---|
<br>| BENIGN | 1.00 | 1.00 | 1.00 | 1016137 |
<br>| DDoS | 0.99 | 0.99 | 0.99 | 956 |
<br>| PortScan | 1.00 | 1.00 | 1.00 | 53043 |
<br>| Bot | 1.00 | 1.00 | 1.00 | 3934 |
<br>| Infiltration | 1.00 | 1.00 | 1.00 | 44578 |
<br>| Web Attack Brute Force | 0.99 | 1.00 | 0.99 | 2257 |
<br>| Web Attack XSS | 1.00 | 0.99 | 1.00 | 2209 |
<br>| Web Attack Sql Injection | 1.00 | 1.00 | 1.00 | 3967 |
<br>| FTP-Patator | 1.00 | 1.00 | 1.00 | 3 |
<br>| SSH-Patator | 1.00 | 0.79 | 0.88 | 14 |
<br>| DoS slowloris | 0.99 | 1.00 | 1.00 | 79315 |
<br>| DoS Slowhttptest | 1.00 | 1.00 | 1.00 | 2949 |
<br>| DoS Hulk | 0.97 | 0.99 | 0.98 | 753 |
<br>| DoS GoldenEye | 0.83 | 0.91 | 0.87 | 11 |
<br>| Heartbleed | 0.98 | 0.91 | 0.94 | 326 |
<br>| **accuracy** |  |  | 1.00 | 1210452 |
<br>| **macro avg** | 0.98 | 0.97 | 0.98 | 1210452 |
<br>| **weighted avg** | 1.00 | 1.00 | 1.00 | 1210452 |
<br>
- Testing Results:
<br>| **precision** | **recall** | **f1-score** | **support** |
<br>|---|---|---|---|
<br>| BENIGN | 1.00 | 1.00 | 1.00 | 1016136 |
<br>| DDoS | 0.86 | 0.78 | 0.82 | 956 |
<br>| PortScan | 1.00 | 1.00 | 1.00 | 53043 |
<br>| Bot | 1.00 | 0.99 | 0.99 | 3934 |
<br>| Infiltration | 1.00 | 1.00 | 1.00 | 44578 |
<br>| Web Attack Brute Force | 0.99 | 0.99 | 0.99 | 2257 |
<br>| Web Attack XSS | 0.99 | 0.99 | 0.99 | 2208 |
<br>| Web Attack Sql Injection | 1.00 | 1.00 | 1.00 | 3968 |
<br>| FTP-Patator | 1.00 | 0.33 | 0.50 | 3 |
<br>| SSH-Patator | 0.83 | 0.33 | 0.48 | 15 |
<br>| DoS slowloris | 0.99 | 1.00 | 1.00 | 79316 |
<br>| DoS Slowhttptest | 1.00 | 1.00 | 1.00 | 2948 |
<br>| DoS Hulk | 0.75 | 0.84 | 0.79 | 754 |
<br>| DoS GoldenEye | 0.42 | 0.50 | 0.45 | 10 |
<br>| Heartbleed | 0.48 | 0.33 | 0.39 | 326 |
<br>| **accuracy** |  |  | 1.00 | 1210452 |
<br>| **macro avg** | 0.89 | 0.81 | 0.83 | 1210452 |
<br>| **weighted avg** | 1.00 | 1.00 | 1.00 | 1210452 |

- Feature importence distribution is located in: Images/RF_Feature_Importence.png

<br> In the initial stages of exploration, I attempted to train an XGBoost model. However, due to the inherent characteristics of XGBoost, training time proved to be excessively long for the project's scope. Additionally, memory usage became a significant constraint.
<br> It's important to note that in my professional work environment, I leverage AWS SageMaker. This platform offers a variety of instance types optimized for different machine learning tasks, which can significantly improve training efficiency for resource-intensive algorithms like XGBoost.

<br> **Interesting insight**:
<br> While a comprehensive model wasn't achievable due to resource constraints, the exploration using Random Forest identified promising features for network intrusion detection of ten attack types.
- Analysis revealed that Destination Port, Maximum Forward Packet Length, PSH Flag Count, Total Forward Packet Length, and Forward Header Length emerged as the most significant features for attack classification.
- These findings warrant further investigation by security researchers and domain experts to refine the model and optimize detection capabilities for the identified attack types.

