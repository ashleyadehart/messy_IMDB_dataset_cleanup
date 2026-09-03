# IMDB Dataset Cleaning & Data Preparation

## Project Overview

This project focuses on cleaning and preparing a messy IMDb movie dataset for use in downstream data analysis.

The original dataset contains information about 100 movies across multiple attributes, including movie titles, release years, genres, duration, country, content ratings, directors, worldwide income, number of votes, and IMDb scores.

The dataset was intentionally created as an example of "dirty data" and contains a variety of common data-quality problems, including missing values, empty rows and columns, inconsistent variable names, multiple date formats, improperly formatted numeric values, inconsistent decimal and thousands separators, typographical errors, and incorrectly encoded categorical data.

The goal of this project was to identify and correct these issues and produce a clean, consistent, analysis-ready dataset.

This project demonstrates practical skills in:

- Data quality assessment
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Missing-value handling
- Duplicate and empty-record detection
- Data type conversion
- String cleaning and standardization
- Numeric data cleaning
- Categorical data standardization
- Data validation
- Python programming
- Pandas
- Jupyter Notebook

## Project Structure

```
├── messy_IMDB_dataset.csv
├── clean_IMDB_dataset.csv
├── main.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run This Project

### 1. Clone This Repository

```bash
git clone <REPOSITORY_URL>
```

### 2. Create a Virtual Environment

Run the following command to create a virtual environment in a folder named `.venv`:

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### **Windows (Command Prompt):**

```bash
.venv\Scripts\activate.bat
```

#### **Windows (PowerShell):**

```bash
.venv\Scripts\Activate.ps1
```

#### **macOS / Linux (Git Bash):**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

Once the virtual environment is activated:

```bash
pip install -r requirements.txt
```

### Deactivating

When you are finished working on the project:

```bash
deactivate
```

## Tools & Technologies

* Python
* Pandas
* VS Code
* Git
* GitHub

## Dataset

### Source

The original dataset was obtained from the Kaggle **Messy IMDb Dataset**, created by David Fuente Herraiz.

The dataset contains information about 100 movies and includes 11 primary variables covering movie identification, titles, release years, genres, duration, country, content rating, director, worldwide income, number of votes, and IMDb score.

The dataset was intentionally designed to demonstrate common real-world data-quality problems.

### Original Dataset Location

```text
messy_IMDB_dataset.csv
```

### Processed Dataset Location

```text
clean_IMDB_dataset.csv
```

## Data Quality Issues

The initial inspection of the dataset revealed several data-quality issues.

### Missing Values

Missing values were identified throughout the dataset and evaluated on a column-by-column basis.

The cleaning process included:

* Identifying columns containing missing values.
* Determining the extent of missingness.
* Evaluating whether missing values should be removed, replaced, or retained.
* Documenting the approach used for each affected variable.

### Empty Rows and Columns

The dataset contained empty rows and/or columns that needed to be identified and removed where appropriate.

### Inconsistent Variable Names

The original dataset contained variable names that required cleaning and standardization.

Column names were reviewed and modified to improve:

* Consistency
* Readability
* Usability in Python
* Compatibility with downstream analysis

### Inconsistent Date Formats

Release-year/date information contained inconsistent formatting.

The cleaning process standardized these values into a consistent format appropriate for analysis.

### Numeric Formatting Issues

Several numeric variables contained formatting problems, including:

* Currency symbols
* Units
* Non-numeric characters
* Thousands separators
* Inconsistent decimal separators

These values were cleaned and converted into appropriate numeric data types.

### Typographical Errors

Text fields were inspected for typographical inconsistencies and corrected where the intended value could be determined reliably.

### Categorical Data Issues

Categorical variables contained inconsistent or incorrectly encoded values.

These values were reviewed and standardized so that equivalent categories could be represented consistently.

## Data Cleaning Methodology

The cleaning workflow followed a systematic process:

1. Loaded the raw IMDb dataset.
2. Inspected the dataset dimensions and structure.
3. Examined column names and data types.
4. Identified empty rows and columns.
5. Assessed missing values.
6. Identified inconsistent text values.
7. Standardized column names.
8. Cleaned and standardized categorical variables.
9. Cleaned numeric fields containing symbols and formatting characters.
10. Standardized decimal and thousands separators.
11. Converted variables to appropriate data types.
12. Investigated potential data-quality anomalies.
13. Validated the cleaned dataset.
14. Exported the processed dataset.

## Before & After

### Original Dataset

The original dataset contained a variety of intentionally introduced data-quality problems, including:

* Missing values
* Empty rows and columns
* Inconsistent variable names
* Multiple date formats
* Improperly formatted numeric values
* Currency symbols and units
* Inconsistent decimal separators
* Typographical errors
* Incorrect categorical encoding

### Cleaned Dataset

The resulting dataset was transformed into a more consistent, structured, and analysis-ready format.

The cleaned dataset contains:

* Standardized column names
* Appropriate data types
* Consistent categorical values
* Clean numeric fields
* Standardized date/year information
* Addressed missing values
* Removed unnecessary empty records or fields

## Data Validation

After the cleaning process was completed, the resulting dataset was validated to ensure that:

* Column names were standardized.
* Expected columns were present.
* Data types were appropriate.
* Numeric fields contained valid numeric values.
* Categorical values were standardized.
* Missing values were addressed according to the project's cleaning rules.
* Empty rows and columns were removed where appropriate.
* The processed dataset could be successfully loaded for downstream analysis.

## Exploratory Data Analysis

After cleaning the dataset, exploratory analysis was performed to verify the structure and quality of the processed data.

Potential areas of exploration include:

* IMDb score distributions
* Movie release years
* Movie genres
* Worldwide income
* Number of votes
* Movie duration
* Content ratings
* Relationships between IMDb scores and other variables

## Key Findings & Insights

The data-cleaning process demonstrated several important characteristics of real-world datasets:

* Data may require substantial preprocessing before analysis can begin.
* Missing values require careful evaluation rather than automatic removal.
* Numeric values may contain symbols, units, or formatting inconsistencies that prevent direct analysis.
* Categorical variables can contain inconsistencies that affect grouping and aggregation.
* Proper data types are essential for accurate analysis.
* Data validation is an important part of the cleaning process.

## Challenges & Lessons Learned

### Challenges

Some of the primary challenges encountered during the project included:

* Identifying the different types of data-quality problems.
* Determining how each type of inconsistency should be handled.
* Converting improperly formatted numeric values.
* Standardizing categorical values without losing meaningful information.
* Determining appropriate strategies for missing data.

### Lessons Learned

This project reinforced several important data-analysis principles:

* Data cleaning is an essential part of the analytical process.
* There is rarely a single cleaning strategy that works for every column.
* Understanding the meaning of a variable is important before modifying its values.
* Exploratory analysis can reveal data-quality problems that are not immediately obvious.
* Clean data should be validated before being used for further analysis.

## Future Enhancements

Potential future enhancements include:

* Perform a more extensive exploratory analysis of the cleaned dataset.
* Create an interactive dashboard.
* Analyze relationships between IMDb scores, revenue, genres, and other movie characteristics.
* Perform statistical analysis.
* Develop a predictive model for IMDb scores.
* Create automated data-quality checks.
* Convert the cleaning workflow into a reusable Python data-cleaning pipeline.

## Data Source

**Messy IMDb Dataset — David Fuente Herraiz**

The dataset is available through Kaggle and was specifically created as an example of dirty data that requires preprocessing and cleaning.

## AI Usage

Generative AI was used in the following way(s):

* Assistance with README documentation.
* Code-generation assistance for selected data-cleaning tasks.
* Explanation of Python and Pandas functionality.
* Troubleshooting and debugging.
* Brainstorming approaches for handling data-quality issues.

AI-generated suggestions and code were reviewed, tested, and modified as necessary before being incorporated into the project.

## Author

Ashley A. Dehart

## License

MIT License