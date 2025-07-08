import streamlit as st

st.markdown("""
### Hemisphere Classification 

The following function classifies countries into either the **Northern** or **Southern Hemisphere** based on a predefined mapping. This is useful for feature engineering in geographical or climate-related data analysis.

#### 🔧 Function: `classify_hemisphere(df)`

```python
def classify_hemisphere(df):
    ...
```

This function takes as input a **Pandas DataFrame** that must contain at least two columns:
- `'country_name'`: The name of the country.
- `'hemisphere'`: A column to be updated with the hemisphere classification.

---

### 🗺️ How It Works

1. **Mapping Definition**
   
   A dictionary called `hemisphere_mapping` assigns each country to its corresponding hemisphere:

   ```python
   hemisphere_mapping = {
       'Argentina': 'Southern',
       'Australia': 'Southern',
       'Austria': 'Northern',
       ...
   }
   ```

2. **Validation Step**

   The function checks whether all unique country names in the `'country_name'` column are present in the `hemisphere_mapping`.  
   If any country is missing, it raises an error listing those countries:

   ```python
   if missing_countries:
       raise ValueError(f"Countries not in hemisphere mapping: {missing_countries}")
   ```

3. **Mapping Application**

   Using `.map()`, the function creates or updates the `'hemisphere'` column by looking up each country in the dictionary:

   ```python
   df['hemisphere'] = df['country_name'].map(hemisphere_mapping)
   ```

4. **Return Value**

   Returns the original DataFrame with the updated `'hemisphere'` column.

---

### ✅ Example Output

| country_name     | hemisphere |
|------------------|------------|
| Australia        | Southern   |
| Germany          | Northern   |
| Brazil           | Southern   |

---

### 💡 Use Case in Data Science

This kind of categorical mapping is a form of **feature engineering**, where raw country names are converted into a more general geographical feature (`hemisphere`). This can improve model performance when hemispheric patterns (e.g., seasonal differences) are relevant to the target variable.

---
for full code, look at [this](https://github.com/Losif01/World-Energy/blob/main/notebooks/EDA.ipynb)
""")

st.markdown("""

# 🌐 Electricity Data Enrichment with World Bank GDP Indicators

This script enriches a dataset (presumably containing country names and dates) by fetching **GDP** and **GDP per capita** data from the **World Bank API** and appending it to the original DataFrame.

---

## 🧰 Libraries Used

```python
import pandas as pd
import requests
from functools import lru_cache
from tqdm import tqdm
import pycountry
import time
```

- `pandas`: For data manipulation.
- `requests`: To fetch data from the World Bank API.
- `functools.lru_cache`: To cache API responses and avoid redundant calls.
- `tqdm`: To show progress bars during looping.
- `pycountry`: To convert country names into ISO-2 codes.
- `time`: To add small delays between API calls to avoid rate limiting.

---

## 🔁 Helper Function: `name_to_iso2(name)`

```python
def name_to_iso2(name: str) -> str | None:
    ...
```

### ✅ Purpose:
Converts country names into their 2-letter **ISO country codes** (e.g., "Germany" → "de").

### 🔍 Details:
- Uses `pycountry.countries.lookup()` for automatic conversion.
- Includes a **manual mapping dictionary** for known edge cases like:
  - `"Czech Republic"` → `"cz"`
  - `"Korea"` → `"kr"`
  - `"United States"` → `"us"`

>Returns `None` if no match is found.

---

## 🌐 API Fetching Function: `fetch_indicator_series(iso, indicator)`

```python
@lru_cache(maxsize=None)
def fetch_indicator_series(iso: str, indicator: str, start=1990, end=2025):
    ...
```

### ✅ Purpose:
Fetches a specific economic indicator from the [World Bank API](https://api.worldbank.org/v2/country/) for a given country (by ISO code).

### 📊 Example Indicator Codes:
- `"NY.GDP.MKTP.CD"`: GDP (current US$)
- `"NY.GDP.PCAP.CD"`: GDP per capita (current US$)

### 🚀 How It Works:
- Constructs a URL like:
  ```
  https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}?format=json&...
  ```
- Returns a dictionary of `{year: value}` pairs.

### ⚠️ Notes:
- Uses caching (`@lru_cache`) to avoid repeated API calls.
- Adds a small delay (`time.sleep(0.15)`) to prevent hitting API rate limits.

---

## 📥 Main Loop: Fetching GDP and GDP Per Capita

```python
for _, row in tqdm(df.iterrows(), total=len(df)):
    ...
```

### ✅ Purpose:
Iterates over each row in the DataFrame to:
1. Convert the country name to an ISO code.
2. Fetch GDP and GDP per capita for that country/year.
3. Append values to lists which are later added as new columns.

### 🧠 Key Steps:
- Extract `country_name` and `year` from each row.
- Use `name_to_iso2()` to get ISO code.
- If not cached, fetch GDP data using `fetch_indicator_series()`.
- Append corresponding year’s GDP and GDP per capita or `None`.

---

## 📦 Final Output: New Columns Added

```python
df["GDP"] = gdp_values
df["GDP_per_capita"] = gdp_per_capita_values
```

### 📋 Result:
| country_name | date       | year | GDP         | GDP_per_capita |
|--------------|------------|------|-------------|----------------|
| Germany      | 2020-01-01 | 2020 | 3806000000000| 45780          |

>Now you can analyze how electricity usage or other metrics relate to macroeconomic indicators!

---

## 💡 Why This Matters

This type of **feature engineering** adds valuable socioeconomic context to your dataset. By linking country-level economic data to your existing records, you can:

- Build more accurate predictive models.
- Perform deeper exploratory data analysis.
- Understand regional disparities in development and energy use.

---
""")

st.markdown("""

# ☀️  Daylight Hours and Nuclear Plant Outage Detection

This script adds two new features to the dataset:

1. **Daylight Hours** – based on latitude, month, and year.
2. **Nuclear Plant Status (Inferred Outage)** – based on drops in nuclear electricity generation.

These features can be valuable for understanding seasonal patterns or detecting anomalies in energy production.

---

## 🧰 Libraries Used

```python
import pandas as pd
import numpy as np
from astral import LocationInfo
from astral.sun import sun
import datetime
```

- `pandas` / `numpy`: For data manipulation and handling missing values.
- `astral`: To calculate sunrise and sunset times.
- `datetime`: To handle date logic.
- `LocationInfo` and `sun` from `astral`: Core tools for daylight calculations.

---

## 🔁 Function: `calculate_daylight_hours(latitude, year, month)`

```python
def calculate_daylight_hours(latitude, year, month):
    ...
```

### ✅ Purpose:
Calculate approximate **hours of daylight** for a given location and time of year.

### 🌍 Key Features:
- Uses the 15th of the month as a representative day.
- Handles edge cases near the **Arctic/Antarctic Circles** where there's **continuous daylight or darkness** during summer/winter.
- Returns daylight hours as a float (e.g., `24.0`, `0.0`, or `8.5`).

### ⚠️ Error Handling:
- If the sun doesn’t rise or set due to polar conditions, it returns:
  - `24.0` for continuous daylight
  - `0.0` for continuous darkness
- Catches exceptions and returns `NaN` with an error message if something goes wrong.

---

## 📈 Step-by-Step Execution

### 1. Extract Unique Country-Year-Month Combinations

```python
unique_daylight_combinations = df[['country_name', 'year', 'month']].drop_duplicates()
```

>Prevents redundant daylight calculations by grouping rows by country, year, and month.

---

### 2. Merge Latitude Data

```python
daylight_df = pd.merge(unique_daylight_combinations, country_lat_df, on='country_name', how='left')
```

>Assumes you have a separate DataFrame (`country_lat_df`) that maps country names to their latitudes.

---

### 3. Calculate Daylight Hours

```python
daylight_df['Daylight Hours'] = daylight_df.apply(
    lambda row: calculate_daylight_hours(row['latitude'], row['year'], row['month']), axis=1
)
```

>Applies the function to each unique combination and stores results in a new column.

---

### 4. Merge Back into Main DataFrame

```python
df = pd.merge(df, daylight_df, on=['country_name', 'year', 'month'], how='left')
```

>Adds the computed daylight hours back to the original dataset.

---

## ⚛️ Feature: Inferred Nuclear Plant Outages

This part tries to **detect potential nuclear plant outages** by looking at unusual drops in nuclear power production.

### 🔍 Logic Summary:
1. Identify countries that produce nuclear energy.
2. For each such country:
   - Compute a **baseline production level** (75th percentile).
   - Flag any month where production falls below **25% of baseline** as a possible outage.
3. Add a binary flag column:
   - `0`: No inferred outage
   - `1`: Potential outage detected

### 📊 Example Output:

| country_name | date       | value (nuclear) | Baseline | Threshold | Outage Flag |
|--------------|------------|------------------|----------|-----------|-------------|
| Canada       | 2020-03-01 | 6000             | 8000     | 2000      | 0           |
| Canada       | 2020-04-01 | 1500             | 8000     | 2000      | 1           |

---

## 📋 Final Output Sample

The final script prints sample outputs for high-latitude and mid-latitude countries like:

- Iceland (high latitude, large daylight variation)
- Canada
- Argentina

It also shows the updated columns:

```python
[
    'country_name',
    'date',
    'Monthly Temperature Averages',
    'Daylight Hours',
    'Nuclear Plant Status (Inferred Outage)'
]
```

---
""")

st.sidebar.markdown("for full code, look at [this](https://github.com/Losif01/World-Energy/blob/main/notebooks/EDA.ipynb)")