# Anonymisation with Synthetic Data Tutorial

## Credit to others

*From the QUIPP-pipeline team:*
This tutorial is based on the work of the Open Data Institute.
The original version of the tutorial can be found [here](https://github.com/theodi/synthetic-data-tutorial).
We have shortened the tutorial to focus on the data generation process, as we'll use the resulting dataset to illustrate the different methods of data synthesis found in the `methods` directory of this repository.

*From the ODI:*
This tutorial is inspired by the [NHS England and ODI Leeds' research](https://odileeds.org/events/synae/) in creating a synthetic dataset from NHS England's accident and emergency admissions. Please do read about their project, as it's really interesting and great for learning about the benefits and risks in creating synthetic data.

## Setup

Install required dependent libraries.
We'll use conda for this example.

```bash
cd /path/to/repo/generators/odi-nhs-ae
conda env create -f environment.yml
```

Next we'll go through how to create and de-identify the dataset.

There are small differences between the code presented here and what's in the Python scripts but it's mostly down to variable naming.
I'd encourage you to run, edit and play with the code locally.

## Generate mock NHS A&E dataset

The data already exists in `datasets/generated/nhs_ae_mock.csv` so feel free to browse that.
You can generate your own fresh dataset using the `generate.py` script.

To do this, you'll need to download one dataset first.
It's a list of all postcodes in London.
You can find it at this page on [doogal.co.uk](https://www.doogal.co.uk/PostcodeDownloads.php), at the _London_ link under the _By English region_ section.
Download the file (133MB) and place it in the `data` directory using the following command:
```bash
curl -o "./data/London postcodes.csv" https://www.doogal.co.uk/UKPostcodesCSV.ashx?region=E12000007
```
The `.gitignore` file in the current folder will prevent the postcodes file being accidentally added to the version control system.

Then, to generate the data, from the project root directory run the `generate.py` script.

```bash
python generate.py
```

Voila! You'll now see a new `hospital_ae_data.csv` file in the `/data` directory. Open it up and have a browse. It's contains the following columns:

- **Health Service ID**: NHS number of the admitted patient  
- **Age**: age of patient
- **Time in A&E (mins)**: time in minutes of how long the patient spent in A&E. This is generated to correlate with the age of the patient.
- **Hospital**: which hospital admitted the patient - with some hospitals being more prevalent in the data than others
- **Arrival Time**: what time and date the patient was admitted - with weekends as busier and and a different peak time for each day
- **Treatment**: what the person was treated for - with certain treatments being more common than others
- **Gender**: patient gender - based on [NHS patient gender codes](https://www.datadictionary.nhs.uk/data_dictionary/attributes/p/person/person_gender_code_de.asp?shownav=1)
- **Postcode**: postcode of patient - random, in use, London postcodes extracted from the `London postcodes.csv` file.

We can see this dataset obviously contains some personal information. For instance, if we knew roughly the time a neighbour went to A&E we could use their postcode to figure out exactly what ailment they went in with. Or, if a list of people's Health Service ID's were to be leaked in future, lots of people could be re-identified.

Because of this, we'll need to take some de-identification steps.

## De-identification

For this stage, we're going to be loosely following the de-identification techniques used by Jonathan Pearson of NHS England, and described in a blog post about [creating its own synthetic data](https://odileeds.org/blog/2019-01-24-exploring-methods-for-creating-synthetic-a-e-data).

If you look in `deidentify.py` you'll see the full code of all de-identification steps. You can run this code easily.

```bash
python deidentify.py
```

It takes the `QUIPP-pipeline/datasets/generated/odi_nhs_ae/hospital_ae_data.csv` file, runs the steps, and saves the new dataset to `hospital_ae_data_deidentify.csv`.

Breaking down each of these steps.
It first loads the `QUIPP-pipeline/datasets/generated/odi_nhs_ae/nhs_ae_data.csv` file in to the Pandas DataFrame as `hospital_ae_df`.

```python
# _df is a common way to refer to a Pandas DataFrame object
hospital_ae_df = pd.read_csv(filepaths.hospital_ae_data)
```

(`filepaths.py` is, surprise, surprise, where all the filepaths are listed)

### Remove Health Service ID numbers

Health Service ID numbers are direct identifiers and should be removed. So we'll simply drop the entire column.

```python
hospital_ae_df = hospital_ae_df.drop('Health Service ID', 1)
```

### Where a patient lives

Pseudo-identifiers, also known as [quasi-identifiers](https://en.wikipedia.org/wiki/Quasi-identifier), are pieces of information that don't directly identify people but can used with other information to identify a person. If we were to take the age, postcode and gender of a person we could combine these and check the dataset to see what that person was treated for in A&E.

The data scientist from NHS England, Jonathan Pearson, describes this in the blog post:

> I started with the postcode of the patients resident lower super output area (LSOA). This is a geographical definition with an average of 1500 residents created to make reporting in England and Wales easier. I wanted to keep some basic information about the area where the patient lives whilst completely removing any information regarding any actual postcode. A key variable in health care inequalities is the patients Index of Multiple deprivation (IMD) decile (broad measure of relative deprivation) which gives an average ranked value for each LSOA. By replacing the patients resident postcode with an IMD decile I have kept a key bit of information whilst making this field non-identifiable.

We'll do just the same with our dataset.

First we'll map the rows' postcodes to their LSOA and then drop the postcodes column.

```python
postcodes_df = pd.read_csv(filepaths.postcodes_london)
hospital_ae_df = pd.merge(
    hospital_ae_df,
    postcodes_df[['Postcode', 'Lower layer super output area']],
    on='Postcode'
)
hospital_ae_df = hospital_ae_df.drop('Postcode', 1)
```

Then we'll add a mapped column of "Index of Multiple Deprivation" column for each entry's LSOA.

```python
hospital_ae_df = pd.merge(
    hospital_ae_df,
    postcodes_df[['Lower layer super output area', 'Index of Multiple Deprivation']].drop_duplicates(),
    on='Lower layer super output area'
)
```

Next calculate the decile bins for the IMDs by taking all the IMDs from large list of London. We'll use the Pandas `qcut` (quantile cut), function for this.

```python
_, bins = pd.qcut(
    postcodes_df['Index of Multiple Deprivation'],
    10,
    retbins=True,
    labels=False
)
```

Then we'll use those decile `bins` to map each row's IMD to its IMD decile.

```python
# add +1 to get deciles from 1 to 10 (not 0 to 9)
hospital_ae_df['Index of Multiple Deprivation Decile'] = pd.cut(
    hospital_ae_df['Index of Multiple Deprivation'],
    bins=bins,
    labels=False,
    include_lowest=True) + 1
```

And finally drop the columns we no longer need.

```python
hospital_ae_df = hospital_ae_df.drop('Index of Multiple Deprivation', 1)
hospital_ae_df = hospital_ae_df.drop('Lower layer super output area', 1)
```

### Individual hospitals

The data scientist at NHS England masked individual hospitals giving the following reason.

> As each hospital has its own complex case mix and health system, using these data to identify poor performance or possible improvements would be invalid and un-helpful. Therefore, I decided to replace the hospital code with a random number.

So we'll do as they did, replacing hospitals with a random six-digit ID.

```python
hospitals = hospital_ae_df['Hospital'].unique().tolist()
random.shuffle(hospitals)
hospitals_map = {
    hospital : ''.join(random.choices(string.digits, k=6))
    for hospital in hospitals
}
hospital_ae_df['Hospital ID'] = hospital_ae_df['Hospital'].map(hospitals_map)
```

And remove the `Hospital` column.

```python
hospital_ae_df = hospital_ae_df.drop('Hospital', 1)
```

### Time in the data

> The next obvious step was to simplify some of the time information I have available as health care system analysis doesn't need to be responsive enough to work on a second and minute basis. Thus, I removed the time information from the 'arrival date', mapped the 'arrival time' into 4-hour chunks

First we'll split the `Arrival Time` column in to `Arrival Date` and `Arrival Hour`.

```python
arrival_times = pd.to_datetime(hospital_ae_df['Arrival Time'])
hospital_ae_df['Arrival Date'] = arrival_times.dt.strftime('%Y-%m-%d')
hospital_ae_df['Arrival Hour'] = arrival_times.dt.hour
hospital_ae_df = hospital_ae_df.drop('Arrival Time', 1)
```

Then we'll map the hours to 4-hour chunks and drop the `Arrival Hour` column.

```python
hospital_ae_df['Arrival hour range'] = pd.cut(
    hospital_ae_df['Arrival Hour'],
    bins=[0, 4, 8, 12, 16, 20, 24],
    labels=['00-03', '04-07', '08-11', '12-15', '16-19', '20-23'],
    include_lowest=True
)
hospital_ae_df = hospital_ae_df.drop('Arrival Hour', 1)
```

### Patient demographics

> I decided to only include records with a sex of male or female in order to reduce risk of re identification through low numbers.

```python
hospital_ae_df = hospital_ae_df[hospital_ae_df['Gender'].isin(['Male', 'Female'])]
```

> For the patients age it is common practice to group these into bands and so I've used a standard set - 1-17, 18-24, 25-44, 45-64, 65-84, and 85+ - which although are non-uniform are well used segments defining different average health care usage.

```python
hospital_ae_df['Age bracket'] = pd.cut(
    hospital_ae_df['Age'],
    bins=[0, 18, 25, 45, 65, 85, 150],
    labels=['0-17', '18-24', '25-44', '45-64', '65-84', '85-'],
    include_lowest=True
)
hospital_ae_df = hospital_ae_df.drop('Age', 1)
```

That's all the steps we'll take. We'll finally save our new de-identified dataset to `hospital_ae_data_deidentify.csv`.

```python
hospital_ae_df.to_csv(filepaths.hospital_ae_data_deidentify, index=False)
```

## Data Description

Finally, we create a description of our generated data, defining the datatypes and which are the categorical variables.
We show below the desription of the deidentified dataset, which we save alongside the data file as `hospital_ae_data_deidentify.json`.
We also save a similar file for the data before we performed the deidentification process.
```json
[ {"name": "Time in A&E (mins)",
   "type": "Integer",
   "categorical": "False"},
  {"name": "Treatment",
   "type": "String",
   "categorical": "True"},
  {"name": "Gender",
   "type": "String",
   "categorical": "True"},
  {"name": "Index of Multiple Deprivation Decile",
   "type": "Integer",
   "categorical": "False"},
  {"name": "Hospital ID",
   "type": "String",
   "categorical": "True"},
  {"name": "Arrival Date",
   "type": "String",
   "categorical": "True"},
  {"name": "Arrival hour range",
   "type": "String",
   "categorical": "True"},
  {"name": "Age bracket",
   "type": "String",
   "categorical": "True"} ]
```

