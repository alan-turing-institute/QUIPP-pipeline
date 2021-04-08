"Costa Rican Household Poverty Level Prediction" dataset found in Kaggle [here](https://www.kaggle.com/c/costa-rican-household-poverty-prediction/data).

The dataset needs to be downloaded and placed in this folder (`generators/household_poverty/data`) before using it. You can download the dataset (only the training set `train.csv`) manually from the link above or:
- Install the Kaggle API (by running `pip install kaggle`)
- Add your API token to your local `~/.kaggle` folder (the token can be generated in the Kaggle website from your account settings)
- Run `kaggle competitions download -c costa-rican-household-poverty-prediction`
- Unzip the contents and copy the `train.csv` file into `datasets/household_poverty/data` under the QUIPP root directory

The dataset is not part of the QUIPP repository because its publication outside the Kaggle website is not permitted (see competition rules [here](https://www.kaggle.com/c/costa-rican-household-poverty-prediction/rules))