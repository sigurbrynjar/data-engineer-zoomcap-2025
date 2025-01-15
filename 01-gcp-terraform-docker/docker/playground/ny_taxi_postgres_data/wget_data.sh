#!/bin/bash

wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-01.parquet
wget https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv

conda activate dezoom  # The env we use
pip install pyarrow parquet-tools

# Create a csv file as well -- tutorial uses csv
# Parquet seem to be better as pandas is able to load the entire parquet file into memory
# but is unable to load the entire csv file, which means we must use iterators (which is still good practice)
python -c "
import pandas as pd
df = pd.read_parquet('yellow_tripdata_2021-01.parquet')
df.to_csv('yellow_tripdata_2021-01.csv', index=False)
"
