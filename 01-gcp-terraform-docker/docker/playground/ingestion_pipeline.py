import os
import pandas as pd
import argparse
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from time import time
from enum import Enum

# SCRIPT_DIR = os.path.dirname(__file__)

class WriteMode(Enum):
    REPLACE = 'replace'
    APPEND = 'append'
    FAIL = 'fail'

def download_data(file_url: str, data_dir='data') -> str:
    filename = file_url.split('/')[-1]
    filepath = os.path.join(data_dir, filename)
    os.system(f"mkdir -p {data_dir}; rm -rf {filename}")
    os.system(f"wget {file_url}")
    os.system(f"mv {filename} {filepath}")
    # create checksum
    os.system(f"md5sum {filepath} > {filepath}.md5")
    return filepath

def parquet_to_csv(filepath: str) -> str:
    csv_filepath = filepath.replace('.parquet', '.csv')
    os.system(f"parquet-tools csv {filepath} > {csv_filepath}")
    return csv_filepath

def load_data_iter(file_name, chunksize=100000) -> iter:
    # Load data
    data_iter = pd.read_csv(file_name, iterator=True, chunksize=chunksize)
    return data_iter

def create_db_engine(user, password, host, port, db_name) -> Engine:
    # Create database engine
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
    return engine

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    # Detect datetimes as strings and cast to datetime
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def upload_data_to_db(data_iter: iter, table_name, engine: Engine, mode: WriteMode.REPLACE) -> None:
    # Upload data to database
    df: pd.DataFrame = next(data_iter, None)
    if df is None:
        return

    # Drop table if exists and create new table with the same schema
    df = transform_data(df)
    if mode == WriteMode.REPLACE:
        df[:0].to_sql(table_name, con=engine, if_exists=WriteMode.REPLACE.value, index=False)
    t0 = time()
    n = 1
    print(f'Uploading chunk {n}...')
    df.to_sql(table_name, con=engine, if_exists=WriteMode.APPEND.value, index=False)
    print(f'Chunk uploaded in {time() - t0:.3f} seconds')

    while (df := next(data_iter, None)) is not None:
        # Upload data
        t0 = time()
        n += 1
        print(f'Uploading chunk {n}...')
        df = transform_data(df)
        df.to_sql(table_name, con=engine, if_exists=WriteMode.APPEND.value, index=False)
        print(f'Chunk uploaded in {time() - t0:.3f} seconds')


def main(args):
    file_url = args.url
    user = args.user
    password = args.password
    host = args.host
    port = args.port
    db_name = args.db
    table_name = args.table_name

    # Download data
    file_name = download_data(file_url)
    if file_name.endswith('.parquet'):
        file_name = parquet_to_csv(file_name)

    # Extract data
    data_iter = load_data_iter(file_name)

    # Create database engine
    engine = create_db_engine(user, password, host, port, db_name)

    # Upload data to database
    upload_data_to_db(data_iter, table_name, engine, WriteMode.REPLACE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload data to database')
    parser.add_argument('--url', type=str, help='Path to the data file',
                        default='https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-01.parquet')
    parser.add_argument('--host', type=str, help='Database host', default='localhost')
    parser.add_argument('--port', type=int, help='Database port', default=5432)
    parser.add_argument('--db', type=str, help='Database name', default='ny_taxi')
    parser.add_argument('--table-name', type=str, help='Table name', default='yellow_taxi_data')
    parser.add_argument('--user', type=str, help='Database user', default='root')
    parser.add_argument('--password', type=str, help='Database password', default='root')

    args = parser.parse_args()

    main(args)
