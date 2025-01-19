# Module 1: Docker, PostgreSQL, pgAdmin, GCP, Terraform


## Ingestion Pipeline

```bash
python ingestion_pipeline.py \
    --user=root \
    --password=root \
    --host=localhost \
    --port=5432 \
    --db=ny_taxi \
    --table-name=yellow_taxi_trips \
    --url="https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2021-01.csv" 
```

```bash
docker build -t taxi_ingest:v001 .
docker run -it \
    --network=pg-network \
    taxi_ingest:v001 \
    --user=root \
    --password=root \
    --host=localhost \
    --port=5432 \
    --db=ny_taxi \
    --table-name=yellow_taxi_trips \
    --url="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-01.parquet" 
```