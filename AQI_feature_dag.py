from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'talha',
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    dag_id='aqi_training_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2025, 10, 28),
    catchup=False,
    tags=['aqi', 'training'],
) as dag:

    run_train = BashOperator(
        task_id='run_training_pipeline',
        bash_command='python /path/to/AQIAgent.py train',
        do_xcom_push=False
    )