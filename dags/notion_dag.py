from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
import os

script_path = os.path.join(os.path.dirname(__file__), "notionapi.py")

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}


# Define the DAG
with DAG(
    dag_id='run_external_python_script',
    default_args=default_args,
    description='A DAG to run a Python script in another workspace',
    schedule='0 9 * * 0',  # Set to None for manual triggering
    start_date=datetime(2025, 8, 23),
    catchup=False,
    tags=['example'],
) as dag:

    # Task to run the Python script
    run_python_script = BashOperator(
        task_id='run_python_script',
        bash_command=f"python3 {script_path}"
    )

    # Add more tasks if needed

    # Define task dependencies (if applicable)
    run_python_script
