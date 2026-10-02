from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = "/opt/airflow/project"


with DAG(
    dag_id="adaptive_ticket_routing_mlops",
    start_date=datetime(2026, 9, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "ticket-routing"],
) as dag:

    train_model = BashOperator(
        task_id="train_svm",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python src/models/train_svm.py"
        ),
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python src/evaluation/evaluate.py"
        ),
    )

    train_model >> evaluate_model