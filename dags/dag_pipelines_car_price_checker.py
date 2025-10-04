import sys
from pathlib import Path
PATH_THIS_FILE = Path(__file__).resolve()
PATH_PROJECT_ROOT = PATH_THIS_FILE.parent.parent
if str(PATH_PROJECT_ROOT) not in sys.path:
    # sys.path.append(str(PATH_PROJECT_ROOT))
    sys.path.insert(0, str(PATH_PROJECT_ROOT))  # insert at first position
import config.paths as paths
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
from python_mailer import send_email

# PATH_BASE = "/home/ubuntu/car_price_checker_api"
# PATH_VENV = f"{PATH_BASE}/.venv"
# PATH_PYTHON = f"{PATH_VENV}/bin/python3"

email_creds = str(paths.EMAIL_CREDENTIALS_FILE)

# ----------Callbacks for task events----------
def on_task_execute(context):
    dag_id = context['dag'].dag_id
    ti = context["task_instance"]
    subject = f"[START] {dag_id}.{ti.task_id}"
    body = (
        f"Task {ti.task_id} STARTED\n"
        f"DAG: {dag_id}\n"
        f"Run: {context['run_id']}\n"
        f"Execution date: {context['execution_date']}\n"
    )
    send_email(
        credentials=email_creds,
        subject=subject,
        body=body
    )


def on_task_success(context):
    dag_id = context['dag'].dag_id
    ti = context["task_instance"]
    duration = str((ti.end_date - ti.start_date)) if ti.end_date and ti.start_date else "Unknown"
    subject = f"[OK] {dag_id}.{ti.task_id}"
    body = (
        f"Task {ti.task_id} FINISHED OK\n"
        f"DAG: {dag_id}\n"
        f"Run: {context['run_id']}\n"
        f"Duration: {duration}\n"
    )
    send_email(
        credentials=email_creds,
        subject=subject,
        body=body
    )


def on_task_failure(context):
    dag_id = context["dag"].dag_id
    ti = context["task_instance"]
    subject = f"[FAIL] {dag_id}.{ti.task_id}"
    body = (
        f"Task {ti.task_id} FAILED\n"
        f"DAG: {dag_id}\n"
        f"Run: {context['run_id']}\n"
        f"Log URL: {ti.log_url}\n"
    )
    send_email(
        credentials=email_creds,
        subject=subject,
        body=body
    )


# ----------Callbacks for DAG events----------
def on_dag_success(context):
    dr = context["dag_run"]
    duration = str((dr.end_date - dr.start_date)) if dr.end_date and dr.start_date else "Uknown"
    subject = f"[DAG OK] {dr.dag_id}"
    body = (
        f"DAG {dr.dag_id} FINISHED OK\n"
        f"Run: {dr.run_id}\n"
        f"Duration: {duration}\n"
    )
    send_email(
        credentials=email_creds,
        subject=subject,
        body=body
    )


def on_dag_failure(context):
    dr = context["dag_run"]
    subject = f"[DAG FAIL] {dr.dag_id}"
    body = (
        f"DAG {dr.dag_id} FAILED\n"
        f"Run: {dr.run_id}\n"
    )
    send_email(
        credentials=email_creds,
        subject=subject,
        body=body
    )


# ----------DAG definition----------
with DAG(
    dag_id="dag_pipelines_car_price_checker",
    start_date=datetime(2025, 9, 1),    # start date
    schedule="0 0 1 * *",               # cron: each first day of the month at midnight
    catchup=True,                       # run past dates
    max_active_runs=1,                  # only one active run of this DAG at a time
    on_success_callback=on_dag_success, # callback on overall DAG success
    on_failure_callback=on_dag_failure, # callback on overall DAG failure
    tags=["car-price", "ml"],
) as dag:

    # Task 1: Ingestion
    ingestion = BashOperator(
        task_id="run_ingestion",
        bash_command=f"{paths.PYTHON} {paths.PIPELINES_DIR}/a_ingestion/ingestion_pipeline.py",
        env={"PYTHONPATH": str(paths.ROOT_DIR)},    # set PYTHONPATH to project root
        retries=10,                                 # number of retries if the task fails
        retry_delay=300,                            # delay between retries in seconds
        on_execute_callback=on_task_execute,        # callback on task start
        on_success_callback=on_task_success,        # callback on task success
        on_failure_callback=on_task_failure,        # callback on task failure
    )

    # Task 2: Training
    training = BashOperator(
        task_id="run_training",
        bash_command=f"{paths.PYTHON} {paths.PIPELINES_DIR}/c_training/training_pipeline.py",
        env={"PYTHONPATH": str(paths.ROOT_DIR)},  # set PYTHONPATH to project root
        retries=10,                               # number of retries if the task fails
        retry_delay=300,                          # delay between retries in seconds
        on_execute_callback=on_task_execute,      # callback on task start
        on_success_callback=on_task_success,      # callback on task success
        on_failure_callback=on_task_failure,      # callback on task failure
    )

    ingestion >> training  # dependency: first ingestion, then training
