import os
import time
import requests
import mlflow

from config_mlflow import MLFLOW_EXPERIMENT_NAME

# --- Config from environment (override these in GitHub Actions env) ---
BASE_URL = os.getenv("TEXTPROC_BASE_URL", "http://localhost:8001")
SMOKE_BOOK_ID = os.getenv("SMOKE_BOOK_ID", "ci-test-book")
SMOKE_BOOK_FOLDER = os.getenv("SMOKE_BOOK_FOLDER", "data/ci_smoke_book")

# Note: we use env var for tracking URI so CI can talk to localhost:5000
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def wait_for_health(timeout_sec: int = 120, interval_sec: int = 5):
    """
    Poll /health until it returns 200 or timeout.
    Raises an exception if service never becomes healthy.
    """
    deadline = time.time() + timeout_sec
    last_exception = None

    while time.time() < deadline:
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=10)
            if resp.status_code == 200:
                return resp
        except Exception as e:
            last_exception = e
        time.sleep(interval_sec)

    raise RuntimeError(f"Service /health did not become ready in {timeout_sec}s. Last error: {last_exception}")


def run_smoke_test():
    # --- Configure MLflow for CI context ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    run = mlflow.start_run(run_name=f"ci_smoke_{SMOKE_BOOK_ID}")

    try:
        mlflow.log_param("base_url", BASE_URL)
        mlflow.log_param("book_id", SMOKE_BOOK_ID)
        mlflow.log_param("folder_path", SMOKE_BOOK_FOLDER)

        # 1. Wait for /health
        t0 = time.time()
        health_resp = wait_for_health()
        t1 = time.time()

        mlflow.log_metric("health_status_code", health_resp.status_code)
        mlflow.log_metric("health_wait_sec", t1 - t0)
        try:
            mlflow.log_dict(health_resp.json(), "health_response.json")
        except Exception:
            pass  # if /health is not JSON, skip

        # 2. Call /process-full to ingest a tiny test book
        #    NOTE: inside the container, this folder must exist:
        #    /app/data/ci_smoke_book/*.txt  (on host: Data-Pipeline/working_data/ci_smoke_book/*.txt)
        process_payload = {
            "book_id": SMOKE_BOOK_ID,
            "file_path": None,
            "file_paths": None,
            "folder_path": SMOKE_BOOK_FOLDER,
            "target_tokens": 200,
            "overlap_tokens": 50,
            "add_to_vector_db": True
        }

        t2 = time.time()
        pf_resp = requests.post(
            f"{BASE_URL}/process-full",
            json=process_payload,
            timeout=600
        )
        t3 = time.time()

        mlflow.log_metric("process_full_status_code", pf_resp.status_code)
        mlflow.log_metric("process_full_latency_sec", t3 - t2)
        pf_resp.raise_for_status()

        pf_data = pf_resp.json()
        mlflow.log_dict(pf_data, "process_full_response.json")

        for key in ("chunks_count", "chapters_count", "entities_count", "embeddings_count"):
            if key in pf_data:
                mlflow.log_metric(f"process_full_{key}", pf_data[key])

        # 3. Call /qa/ask using the just-ingested book
        qa_payload = {
            "book_id": SMOKE_BOOK_ID,
            "query": "Give a brief 2-sentence summary of this book.",
            "top_k": 5
        }

        t4 = time.time()
        qa_resp = requests.post(
            f"{BASE_URL}/qa/ask",
            json=qa_payload,
            timeout=180
        )
        t5 = time.time()

        mlflow.log_metric("qa_status_code", qa_resp.status_code)
        mlflow.log_metric("qa_latency_sec", t5 - t4)
        qa_resp.raise_for_status()

        qa_data = qa_resp.json()
        mlflow.log_dict(qa_data, "qa_response.json")
        mlflow.log_metric("qa_answer_len", len(qa_data.get("answer", "") or ""))

        mlflow.set_tag("status", "success")
        mlflow.end_run(status="FINISHED")

    except Exception as e:
        # Record failure in MLflow and re-raise so CI job fails
        mlflow.log_param("error", str(e))
        mlflow.set_tag("status", "failed")
        mlflow.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    run_smoke_test()
