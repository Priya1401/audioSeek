
import logging
from google.cloud import firestore
from domain.models_firestore import User, Job, JobStatus
from typing import Optional, List
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class FirestoreService:
    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.database_id = os.getenv("FIRESTORE_DB", "(default)")
        self._mock_db = {}
        self._mock_users = {}
        
        if not self.project_id:
            logger.warning("GCP_PROJECT_ID not set. Using in-memory mock for Firestore.")
            self.db = None
            self.use_mock = True
        else:
            try:
                self.db = firestore.Client(project=self.project_id, database=self.database_id)
                logger.info(f"Firestore initialized for project: {self.project_id}, database: {self.database_id}")
                self.use_mock = False
            except Exception as e:
                logger.error(f"Failed to initialize Firestore: {e}. Falling back to mock.")
                self.db = None
                self.use_mock = True

    def _check_db(self):
        if not self.db and not self.use_mock:
            raise RuntimeError("Firestore client is not initialized.")

    # ---------------- USER OPERATIONS ----------------

    def get_user(self, email: str) -> Optional[User]:
        if self.use_mock:
            data = self._mock_users.get(email)
            return User(**data) if data else None
            
        self._check_db()
        doc_ref = self.db.collection("users").document(email)
        doc = doc_ref.get()
        if doc.exists:
            return User(**doc.to_dict())
        return None

    def create_or_update_user(self, user: User):
        if self.use_mock:
            self._mock_users[user.email] = user.dict()
            return

        self._check_db()
        doc_ref = self.db.collection("users").document(user.email)
        doc_ref.set(user.dict(), merge=True)

    # ---------------- JOB OPERATIONS ----------------

    def create_job(self, job: Job):
        if self.use_mock:
            self._mock_db[job.job_id] = job.dict()
            return

        self._check_db()
        doc_ref = self.db.collection("jobs").document(job.job_id)
        doc_ref.set(job.dict())

    def get_job(self, job_id: str) -> Optional[Job]:
        if self.use_mock:
            data = self._mock_db.get(job_id)
            return Job(**data) if data else None

        self._check_db()
        doc_ref = self.db.collection("jobs").document(job_id)
        doc = doc_ref.get()
        if doc.exists:
            return Job(**doc.to_dict())
        return None

    def update_job_status(self, job_id: str, status: JobStatus, message: str = None, progress: float = None, result: dict = None, error: str = None):
        if self.use_mock:
            if job_id in self._mock_db:
                self._mock_db[job_id]["status"] = status
                self._mock_db[job_id]["updated_at"] = datetime.utcnow()
                if message: self._mock_db[job_id]["message"] = message
                if progress is not None: self._mock_db[job_id]["progress"] = progress
                if result: self._mock_db[job_id]["result"] = result
                if error: self._mock_db[job_id]["error"] = error
            return

        self._check_db()
        doc_ref = self.db.collection("jobs").document(job_id)
        
        updates = {
            "updated_at": datetime.utcnow(),
            "status": status
        }
        if message:
            updates["message"] = message
        if progress is not None:
            updates["progress"] = progress
        if result:
            updates["result"] = result
        if error:
            updates["error"] = error
            
        doc_ref.update(updates)

    def get_user_jobs(self, user_email: str) -> List[Job]:
        if self.use_mock:
            jobs = [Job(**j) for j in self._mock_db.values() if j["user_email"] == user_email]
            return sorted(jobs, key=lambda x: x.created_at, reverse=True)

        self._check_db()
        jobs_ref = self.db.collection("jobs")
        query = jobs_ref.where("user_email", "==", user_email).order_by("created_at", direction=firestore.Query.DESCENDING)
        results = query.stream()
        return [Job(**doc.to_dict()) for doc in results]

    def get_all_jobs_stats(self) -> dict:
        """Get aggregate stats for all jobs (Admin only)"""
        stats = {
            "total": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        }
        
        if self.use_mock:
            stats["total"] = len(self._mock_db)
            for job in self._mock_db.values():
                status = job.get("status")
                if status in stats:
                    stats[status] += 1
            return stats

        self._check_db()
        # Note: In high-scale Firestore, counting all docs is expensive. 
        # For this scale, it's fine. Ideally use distributed counters.
        jobs_ref = self.db.collection("jobs")
        docs = jobs_ref.stream()
        
        for doc in docs:
            data = doc.to_dict()
            stats["total"] += 1
            status = data.get("status")
            if status in stats:
                stats[status] += 1
                
        return stats

    def get_stale_jobs(self, threshold_minutes: int = 75) -> List[Job]:
        """
        Get jobs that have been in PROCESSING state for longer than threshold_minutes.
        """
        from datetime import timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=threshold_minutes)
        
        if self.use_mock:
            stale_jobs = []
            for j in self._mock_db.values():
                if j["status"] == JobStatus.PROCESSING:
                    updated_at = j.get("updated_at")
                    # Handle mock data where updated_at might be string or datetime
                    if isinstance(updated_at, str):
                        try:
                            updated_at = datetime.fromisoformat(updated_at)
                        except:
                            continue
                            
                    if updated_at and updated_at < cutoff_time:
                        stale_jobs.append(Job(**j))
            return stale_jobs

        self._check_db()
        jobs_ref = self.db.collection("jobs")
        # Query: status == 'processing' AND updated_at < cutoff_time
        # Note: This requires a composite index in Firestore.
        query = jobs_ref.where("status", "==", JobStatus.PROCESSING).where("updated_at", "<", cutoff_time)
        results = query.stream()
        return [Job(**doc.to_dict()) for doc in results]

    def get_jobs_by_status(self, status: str = "processing") -> List[dict]:
        """Get jobs by status (processing, completed, failed) with details"""
        if self.use_mock:
            jobs = [j for j in self._mock_db.values() if j["status"] == status]
            return sorted(jobs, key=lambda x: x["created_at"], reverse=True)

        self._check_db()
        jobs_ref = self.db.collection("jobs")
        
        # Filter by the provided status
        query = jobs_ref.where("status", "==", status)
        results = query.stream()
        
        # Sort in memory to avoid Firestore Composite Index requirements
        jobs = [doc.to_dict() for doc in results]
        return sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True)

# Global instance
firestore_db = FirestoreService()
