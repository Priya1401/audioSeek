resource "google_service_account" "airflow" {
  account_id   = "audioseek-airflow"
  display_name = "audioseek-airflow"
}

resource "google_project_iam_member" "airflow_roles" {
  for_each = toset([
    "roles/run.admin",
    "roles/compute.storageAdmin",
    "roles/logging.logWriter"
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.airflow.email}"
}

resource "google_service_account" "gitconnect" {
  account_id   = "audioseekgitconnect"
  display_name = "audioseekgitconnect"
}

resource "google_project_iam_member" "gitconnect_roles" {
  for_each = toset([
    "roles/artifactregistry.reader",
    "roles/artifactregistry.writer",
    "roles/cloudbuild.builds.editor", # Cloud Build Service Account
    "roles/run.admin",
    "roles/container.admin",
    "roles/container.developer",
    "roles/secretmanager.secretAccessor",
    "roles/iam.serviceAccountUser",
    "roles/storage.admin"
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.gitconnect.email}"
}

resource "google_service_account" "dvc_storage" {
  account_id   = "dvc-storage"
  display_name = "DVC Storage Service Account"
}

resource "google_project_iam_member" "dvc_storage_roles" {
  for_each = toset([
    "roles/datastore.user",
    "roles/run.admin",
    "roles/firebase.admin",
    "roles/storage.objectAdmin"
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.dvc_storage.email}"
}

# Allow public read access to the 'images/' folder in the bucket
resource "google_storage_bucket_iam_member" "public_images" {
  bucket = "audioseek-bucket"
  role   = "roles/storage.objectViewer"
  member = "allUsers"

  condition {
    title       = "public_images_only"
    description = "Allow public access only to objects in the images/ folder"
    expression  = "resource.name.startsWith('projects/_/buckets/audioseek-bucket/objects/images/')"
  }
}
