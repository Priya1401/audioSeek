resource "google_cloud_run_service" "frontend" {
  name     = "audioseek-frontend"
  location = var.region
  project  = var.project_id

  template {
    spec {
      containers {
        image = "us-docker.pkg.dev/cloudrun/container/hello" # Placeholder, managed by deploy.sh
        
        resources {
          limits = {
            cpu    = "1000m"
            memory = "512Mi"
          }
        }
      }
      service_account_name = google_service_account.gitconnect.email
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  lifecycle {
    ignore_changes = [
      template, # Ignore changes to the template (image, env vars) as they are managed by deploy.sh
    ]
  }

  depends_on = [google_project_service.run]
}

# Allow unauthenticated access
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_service.frontend.name
  location = google_cloud_run_service.frontend.location
  project  = google_cloud_run_service.frontend.project
  role     = "roles/run.invoker"
  member   = "allUsers"
}
