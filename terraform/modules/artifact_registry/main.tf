resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = var.repository_name
  description   = "Docker repository for ${var.repository_name}"
  format        = "DOCKER"
}
