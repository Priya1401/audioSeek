# Enable required APIs
resource "google_project_service" "container" {
  project            = var.project_id
  service            = "container.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute" {
  project            = var.project_id
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  project            = var.project_id
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

module "network" {
  source       = "./modules/network"
  service_name = var.service_name
  region       = var.region
  
  depends_on = [google_project_service.compute]
}

module "artifact_registry" {
  source          = "./modules/artifact_registry"
  repository_name = var.artifact_registry_name
  region          = var.region
  project_id      = var.project_id
  
  depends_on = [google_project_service.artifactregistry]
}

module "gke" {
  source          = "./modules/gke"
  service_name    = var.service_name
  network_name    = module.network.network_name
  subnetwork_name = module.network.subnetwork_name
  region          = var.region
  
  depends_on = [google_project_service.container]
}
