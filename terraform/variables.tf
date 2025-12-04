variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "ie7374-475102"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-east1"
}

variable "service_name" {
  description = "Name of the service"
  type        = string
  default     = "audioseek"
}

variable "artifact_registry_name" {
  description = "Name of the artifact registry repository"
  type        = string
  default     = "audioseek-repo"
}
