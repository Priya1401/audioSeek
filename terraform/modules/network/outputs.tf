output "network_name" {
  value = google_compute_network.vpc.name
}

output "subnetwork_name" {
  value = google_compute_subnetwork.subnet.name
}

output "network_self_link" {
  value = google_compute_network.vpc.self_link
}

output "subnetwork_self_link" {
  value = google_compute_subnetwork.subnet.self_link
}
