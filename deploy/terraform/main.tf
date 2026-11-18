# Infrastructure as code for the AWS deployment (one EC2 VM running docker compose).
# Usage:  cd deploy/terraform && terraform init && terraform apply -var="key_name=<your-ec2-key-pair>" -var="repo_url=<your-github-url>"
# Destroy when finished:  terraform destroy

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "region"        { default = "eu-west-1" }
variable "instance_type" { default = "t3.large" } # 2 vCPU / 8 GB RAM: enough for TensorFlow + retraining
variable "key_name"      { description = "Existing EC2 key pair name for SSH" }
variable "repo_url"      { description = "Public GitHub URL of this project" }

provider "aws" { region = var.region }

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_security_group" "app" {
  name        = "pneumonia-app"
  description = "SSH, API (nginx) and UI"
  ingress { 
    from_port = 22   
    to_port = 22   
    protocol = "tcp" 
    cidr_blocks = ["0.0.0.0/0"] 
  }
  ingress { 
    from_port = 8080 
    to_port = 8080 
    protocol = "tcp" 
    cidr_blocks = ["0.0.0.0/0"] 
  }
  ingress { 
    from_port = 8501 
    to_port = 8501 
    protocol = "tcp" 
    cidr_blocks = ["0.0.0.0/0"] 
  }
  egress { 
    from_port = 0 
    to_port = 0 
    protocol = "-1" 
    cidr_blocks = ["0.0.0.0/0"] 
  }
}

resource "aws_instance" "app" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.app.id]

  root_block_device { volume_size = 40 } # images + dataset + models

  user_data = <<-EOT
    #!/bin/bash
    apt-get update -y
    apt-get install -y docker.io docker-compose-v2 git unzip python3-pip
    systemctl enable --now docker
    cd /home/ubuntu
    git clone ${var.repo_url} app
    chown -R ubuntu:ubuntu app
  EOT

  tags = { Name = "pneumonia-ml-pipeline" }
}

output "public_ip" { value = aws_instance.app.public_ip }
output "ui_url"    { value = "http://${aws_instance.app.public_ip}:8501" }
output "api_docs"  { value = "http://${aws_instance.app.public_ip}:8080/docs" }
