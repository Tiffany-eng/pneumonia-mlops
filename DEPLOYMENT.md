# Cloud deployment

**Choice: one cloud VM running `docker compose`** (AWS EC2, Azure VM or GCP Compute Engine).
Why: retraining writes new model files and uploads to disk and needs several GB of RAM. Free serverless / free web tiers often have no persistent disk or enough memory, so they cannot retrain. A VM also lets us scale the API containers with `--scale api=N` for the flood test.

| | Needs |
|---|---|
| VM | 2 vCPU, 8 GB RAM (e.g. AWS t3.large), 40 GB disk, Ubuntu 22.04 |
| Open ports | 22 (SSH), 8080 (API through nginx), 8501 (UI) |

## Option A: AWS with Terraform (infrastructure as code)
```bash
cd deploy/terraform
terraform init
terraform apply -var="key_name=<your-ec2-key-pair>" -var="repo_url=<your-github-url>"
```
Terraform creates the security group and the EC2 instance, installs Docker and clones the repo. It prints the public IP and the URLs. Run `terraform destroy` when you are done to stop charges.

## Option B: Azure or GCP (manual, same steps)
1. Create an Ubuntu 22.04 VM (Azure: student credits, GCP: free-trial credit). Open ports 22, 8080, 8501.
2. SSH in and install Docker:
   ```bash
   sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git unzip python3-pip
   sudo systemctl enable --now docker
   git clone <your-github-url> app
   ```

## Steps on the VM (both options)
```bash
ssh ubuntu@<public-ip>
cd app
# 1. dataset (needed for the insights page, training and retraining)
pip3 install kagglehub pillow
mkdir -p ~/.kaggle && nano ~/.kaggle/kaggle.json       # paste your Kaggle token
python3 -c "from src.utils.data_acquisition import download_dataset; download_dataset()"   # no TensorFlow needed here

# 2. the trained model models/pneumonia_model.h5 must be in the repo (git add -f) or copied with scp

# 3. start everything (change 1 to 2 or 4 to scale the API)
sudo docker compose up --build -d --scale api=1
```
- UI: `http://<public-ip>:8501`
- API docs: `http://<public-ip>:8080/docs`

If you replace the model file later: `sudo docker compose down -v && sudo docker compose up --build -d`.

## Evaluating the model in production
- The **Monitoring** page shows uptime, request counts, average latency and the test-set metrics of the model currently deployed.
- After a retrain, the metrics and history table update, and the new model is only promoted if recall does not drop.
- Run Locust against `http://<public-ip>:8080` (see the README).

## Keep it running
Keep the VM up until the work has been graded, then run `terraform destroy` (or delete the VM).
