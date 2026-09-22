# ChurnIQ AWS Deployment

This document describes the AWS layer that is actually implemented by the project.

## Architecture

```text
Streamlit
    |
    | HTTPS/HTTP
    v
AWS EC2
    |
  Nginx
    |
Gunicorn
    |
Flask REST API
    |
ChurnIQ model (.joblib)

AWS S3
    |
    +-- raw dataset
    +-- trained model
    +-- metadata
```

EC2 hosts the Flask inference API. S3 stores ChurnIQ artifacts.

## 1. AWS prerequisites

Create:

- an AWS account
- an S3 bucket
- an EC2 Ubuntu instance

Recommended EC2 security-group rules:

- TCP 22: your IP only
- TCP 80: `0.0.0.0/0`
- TCP 443: `0.0.0.0/0` if HTTPS is later configured

Do not expose port 8000 publicly when nginx is used as the reverse proxy.

## 2. S3 configuration

Set:

```env
AWS_REGION=ap-south-1
AWS_S3_BUCKET=your-unique-churniq-bucket
AWS_S3_PREFIX=churniq
```

Do not put AWS access keys in `.env` on EC2 if an IAM role can be attached to the instance.

The S3 utility is:

```text
src/cloud/s3_manager.py
```

Upload the model/data artifacts with:

```bash
python -m src.cloud.s3_sync
```

## 3. EC2 deployment

SSH into the instance:

```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

Clone the repository:

```bash
sudo mkdir -p /opt
sudo chown ubuntu:ubuntu /opt
cd /opt
git clone https://github.com/vinayygoudd/ChurnIQ.git
cd ChurnIQ
```

Run:

```bash
chmod +x deploy/aws/setup_ec2.sh
./deploy/aws/setup_ec2.sh
```

The script installs Python, nginx, creates the virtual environment, installs dependencies, and registers the Gunicorn systemd service.

## 4. Environment

Create:

```bash
nano /opt/ChurnIQ/.env
```

Example:

```env
FLASK_ENV=production
AWS_REGION=ap-south-1
AWS_S3_BUCKET=your-unique-churniq-bucket
AWS_S3_PREFIX=churniq
```

Do not commit this file.

## 5. Verify the API

From your local machine:

```bash
curl http://YOUR_EC2_PUBLIC_IP/health
```

Expected shape:

```json
{
  "status": "ok",
  "model_version": "1.0.0"
}
```

Then test:

```bash
curl http://YOUR_EC2_PUBLIC_IP/model-info
```

## 6. Service management

Check:

```bash
sudo systemctl status churniq
```

Restart:

```bash
sudo systemctl restart churniq
```

View logs:

```bash
sudo journalctl -u churniq -f
```

Nginx logs:

```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

## 7. Streamlit configuration

Set the Streamlit deployment secret/environment variable:

```env
CHURN_API_URL=http://YOUR_EC2_PUBLIC_IP
```

The Streamlit API client in:

```text
src/api/client.py
```

provides:

- `health()`
- `predict(payload)`
- `explain(payload)`
- `model_info()`
- `analytics()`

The Streamlit application should call these functions in production so the browser-facing UI communicates with the AWS-hosted Flask service.

## 8. HTTPS

For a portfolio/demo deployment, HTTP can be used initially.

For a production-style deployment, use a domain name and configure HTTPS with an appropriate AWS/networking setup before exposing customer data.

## 9. Security

- Never commit `.env`.
- Never hard-code AWS credentials.
- Prefer an EC2 IAM role with only the S3 permissions required.
- Restrict SSH access to your IP.
- Do not expose port 8000 publicly.
- Do not log raw customer payloads.
- Keep S3 buckets private unless public access is explicitly required.
