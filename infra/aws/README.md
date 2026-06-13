# AWS Infra Handoff

Last verified: 2026-06-12

This repo currently manages one AWS EC2 A10G node for `gemma-chall` work. The launcher code is in `infra/aws/src/a10g_node/cli.py` and is run through `uv`.

## Auth

The currently working launcher path uses this local AWS SSO profile:

```text
sandbox-sso
```

Expected local files:

- `~/.aws/config` contains SSO profile definitions from the WandB AWS config.
- `.env` points `AWS_PROFILE` at that profile.
- `~/.aws/credentials` may contain older console-copied temporary credentials under `770934259321_SandboxPowerUsers`.
- `.credentials` may exist as a local scratch copy, but it is ignored by git and should not be committed.

If AWS calls start failing with missing or expired SSO token errors, refresh with `aws sso login --profile sandbox-sso`.

SSO config note: the WandB SSO start URL is `https://wandb.awsapps.com/start`, but the SSO region is `us-east-2`. Using `us-east-1` for the SSO region causes `Invalid start url provided`.

These non-secret SSO profiles are configured in `~/.aws/config`:

```text
deployments
sre
AWSPowerUserAccess-830241207209
sandbox-sso
SandboxPowerUsers-770934259321
```

For this sandbox account, the useful SSO profile is:

```text
sandbox-sso
```

To refresh via SSO:

```bash
aws sso login --profile sandbox-sso
```

`.env` should contain:

```dotenv
AWS_PROFILE=sandbox-sso
AWS_REGION=us-east-1
```

Do not switch away from `sandbox-sso` unless SSO is unavailable and you intentionally fall back to console-copied temporary credentials.

Check auth:

```bash
cd /Users/mmcguire/ML/gemma_chall
uv run a10g check-auth
```

Expected account:

```text
770934259321
```

## Current Running Node

Total live GPU count: **8 NVIDIA A10G GPUs**

Primary node:

```text
Instance ID:     i-0554325ee1d640aaf
Name tag:        gemma-a10g-8gpu
Project tag:     gemma-chall
Instance type:   g5.48xlarge
GPU count:       8 NVIDIA A10G
Region:          us-east-1
AZ:              us-east-1c
AMI:             ami-097fdc6a0158e9c8b
State:           running
Public IP:       107.22.25.10
Public DNS:      ec2-107-22-25-10.compute-1.amazonaws.com
Private IP:      10.10.2.220
Subnet:          subnet-07906ec4a58796a70
SSH user:        ubuntu
EC2 key pair:    gemma-a10g
Local key file:  /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem
Root volume:     1000 GB gp3
```

SSH to the 8-GPU node:

```bash
ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com
```

Verify all 8 GPUs:

```bash
ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
  'nvidia-smi -L'
```

Last verified 8-GPU output:

```text
GPU 0: NVIDIA A10G
GPU 1: NVIDIA A10G
GPU 2: NVIDIA A10G
GPU 3: NVIDIA A10G
GPU 4: NVIDIA A10G
GPU 5: NVIDIA A10G
GPU 6: NVIDIA A10G
GPU 7: NVIDIA A10G
```

Terminated old 1-GPU node: `i-031a9640edf2921c5` (`g5.xlarge`, formerly `gemma-a10g` in `us-east-1a`). Last observed state: `terminated`, contributing 0 running GPUs.

## Network Resources

The account did not have a default VPC in `us-east-1`, so the launcher is pinned to an existing subnet in `.env`.

```text
VPC:             vpc-039a5ded9effa16d4
VPC CIDR:        10.10.0.0/16
Public IPs:      MapPublicIpOnLaunch=true
Route table:     rtb-0ba8e9d1ad219b340
Internet GW:     igw-0120229c8791d3e44
```

Public subnets created/used for A10G capacity:

```text
subnet-0cf052db6f6880fe3  us-east-1a  10.10.0.0/24  former 1-GPU node subnet
subnet-068a3a8502b5e3445  us-east-1b  10.10.1.0/24  created while searching for g5.48xlarge capacity
subnet-07906ec4a58796a70  us-east-1c  10.10.2.0/24  current 8-GPU node
subnet-025c0b1feebf48786  us-east-1d  10.10.3.0/24  created while searching for g5.48xlarge capacity
subnet-0910f1e659af36b74  us-east-1f  10.10.4.0/24  created while searching for g5.48xlarge capacity
```

Important history: the subnet's default route originally pointed at a blackholed internet gateway. We created `igw-0120229c8791d3e44`, attached it to `vpc-039a5ded9effa16d4`, and replaced the route table default route:

```text
0.0.0.0/0 -> igw-0120229c8791d3e44
```

Do not "clean up" that internet gateway or route while this node needs SSH access.

## Security Group

```text
Security group:  sg-04180a85dbbea2a71
Name:            gemma-a10g-ssh
Ingress:         TCP 22 from 79.97.210.242/32
```

If SSH times out and the instance is running, check your current public IP:

```bash
curl https://checkip.amazonaws.com
```

If it differs from `79.97.210.242`, add a new SSH ingress rule for the current `/32` or update `ALLOWED_SSH_CIDR` in `.env` before launching a replacement node.

## Launcher Commands

Preview the launch plan:

```bash
cd /Users/mmcguire/ML/gemma_chall
uv run a10g plan
```

Launch a replacement node:

```bash
cd /Users/mmcguire/ML/gemma_chall
uv run a10g launch
```

Launch an 8-GPU replacement node:

```bash
cd /Users/mmcguire/ML/gemma_chall
AWS_PROFILE=sandbox-sso \
INSTANCE_NAME=gemma-a10g-8gpu \
INSTANCE_TYPE=g5.48xlarge \
VOLUME_GB=1000 \
KEY_NAME=gemma-a10g \
KEY_PATH=./gemma-a10g.pem \
SUBNET_ID=subnet-07906ec4a58796a70 \
SECURITY_GROUP_NAME=gemma-a10g-ssh \
AWS_REGION=us-east-1 \
uv run a10g launch
```

Terminate the 8-GPU node:

```bash
cd /Users/mmcguire/ML/gemma_chall
uv run a10g terminate i-0554325ee1d640aaf
```

## Local Config

The relevant `.env` values are:

```dotenv
AWS_PROFILE=sandbox-sso
AWS_REGION=us-east-1
INSTANCE_NAME=gemma-a10g-8gpu
INSTANCE_TYPE=g5.48xlarge
VOLUME_GB=1000
SUBNET_ID=subnet-07906ec4a58796a70
SECURITY_GROUP_NAME=gemma-a10g-ssh
KEY_NAME=gemma-a10g
KEY_PATH=./gemma-a10g.pem
SSH_USER=ubuntu
MARKET_TYPE=on-demand
```

The AMI is resolved from:

```text
/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.8-ubuntu-24.04/latest/ami-id
```

For the current region, that resolved to:

```text
ami-097fdc6a0158e9c8b
```

## Operational Notes

- The node is on-demand, so it accrues cost while `running`. The `g5.48xlarge` is especially expensive.
- Public IP and DNS can change if the instance is stopped and started; the current launcher terminates, it does not stop.
- The AWS key pair private key exists only locally at `gemma-a10g.pem`; preserve it if this exact instance must remain accessible.
- G5 instances are the AWS A10G family. `g5.xlarge` gives one A10G GPU; `g5.48xlarge` gives eight A10G GPUs.
- If this work moves to CoreWeave, verify and use the latest supported CoreWeave GPU image at launch time instead of reusing an old tag.
