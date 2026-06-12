# AWS Infra Handoff

Last verified: 2026-06-12

This repo currently manages one AWS EC2 A10G node for `gemma-chall` work. The launcher code is in `src/a10g_node/cli.py` and is run through `uv`.

## Auth

Use this local AWS profile:

```text
770934259321_SandboxPowerUsers
```

Expected local files:

- `~/.aws/credentials` contains the profile above.
- `.env` points `AWS_PROFILE` at that profile.
- `.credentials` may exist as a local scratch copy, but it is ignored by git and should not be committed.

The current credentials are temporary SSO/session credentials. If AWS calls start failing with expired-token errors, refresh them from the AWS console's "Command line or programmatic access" flow and update `~/.aws/credentials`.

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

```text
Instance ID:     i-031a9640edf2921c5
Name tag:        gemma-a10g
Project tag:     gemma-chall
Instance type:   g5.xlarge
GPU:             NVIDIA A10G
Region:          us-east-1
AZ:              us-east-1a
AMI:             ami-097fdc6a0158e9c8b
State:           running
Public IP:       3.87.184.234
Public DNS:      ec2-3-87-184-234.compute-1.amazonaws.com
Private IP:      10.10.0.157
SSH user:        ubuntu
EC2 key pair:    gemma-a10g
Local key file:  /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem
```

SSH:

```bash
ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem ubuntu@ec2-3-87-184-234.compute-1.amazonaws.com
```

Verify GPU:

```bash
ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem ubuntu@ec2-3-87-184-234.compute-1.amazonaws.com \
  'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'
```

Last verified GPU output:

```text
NVIDIA A10G, 23028 MiB, 580.159.04
```

## Network Resources

The account did not have a default VPC in `us-east-1`, so the launcher is pinned to an existing subnet in `.env`.

```text
VPC:             vpc-039a5ded9effa16d4
Subnet:          subnet-0cf052db6f6880fe3
Subnet CIDR:     10.10.0.0/16 VPC, instance private IP 10.10.0.157
Subnet AZ:       us-east-1a
Public IPs:      MapPublicIpOnLaunch=true
Route table:     rtb-0ba8e9d1ad219b340
Internet GW:     igw-0120229c8791d3e44
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

Terminate the current node:

```bash
cd /Users/mmcguire/ML/gemma_chall
uv run a10g terminate i-031a9640edf2921c5
```

## Local Config

The relevant `.env` values are:

```dotenv
AWS_PROFILE=770934259321_SandboxPowerUsers
AWS_REGION=us-east-1
INSTANCE_NAME=gemma-a10g
INSTANCE_TYPE=g5.xlarge
VOLUME_GB=200
SUBNET_ID=subnet-0cf052db6f6880fe3
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

- The node is on-demand, so it accrues cost while `running`.
- Public IP and DNS can change if the instance is stopped and started; the current launcher terminates, it does not stop.
- The AWS key pair private key exists only locally at `gemma-a10g.pem`; preserve it if this exact instance must remain accessible.
- G5 instances are the AWS A10G family. `g5.xlarge` gives one A10G GPU.
- If this work moves to CoreWeave, verify and use the latest supported CoreWeave GPU image at launch time instead of reusing an old tag.
