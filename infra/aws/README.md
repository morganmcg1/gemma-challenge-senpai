# AWS Infra Handoff

Last verified: 2026-06-13

This repo currently manages one AWS EC2 A10G node for `gemma-chall` work. The launcher code is in `infra/aws/src/a10g_node/cli.py` and is run through `uv`.

## Auth

The currently working launcher path uses this local AWS SSO profile:

```text
sandbox-sso
```

Expected local files:

- `~/.aws/config` contains SSO profile definitions from the WandB AWS config,
  including `sandbox-sso`.
- `.env` points `AWS_PROFILE` at that profile.
- `.credentials` may exist as a local scratch copy, but it is ignored by git and should not be committed.

If AWS calls start failing with expired-token errors, refresh SSO with `aws sso
login --profile sandbox-sso`. Console-copied temporary credentials remain a
fallback, but SSO is preferred.

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

To refresh via SSO instead of console-copied temporary credentials:

```bash
aws sso login --profile sandbox-sso
```

Then switch `.env` to:

```dotenv
AWS_PROFILE=sandbox-sso
AWS_REGION=us-east-1
```

Do not switch `.env` to `sandbox-sso` until `aws sso login --profile sandbox-sso` succeeds.

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
Instance ID:     i-0554325ee1d640aaf
Name tag:        gemma-a10g-8gpu
Project tag:     gemma-chall
Instance type:   g5.48xlarge
GPU:             8 x NVIDIA A10G
Region:          us-east-1
AZ:              us-east-1c
AMI:             ami-097fdc6a0158e9c8b
State:           running
Public IP:       107.22.25.10
Public DNS:      ec2-107-22-25-10.compute-1.amazonaws.com
Private IP:      10.10.2.220
SSH user:        ubuntu
EC2 key pair:    gemma-a10g
Local key file:  /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem
```

Operational rule: try SSH before reaching for AWS APIs. SSH uses the local key
and can still work even when SSO or temporary AWS credentials are expired. Use
AWS only when SSH cannot connect, when the node address may have changed, or
when the SSH ingress rule needs to be updated.

SSH:

```bash
ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com
```

Verify GPU:

```bash
ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
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
Subnet:          subnet-07906ec4a58796a70
Subnet CIDR:     10.10.2.0/24, instance private IP 10.10.2.220
Subnet AZ:       us-east-1c
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

If SSH times out at TCP connect and the instance is running, check your current
public IP:

```bash
curl https://checkip.amazonaws.com
```

If it differs from `79.97.210.242`, add a new SSH ingress rule for the current `/32` or update `ALLOWED_SSH_CIDR` in `.env` before launching a replacement node.
If AWS credentials are expired, refresh with:

```bash
aws sso login --profile sandbox-sso
```

Then discover the current node address and retry SSH.

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
INSTANCE_ID=i-0554325ee1d640aaf
SSH_HOST=ec2-107-22-25-10.compute-1.amazonaws.com
SSH_PUBLIC_IP=107.22.25.10
SSH_PRIVATE_IP=10.10.2.220
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
- G5 instances are the AWS A10G family. `g5.xlarge` gives one A10G GPU; `g5.48xlarge` gives eight A10G GPUs.
- If this work moves to CoreWeave, verify and use the latest supported CoreWeave GPU image at launch time instead of reusing an old tag.
