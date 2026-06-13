# Fast Gemma Challenge Agent Notes

## AWS A10G Access

- Before using AWS APIs, try direct SSH to the recorded A10G node. SSH only
  needs the local key and can still work when AWS SSO or temporary credentials
  are expired:

  ```bash
  ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
    ubuntu@ec2-3-87-184-234.compute-1.amazonaws.com
  ```

- If SSH times out at TCP connect, the public IP/DNS may be stale or the
  security group may only allow a previous operator IP. Then refresh AWS access
  with `aws sso login --profile sandbox-sso` or renewed temporary credentials,
  discover the current instance address, and update SSH ingress for the current
  `/32`.
- Do not print `.env`, `.credentials`, or Kubernetes Secret values in logs or
  final answers.
