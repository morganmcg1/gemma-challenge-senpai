from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound
from dotenv import dotenv_values


DEFAULT_ENV_PATH = Path(".env")
DEFAULT_REGION = "us-east-1"
DEFAULT_DLAMI_PARAMETER = (
    "/aws/service/deeplearning/ami/x86_64/"
    "oss-nvidia-driver-gpu-pytorch-2.8-ubuntu-24.04/latest/ami-id"
)
GPU_INSTANCE_TYPES = {
    "g5.xlarge",
    "g5.2xlarge",
    "g5.4xlarge",
    "g5.8xlarge",
    "g5.12xlarge",
    "g5.16xlarge",
    "g5.24xlarge",
    "g5.48xlarge",
}


@dataclass(frozen=True)
class Config:
    region: str
    profile: str | None
    instance_name: str
    instance_type: str
    volume_gb: int
    ami_id: str | None
    dlami_ssm_parameter: str
    key_name: str
    key_path: Path
    ssh_user: str
    subnet_id: str | None
    security_group_id: str | None
    security_group_name: str
    allowed_ssh_cidr: str | None
    iam_instance_profile_name: str | None
    user_data_path: Path | None
    market_type: str

    @classmethod
    def from_env(cls, env_path: Path) -> "Config":
        load_env(env_path)

        return cls(
            region=value("AWS_REGION") or value("AWS_DEFAULT_REGION") or DEFAULT_REGION,
            profile=value("AWS_PROFILE"),
            instance_name=value("INSTANCE_NAME") or "gemma-a10g",
            instance_type=value("INSTANCE_TYPE") or "g5.xlarge",
            volume_gb=int(value("VOLUME_GB") or "200"),
            ami_id=value("AMI_ID"),
            dlami_ssm_parameter=value("DLAMI_SSM_PARAMETER") or DEFAULT_DLAMI_PARAMETER,
            key_name=value("KEY_NAME") or "gemma-a10g",
            key_path=Path(value("KEY_PATH") or "./gemma-a10g.pem").expanduser(),
            ssh_user=value("SSH_USER") or "ubuntu",
            subnet_id=value("SUBNET_ID"),
            security_group_id=value("SECURITY_GROUP_ID"),
            security_group_name=value("SECURITY_GROUP_NAME") or "gemma-a10g-ssh",
            allowed_ssh_cidr=value("ALLOWED_SSH_CIDR"),
            iam_instance_profile_name=value("IAM_INSTANCE_PROFILE_NAME"),
            user_data_path=optional_path(value("USER_DATA_PATH")),
            market_type=(value("MARKET_TYPE") or "on-demand").lower(),
        )

    def validate(self) -> None:
        if self.instance_type not in GPU_INSTANCE_TYPES:
            fail(
                f"{self.instance_type!r} is not a known G5 A10G instance type. "
                f"Use one of: {', '.join(sorted(GPU_INSTANCE_TYPES))}"
            )
        if self.volume_gb < 30:
            fail("VOLUME_GB should be at least 30 for a Deep Learning AMI.")
        if self.market_type not in {"on-demand", "spot"}:
            fail("MARKET_TYPE must be either 'on-demand' or 'spot'.")
        if self.user_data_path and not self.user_data_path.exists():
            fail(f"USER_DATA_PATH does not exist: {self.user_data_path}")


@dataclass(frozen=True)
class LaunchPlan:
    account: dict[str, Any]
    ami_id: str
    subnet_id: str
    availability_zone: str
    key_name: str
    security_group_id: str
    security_group_created: bool
    ssh_cidr: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Spin up an AWS EC2 G5/A10G node.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to .env")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-auth", help="Verify AWS credentials and identity.")
    subparsers.add_parser("plan", help="Resolve AMI, subnet, SSH key, and security group.")

    launch = subparsers.add_parser("launch", help="Create missing resources and start the node.")
    launch.add_argument(
        "--no-wait",
        action="store_true",
        help="Return after run-instances instead of waiting for running status.",
    )

    terminate = subparsers.add_parser("terminate", help="Terminate an instance by id.")
    terminate.add_argument("instance_id")

    args = parser.parse_args()

    config = Config.from_env(Path(args.env_file))
    config.validate()

    try:
        session = make_session(config)
        ec2 = session.client("ec2")
        ssm = session.client("ssm")
        sts = session.client("sts")

        if args.command == "check-auth":
            print_json({"identity": get_identity(sts), "region": config.region})
        elif args.command == "plan":
            print_plan(resolve_plan(config, ec2, ssm, sts, mutate=False))
        elif args.command == "launch":
            plan = resolve_plan(config, ec2, ssm, sts, mutate=True)
            instance = launch_instance(config, ec2, plan)
            if not args.no_wait:
                wait_for_running(ec2, instance["InstanceId"])
                instance = describe_instance(ec2, instance["InstanceId"])
            print_launch(config, plan, instance)
        elif args.command == "terminate":
            ec2.terminate_instances(InstanceIds=[args.instance_id])
            print(f"Terminating {args.instance_id}")
    except ProfileNotFound as exc:
        fail(f"AWS profile not found: {exc}. Update AWS_PROFILE in .env or use access keys.")
    except NoCredentialsError:
        fail(
            "AWS credentials are not configured. Fill .env with AWS_PROFILE or "
            "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, then run `uv run a10g check-auth`."
        )
    except ClientError as exc:
        explain_client_error(exc)


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    for key, val in dotenv_values(env_path).items():
        if key and val:
            os.environ.setdefault(key, val)


def value(name: str) -> str | None:
    val = os.environ.get(name)
    return val.strip() if val and val.strip() else None


def optional_path(raw: str | None) -> Path | None:
    return Path(raw).expanduser() if raw else None


def make_session(config: Config) -> boto3.Session:
    if config.profile:
        return boto3.Session(profile_name=config.profile, region_name=config.region)
    return boto3.Session(region_name=config.region)


def get_identity(sts: Any) -> dict[str, Any]:
    identity = sts.get_caller_identity()
    return {
        "account": identity["Account"],
        "arn": identity["Arn"],
        "user_id": identity["UserId"],
    }


def resolve_plan(config: Config, ec2: Any, ssm: Any, sts: Any, *, mutate: bool) -> LaunchPlan:
    account = get_identity(sts)
    ami_id = config.ami_id or resolve_ami_id(ssm, config)
    subnet_id, availability_zone = resolve_subnet(ec2, config)
    ensure_instance_type_offered(ec2, config.instance_type, availability_zone)
    ensure_key_pair(ec2, config, mutate=mutate)
    ssh_cidr = config.allowed_ssh_cidr or current_public_ip_cidr()
    security_group_id, security_group_created = resolve_security_group(
        ec2, config, subnet_id, ssh_cidr, mutate=mutate
    )

    return LaunchPlan(
        account=account,
        ami_id=ami_id,
        subnet_id=subnet_id,
        availability_zone=availability_zone,
        key_name=config.key_name,
        security_group_id=security_group_id,
        security_group_created=security_group_created,
        ssh_cidr=ssh_cidr,
    )


def resolve_ami_id(ssm: Any, config: Config) -> str:
    response = ssm.get_parameter(Name=config.dlami_ssm_parameter)
    return response["Parameter"]["Value"]


def resolve_subnet(ec2: Any, config: Config) -> tuple[str, str]:
    if config.subnet_id:
        subnet = ec2.describe_subnets(SubnetIds=[config.subnet_id])["Subnets"][0]
        return subnet["SubnetId"], subnet["AvailabilityZone"]

    zones = offered_zones(ec2, config.instance_type)
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        fail("No default VPC found. Set SUBNET_ID in .env.")

    subnets = ec2.describe_subnets(
        Filters=[
            {"Name": "vpc-id", "Values": [vpcs[0]["VpcId"]]},
            {"Name": "default-for-az", "Values": ["true"]},
        ]
    )["Subnets"]
    offered = [subnet for subnet in subnets if subnet["AvailabilityZone"] in zones]
    if not offered:
        fail(f"No default subnet found in a zone that offers {config.instance_type}. Set SUBNET_ID.")

    subnet = sorted(offered, key=lambda item: item["AvailabilityZone"])[0]
    return subnet["SubnetId"], subnet["AvailabilityZone"]


def offered_zones(ec2: Any, instance_type: str) -> set[str]:
    paginator = ec2.get_paginator("describe_instance_type_offerings")
    zones: set[str] = set()
    for page in paginator.paginate(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [instance_type]}],
    ):
        zones.update(item["Location"] for item in page["InstanceTypeOfferings"])
    if not zones:
        fail(f"{instance_type} is not offered in this region.")
    return zones


def ensure_instance_type_offered(ec2: Any, instance_type: str, availability_zone: str) -> None:
    response = ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[
            {"Name": "instance-type", "Values": [instance_type]},
            {"Name": "location", "Values": [availability_zone]},
        ],
    )
    if not response["InstanceTypeOfferings"]:
        fail(f"{instance_type} is not offered in {availability_zone}. Set a different SUBNET_ID.")


def ensure_key_pair(ec2: Any, config: Config, *, mutate: bool) -> None:
    try:
        ec2.describe_key_pairs(KeyNames=[config.key_name])
        return
    except ClientError as exc:
        if error_code(exc) != "InvalidKeyPair.NotFound":
            raise

    if not mutate:
        return

    config.key_path.parent.mkdir(parents=True, exist_ok=True)
    if config.key_path.exists():
        fail(f"EC2 key pair {config.key_name!r} is missing, but KEY_PATH already exists: {config.key_path}")

    response = ec2.create_key_pair(KeyName=config.key_name, KeyType="rsa", KeyFormat="pem")
    config.key_path.write_text(response["KeyMaterial"], encoding="utf-8")
    config.key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def resolve_security_group(
    ec2: Any,
    config: Config,
    subnet_id: str,
    ssh_cidr: str,
    *,
    mutate: bool,
) -> tuple[str, bool]:
    if config.security_group_id:
        return config.security_group_id, False

    subnet = ec2.describe_subnets(SubnetIds=[subnet_id])["Subnets"][0]
    vpc_id = subnet["VpcId"]
    groups = ec2.describe_security_groups(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "group-name", "Values": [config.security_group_name]},
        ]
    )["SecurityGroups"]
    if groups:
        group_id = groups[0]["GroupId"]
        if mutate:
            ensure_ssh_ingress(ec2, group_id, ssh_cidr)
        return group_id, False

    if not mutate:
        return f"<will create {config.security_group_name}>", True

    response = ec2.create_security_group(
        GroupName=config.security_group_name,
        Description=f"SSH access for {config.instance_name}",
        VpcId=vpc_id,
        TagSpecifications=[
            {
                "ResourceType": "security-group",
                "Tags": tags(config.instance_name, extra={"ManagedBy": "a10g-node-cli"}),
            }
        ],
    )
    group_id = response["GroupId"]
    ensure_ssh_ingress(ec2, group_id, ssh_cidr)
    return group_id, True


def ensure_ssh_ingress(ec2: Any, group_id: str, ssh_cidr: str) -> None:
    try:
        ec2.authorize_security_group_ingress(
            GroupId=group_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": ssh_cidr, "Description": "SSH from operator IP"}],
                }
            ],
        )
    except ClientError as exc:
        if error_code(exc) != "InvalidPermission.Duplicate":
            raise


def current_public_ip_cidr() -> str:
    with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10) as response:
        ip = response.read().decode("utf-8").strip()
    return f"{ip}/32"


def launch_instance(config: Config, ec2: Any, plan: LaunchPlan) -> dict[str, Any]:
    params: dict[str, Any] = {
        "ImageId": plan.ami_id,
        "InstanceType": config.instance_type,
        "KeyName": plan.key_name,
        "SubnetId": plan.subnet_id,
        "SecurityGroupIds": [plan.security_group_id],
        "MinCount": 1,
        "MaxCount": 1,
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": config.volume_gb,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": tags(config.instance_name)},
            {"ResourceType": "volume", "Tags": tags(config.instance_name)},
        ],
    }

    if config.user_data_path:
        params["UserData"] = config.user_data_path.read_text(encoding="utf-8")
    if config.iam_instance_profile_name:
        params["IamInstanceProfile"] = {"Name": config.iam_instance_profile_name}
    if config.market_type == "spot":
        params["InstanceMarketOptions"] = {"MarketType": "spot"}

    response = ec2.run_instances(**params)
    return response["Instances"][0]


def tags(name: str, *, extra: dict[str, str] | None = None) -> list[dict[str, str]]:
    values = {"Name": name, "Project": "gemma-chall"}
    if extra:
        values.update(extra)
    return [{"Key": key, "Value": val} for key, val in values.items()]


def wait_for_running(ec2: Any, instance_id: str) -> None:
    print(f"Waiting for {instance_id} to enter running state...", file=sys.stderr)
    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
    time.sleep(5)


def describe_instance(ec2: Any, instance_id: str) -> dict[str, Any]:
    response = ec2.describe_instances(InstanceIds=[instance_id])
    return response["Reservations"][0]["Instances"][0]


def print_plan(plan: LaunchPlan) -> None:
    print_json(
        {
            "account": plan.account,
            "ami_id": plan.ami_id,
            "subnet_id": plan.subnet_id,
            "availability_zone": plan.availability_zone,
            "key_name": plan.key_name,
            "security_group_id": plan.security_group_id,
            "security_group_created": plan.security_group_created,
            "ssh_cidr": plan.ssh_cidr,
        }
    )


def print_launch(config: Config, plan: LaunchPlan, instance: dict[str, Any]) -> None:
    public_dns = instance.get("PublicDnsName")
    public_ip = instance.get("PublicIpAddress")
    print_json(
        {
            "instance_id": instance["InstanceId"],
            "state": instance["State"]["Name"],
            "instance_type": config.instance_type,
            "availability_zone": plan.availability_zone,
            "public_ip": public_ip,
            "public_dns": public_dns,
            "ssh": ssh_command(config, public_dns or public_ip),
            "ami_id": plan.ami_id,
            "security_group_id": plan.security_group_id,
            "key_path": str(config.key_path),
        }
    )


def ssh_command(config: Config, host: str | None) -> str | None:
    if not host:
        return None
    return f"ssh -i {config.key_path} {config.ssh_user}@{host}"


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def explain_client_error(exc: ClientError) -> None:
    code = error_code(exc)
    message = exc.response.get("Error", {}).get("Message", str(exc))
    hints = {
        "AuthFailure": "Check your AWS credentials in .env.",
        "UnauthorizedOperation": "Your AWS principal lacks one of the required EC2/SSM permissions.",
        "AccessDenied": "Your AWS principal lacks one of the required AWS permissions.",
        "OptInRequired": "Your account may need access enabled for this region or AMI.",
        "VcpuLimitExceeded": "Request a G/VT GPU vCPU quota increase or choose a smaller G5 type.",
        "InsufficientInstanceCapacity": "Try another region/AZ or retry later.",
        "InvalidAMIID.NotFound": "Set AMI_ID or choose a DLAMI_SSM_PARAMETER available in your region.",
        "InvalidSubnetID.NotFound": "Set a valid SUBNET_ID in .env.",
        "InvalidGroup.NotFound": "Set a valid SECURITY_GROUP_ID or let the launcher create one.",
    }
    hint = hints.get(code)
    suffix = f"\nHint: {hint}" if hint else ""
    fail(f"AWS error {code}: {message}{suffix}")


def error_code(exc: ClientError) -> str:
    return exc.response.get("Error", {}).get("Code", "Unknown")


def fail(message: str) -> None:
    raise SystemExit(message)


if __name__ == "__main__":
    main()
