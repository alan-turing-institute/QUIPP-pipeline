"""A Pulumi script for deploying virtual machines on Azure.
Every VM gets it's own resource group and virtual network.
To configure update `stack_vms`. Currently configurable with
the fields of the `VM` data class.

Mounts a file share. Assumes a file share has already been
created in the same subscription

SSH
-----
Once deployed cloud init will configure the machine.
SSH with:

`ssh {user}@{fqdn}`

Once you have a shell run the following to wait until
configuration completes:

`cloud-init status -w`

Once complete check configuration was successful:

`cloud-init status --long`

To see logs:

`sudo cat /var/log/cloud-init-output.log`
"""

from typing import List, Optional
from dataclasses import dataclass, field
import subprocess
from pathlib import Path
import base64
import pulumi
import pulumi_azure as azure
from jinja2 import Environment, FileSystemLoader


def subprocess_command(cmd: str) -> str:
    """Pass a cmd and return stdout as a str.
    utility for getting data from the azure cli (az)

    Args:
        cmd (str): An az cli command
    """

    stdout = subprocess.run(
        cmd, shell=True, check=True, capture_output=True, text=True
    ).stdout

    if not stdout:
        raise IOError("Did not get stdout")
    return stdout.strip()


@dataclass
class User:
    """Class for defining a VM user passed to cloud init"""

    name: str
    public_key: str


@dataclass
class VM:
    "Class detailing VM configuration"

    name: str
    admin_user: User
    cloud_init: Optional[str]
    size: str = "Standard_A1_v2"
    location: str = "North Europe"
    domain_name_label: str = field(init=False)

    def __post_init__(self):
        self.domain_name_label = f"{self.name}".replace("_", "-")


@dataclass
class FileShare:
    resource_group: str
    storage_account: str
    fileshare_name: str
    http_endpoint: str

    storage_account_key: str = field(init=False)

    def __post_init__(self):
        cmd = f"""
        az storage account keys list \
    --resource-group {self.resource_group} \
    --account-name {self.storage_account} \
    --query "[0].value" | tr -d '"'
    """
        self.storage_account_key = subprocess_command(cmd)


def render_cloud_init_template(
    all_users: List[User],
    mount_fileshare: FileShare,
    repo_branch: str = "feature/152-add-noshows-example",
) -> str:
    """Render the cloud_init.yaml file

    Args:
        all_users (List[User]): User name and public ky
        mount_fileshare (FileShare): Config info for fileshare
        repo_branch (str, optional): Branch of QUiPP repo to checkout. Used to install dependencies. Defaults to "feature/152-add-noshows-example".

    Returns:
        str: [description]
    """

    templateLoader = FileSystemLoader(searchpath="./")
    env = Environment(loader=templateLoader)

    template = env.get_template("cloud_init_template.yaml")

    return template.render(
        all_users=all_users,
        repo_branch=repo_branch,
        fileshare_resource_group=mount_fileshare.resource_group,
        fileshare_storage_account_name=mount_fileshare.storage_account,
        fileshare_name=mount_fileshare.fileshare_name,
        storage_account_key=mount_fileshare.storage_account_key,
        http_endpoint=mount_fileshare.http_endpoint,
    )


def create_vm(vm_conf: VM):
    """Provision a vm based on vm_conf.
    Each VM has its own resource group and vnet

    Args:
        vm_conf (VM): A VM configuration
    """
    resource_group = azure.core.ResourceGroup(vm_conf.name, location=vm_conf.location)

    virtual_network = azure.network.VirtualNetwork(
        f"{vm_conf.name}_network",
        address_spaces=["10.0.0.0/16"],
        location=resource_group.location,
        resource_group_name=resource_group.name,
    )

    subnet = azure.network.Subnet(
        f"{vm_conf.name}_subnet",
        resource_group_name=resource_group.name,
        virtual_network_name=virtual_network.name,
        address_prefixes=["10.0.2.0/24"],
    )

    public_ip = azure.network.PublicIp(
        f"{vm_conf.name}-server-ip",
        resource_group_name=resource_group.name,
        location=resource_group.location,
        allocation_method="Dynamic",
        domain_name_label=vm_conf.domain_name_label,
    )

    network_interface = azure.network.NetworkInterface(
        f"{vm_conf.name}_nic",
        location=resource_group.location,
        resource_group_name=resource_group.name,
        ip_configurations=[
            azure.network.NetworkInterfaceIpConfigurationArgs(
                name="internal",
                subnet_id=subnet.id,
                private_ip_address_allocation="Dynamic",
                public_ip_address_id=public_ip.id,
            )
        ],
    )

    linux_virtual_machine = azure.compute.LinuxVirtualMachine(
        f"{vm_conf.name}_vm".replace("_", "-"),
        resource_group_name=resource_group.name,
        location=resource_group.location,
        size=vm_conf.size,
        admin_username=vm_conf.admin_user.name,
        network_interface_ids=[network_interface.id],
        admin_ssh_keys=[
            azure.compute.LinuxVirtualMachineAdminSshKeyArgs(
                username=vm_conf.admin_user.name,
                public_key=vm_conf.admin_user.public_key,
            ),
        ],
        os_disk=azure.compute.LinuxVirtualMachineOsDiskArgs(
            caching="ReadWrite", storage_account_type="Standard_LRS",
        ),
        source_image_reference=azure.compute.LinuxVirtualMachineSourceImageReferenceArgs(
            publisher="Canonical",
            offer="UbuntuServer",
            sku="18.04-LTS",
            version="latest",
        ),
        custom_data=base64.b64encode(vm_conf.cloud_init.encode()).decode("ascii"),
    )

    pulumi.export("dns", public_ip.fqdn)


if __name__ == "__main__":

    # VM Configurations

    # Require an admin user
    admin_user = User("ogiles", public_key="",)

    # All users including admin
    all_users = [
        admin_user,
        User("ostrickson", public_key="",),
    ]

    # Render cloud init template
    cloud_init = render_cloud_init_template(
        all_users,
        FileShare(
            resource_group="FileStorage",
            storage_account="vmexperiments",
            fileshare_name="quippstore",
            http_endpoint="https://vmexperiments.file.core.windows.net/",
        ),
    )

    # List of all VM configurations
    stack_vms = [
        VM(
            "small-a",
            size="Standard_B1s",
            location="UK West",
            admin_user=admin_user,
            cloud_init=cloud_init,
        ),
    ]

    # Create all VMs
    for i in stack_vms:
        create_vm(i)
