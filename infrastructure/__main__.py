"""A Python Pulumi program"""

from dataclasses import dataclass, field
from pathlib import Path
import base64
import pulumi
import pulumi_azure as azure

@dataclass
class VM:
    """Class detailing VM configuration
    """
    name: str
    size: str = "Standard_A1_v2"
    location: str = "North Europe"
    admin_username: str = "adminuser"
    domain_name_label: str = field(init=False)

    def __post_init__(self):
        # Ensure domain name is unique
        self.domain_name_label = f"{self.name}".replace("_", "-")


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
        domain_name_label=vm_conf.domain_name_label
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
        admin_username=vm_conf.admin_username,
        network_interface_ids=[network_interface.id],
        admin_ssh_keys=[
            azure.compute.LinuxVirtualMachineAdminSshKeyArgs(
                username=vm_conf.admin_username,
                public_key=(lambda path: path.open().read())(SSH_admin_key),
            ),
        ],
        os_disk=azure.compute.LinuxVirtualMachineOsDiskArgs(
            caching="ReadWrite", storage_account_type="Standard_LRS",
        ),
        source_image_reference=azure.compute.LinuxVirtualMachineSourceImageReferenceArgs(
            publisher="Canonical",
            offer="UbuntuServer",
            sku="16.04-LTS",
            version="latest",
        ),
        custom_data=(
            lambda path: base64.b64encode(cloud_init.open("rb").read()).decode("ascii")
        )(cloud_init),
    )


# VM Configurations
stack_vms = [VM("large-b", size = "Standard_A2_v2")]
SSH_admin_key = Path("~/.ssh/id_rsa.pub").expanduser()
cloud_init = Path("vm_config.yaml")

# Create all VMs
for i in stack_vms:
    create_vm(i)