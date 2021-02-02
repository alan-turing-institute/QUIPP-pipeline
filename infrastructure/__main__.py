"""A Python Pulumi program"""

from pathlib import Path
import toml
import base64
import pulumi
import pulumi_azure as azure


SSH_admin_key = Path("~/.ssh/id_rsa.pub").expanduser()
cloud_init = Path("vm_config.yaml")
n_vms = 1
name_prefix = "testquippexp"
vm_size = "Standard_D1_v2"

for vm in range(n_vms):
    

    vm_name = name_prefix + f"_{vm}"

    print(f"Provisioning {vm_name}")

    example_resource_group = azure.core.ResourceGroup(vm_name, location="North Europe")

    example_virtual_network = azure.network.VirtualNetwork(
        f"{vm_name}_network",
        address_spaces=["10.0.0.0/16"],
        location=example_resource_group.location,
        resource_group_name=example_resource_group.name,
    )

    example_subnet = azure.network.Subnet(
        f"{vm_name}_subnet",
        resource_group_name=example_resource_group.name,
        virtual_network_name=example_virtual_network.name,
        address_prefixes=["10.0.2.0/24"],
    )

    public_ip = azure.network.PublicIp(
        f"{vm_name}-server-ip",
        resource_group_name=example_resource_group.name,
        location=example_resource_group.location,
        allocation_method="Dynamic",
    )

    example_network_interface = azure.network.NetworkInterface(
        f"{vm_name}_nic",
        location=example_resource_group.location,
        resource_group_name=example_resource_group.name,
        ip_configurations=[
            azure.network.NetworkInterfaceIpConfigurationArgs(
                name="internal",
                subnet_id=example_subnet.id,
                private_ip_address_allocation="Dynamic",
                public_ip_address_id=public_ip.id,
            )
        ],
    )

    example_linux_virtual_machine = azure.compute.LinuxVirtualMachine(
        f"{vm_name}_vm".replace("_", "-"),
        resource_group_name=example_resource_group.name,
        location=example_resource_group.location,
        size=vm_size,
        admin_username="adminuser",
        network_interface_ids=[example_network_interface.id],
        admin_ssh_keys=[
            azure.compute.LinuxVirtualMachineAdminSshKeyArgs(
                username="adminuser",
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
        custom_data=(lambda path: base64.b64encode(cloud_init.open('rb').read()).decode("ascii") )(cloud_init),
    )
