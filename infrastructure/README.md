## Dependencies

[Pulumi](https://www.pulumi.com/docs/get-started/install/) - Infrastructure as code

[Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)

## Deploy VMs


### Ensure you are logged into the azure CLI

```bash
as login
```

Set the subscription:

```bash
az account set -s '{Azure-subscription-id}'
```

Verify the correct subscription is now `isDefault`:

```bash
az account list -o table
```

### Deploy infrastructure

Edit [__main__.py](__main__.py) to customise the configuration (number of VMs, machine size etc).

The [Cloud init script](vm_config.yaml) include public keys, user management and dependency install.


From the infrastructure directory run the following to deploy infrastructure:

```bash
pulumi up
```

You may be prompted to create a pulumi account

## Debugging cloud init

- Some useful [hints](https://blog.gripdev.xyz/2019/02/19/debugging-cloud-init-on-ubuntu-in-azure-or-anywhere/)

## Manual config

SSH into machine. Ensure cloud init has finished:

```bash
cloud-init status -w
```

The file share will be mounted at `/mnt/vmexperiments/quippstore/`

### First time for each user

```bash
source /miniconda/bin/activate
conda init
conda activate py38_quipp
```
