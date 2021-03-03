## Requirements

[Pulumi](https://www.pulumi.com/docs/get-started/install/) - Infrastructure as code

[Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)

## Deploy VMs


### Ensure you are logged into the azure CLI

```bash
as login
```

Set the subscription:

```bash
az account set -s 'c100c04d-970f-4d77-999a-add73cf49668"'
```

Verify the correct subscription is now `isDefault`:

```bash
az account list -o table
```

### Deploy infrastructure

Edit [__main__.py](__main__.py) to customise the configuration (VMs, Users).

- VMs should have unique names, but can have different sizes and location.  
- Add user names and public ssh keys

The script will render a [Cloud init template](cloud_init_template.yaml) which adds users and dependencies to the VM.


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

Once complete check configuration was successful:

```bash
cloud-init status --long
```

To see logs:

```bash
sudo cat /var/log/cloud-init-output.log
```

The file share will be mounted at `/mnt/vmexperiments/quippstore/`

## First time for each user

```bash
source /miniconda/bin/activate
conda init
conda activate py38_quipp
```

## Remount fileshare
If you cant see the fileshare remount it by running:

```bash
sudo bash /etc/mount_filestorage.sh
```