import os
from utils.general import quietly_run


def install_model_libs(cfg):
    quietly_run('pip install tensorflow-addons')
    if cfg.arch_name.startswith('efnv1'):
        quietly_run('pip install efficientnet', debug=False)
    elif cfg.arch_name.startswith('efnv2'):
        quietly_run('pip install -Uq keras-efficientnet-v2', debug=False)
    else:
        # Look for model in tfimm.
        # On colab@GPU tf 2.8.2 works with tfa 0.17.1 and tfimm 0.2.7 (newer python, cuDNN)
        if cfg.cloud == 'kaggle':
            # Problem: tfimm requires tf >= 2.5.0, which breaks 
            #          kaggle_secrets.UserSecretsClient.get_gcloud_credential, which is needed for
            #          private datasets (tfrec can only be accessed from TPU via GCS file system).
            # Solutions:
            #   (1) Update TF on kaggle together with tensorflow-gcs-config, but takes time.
            #   (2) Install tfimm with --no-deps. Best option for TPU, but some models won't work.
            # Current GPU image does not work due to cuDNN issues:
            # with 2.5.0 and convnext: NotFoundError: 'StatelessRandomGetKeyCounter'...
            # with 2.6.0 tfimm import raises "AlreadyExistsError: Another metric with the same name already exists"
            # with 2.6.4 tfimm resnet18 instantiation dies silently on GPU notebook (cuDNN 8.0.5/8005)
            # with 2.8.2 CuDNN 8.0.5 is incompatible (TF was compiled with: 8.1.0)
            # Solution for GPU: 2021-02-23-image with TF 2.4.1 TFA 0.12.1 and tfimm 0.2.7 works.
            tf_version = None
            if tf_version:
                quietly_run(f'pip install tensorflow=={tf_version}')  # install before tfimm to avoid RuntimeError
                print("installing cloud-tpu-client...")
                quietly_run('pip install cloud-tpu-client')
                import tensorflow as tf
                print("importing Client")
                from cloud_tpu_client import Client
                print("getting Client instance...")
                c = Client()
                print("got client and tf is", tf.version)
                c.configure_tpu_version(tf_version, restart_type='ifNeeded')
                print("client configured")
                #c.wait_for_healthy()   # waits forever
                #print("client healthy")
                quietly_run(f'pip install tensorflow-gcs-config=={tf_version}')
                print("Updated TF, TPU client, and GCS config to version", tf_version)
                quietly_run('pip install tfimm timm')
                print("installed tfimm")
            else:
                quietly_run('pip install --no-deps tfimm timm')
            assert os.environ.get('NV_CUDNN_VERSION') != '8.0.5.39', 'cuDNN broken: use image2021-02-23 for tfimm GPU training!'
        else:
            quietly_run('pip install -q tfimm timm')
