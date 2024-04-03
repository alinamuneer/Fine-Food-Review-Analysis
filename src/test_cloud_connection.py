import os
print(os.getcwd())



from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from settings import CONFIG_AZURE_ML


from azureml.core import Workspace
from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration

ws = Workspace.from_config()

# # authenticate
credential = DefaultAzureCredential()

# # # Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=CONFIG_AZURE_ML.subscription_id,
    resource_group_name=CONFIG_AZURE_ML.resource_group,
    workspace_name=CONFIG_AZURE_ML.workspace_name,
)





# # authenticate
# credential = DefaultAzureCredential()


# # set name for logging
# mlflow.set_experiment("Develop on cloud tutorial")
# # enable autologging with MLflow
# mlflow.sklearn.autolog()


from azure.ai.ml.entities import AmlCompute

# Name assigned to the compute cluster
cpu_compute_target = "aa-capstone"

try:
    # let's see if the compute target already exists
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
    print(
        f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    # Let's create the Azure Machine Learning compute object with the intended parameters
    cpu_cluster = AmlCompute(
        name=cpu_compute_target,
        # Azure Machine Learning Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_DS11_V2",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=1,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=120,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )
    print(
        f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}"
    )
    # Now, we pass the object to MLClient's create_or_update method
    cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster)