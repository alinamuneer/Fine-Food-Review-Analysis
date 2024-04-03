from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace
from settings import CONFIG_AZURE_ML, AzureMlConfig

def get_workspace() -> Workspace:
    """Get Azure ML workspace from config."""
    return Workspace.from_config()

def get_ml_client(config_info:AzureMlConfig) -> MLClient:
    """Instantiate MLClient using Azure credentials and workspace details."""
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=config_info.subscription_id,
        resource_group_name=config_info.resource_group,
        workspace_name=config_info.workspace_name,
    )

def ensure_compute_cluster(ml_client: MLClient, cluster_name: str) -> AmlCompute:
    """Ensure a compute cluster exists or create a new one."""
    try:
        cluster = ml_client.compute.get(cluster_name)
        print(f"Using existing cluster: {cluster_name}")
    except Exception:
        print("Creating a new CPU compute target...")
        cluster = AmlCompute(
            name=cluster_name,
            type="amlcompute",
            size="STANDARD_DS11_V2",
            min_instances=0,
            max_instances=1,
            idle_time_before_scale_down=120,
            tier="Dedicated",
        )
        cluster = ml_client.compute.begin_create_or_update(cluster).result()
        print(f"Created cluster: {cluster_name} with size {cluster.size}")
    return cluster

if __name__ == "__main__":
    ws = get_workspace()
    ml_client = get_ml_client(CONFIG_AZURE_ML)
    compute_cluster = ensure_compute_cluster(ml_client, "aa-capstone")
