#!/bin/bash
#SBATCH --job-name=mochi_train
#SBATCH --partition=q-dyvm6xra
#SBATCH --nodes=2              # Number of nodes (adjust based on config['world_size'])
#SBATCH --nodelist=hk01dgx029,hk01dgx030
#SBATCH --tasks-per-node=1      # One task per node for Ray
#SBATCH --cpus-per-task=32      # Adjust based on your needs
#SBATCH --mem=1000GB             # Memory per node
#SBATCH --gres=gpu:8            # One GPU per task
#SBATCH --exclusive
#SBATCH --output=/home/dyvm6xra/dyvm6xrauser02/raphael/mochi-1-preview/models/src/genmo/mochi_preview/logs/mochi_%j.out
#SBATCH --error=/home/dyvm6xra/dyvm6xrauser02/raphael/mochi-1-preview/models/src/genmo/mochi_preview/logs/mochi_%j.err

# Debug information
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"

cd /home/dyvm6xra/dyvm6xrauser02/raphael/mochi-1-preview/models/
module load cuda12.4
source .venv/bin/activate

# Get all network interfaces
echo "Network interfaces on head node:"
ip addr show

# Get the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Define ports first
port=6379
ray_client_port=10001
node_manager_port=20000
object_manager_port=20001
temp_dir="/tmp/ray_${SLURM_JOB_ID}"

# Get the head node's hostname and IP
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Add IPv6 handling
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# Export the ip_head variable
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# Test connectivity between nodes
for node in "${nodes_array[@]}"; do
    echo "Testing connectivity to $node"
    srun --nodes=1 --ntasks=1 -w "$node" ping -c 3 "$head_node_ip"
done

# Create temp directory on all nodes
for node in "${nodes_array[@]}"; do
    srun --nodes=1 --ntasks=1 -w "$node" mkdir -p "$temp_dir"
done

# Start Ray head node
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 32 --num-gpus 8 --block &

# Increase wait time for head node to start
echo "Waiting for head node to start..."
sleep 20

# Simplify worker node startup - remove complex retry logic
worker_num=$((SLURM_NNODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus 32 --num-gpus 8 \
        --block &
    sleep 5
done

# Final check of cluster status
echo "Final Ray cluster status:"
# srun --nodes=1 --ntasks=1 -w "$head_node" ray status

# Export environment variables
export RAY_ADDRESS="$head_node_ip:$port"
export RAY_HEAD_IP="$head_node_ip"

# Run the training script with the required arguments
echo "Starting training script"
python -u /home/dyvm6xra/dyvm6xrauser02/raphael/mochi-1-preview/models/src/genmo/mochi_preview/train_mochi_pl_multinodes.py \
    --head_addr="$head_node_ip:$port" \
    --world_size=$(( ${#nodes_array[@]} * 8 )) \
    --model_dir="/home/dyvm6xra/dyvm6xrauser02/AIGC/mochi" \
    --data_path="/home/dyvm6xra/dyvm6xrauser02/data/vidgen1m"

# Cleanup
echo "Cleaning up Ray processes..."
for node in "${nodes_array[@]}"; do
    srun --nodes=1 --ntasks=1 -w "$node" ray stop
    srun --nodes=1 --ntasks=1 -w "$node" rm -rf "$temp_dir"
done

wait