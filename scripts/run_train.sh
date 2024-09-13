if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

now=$(date +"%Y%m%d_%H%M%S")
# NCCL_DEBUG=INFO python -m torch.distributed.launch   --nproc_per_node=2  --master_port=1239 train.py  --config ${config} --log_time $now

NCCL_DEBUG=INFO torchrun  --nproc_per_node=4  --master_port=1238 train.py  --config ${config} --log_time $now