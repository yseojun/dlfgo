# 배열 선언 (괄호와 공백 필요)
dataset=("knights") 
decomp=("2d")
levels=("1")

# 반복문
# python run.py --dataset_name stanford --datadir ../../../data/stanford_half/tarot_small --render_test --epoch 1 --gpuid 1 --grid_num 8 8 128 128 --expname paper_2dcomb --decomp 2d --grid_dim 16

for ds in "${dataset[@]}"; do
    for dc in "${decomp[@]}"; do
        for l in "${levels[@]}"; do
            echo "dataset: $ds, decomp: $dc, level: $l"
            python run.py --dataset_name stanford --datadir ../../../data/stanford_half/$ds --render_test --epoch 100 --gpuid 1 --expname paper_2dcomb_${ds} --decomp 2d --grid_dim 16
        done
    done
done