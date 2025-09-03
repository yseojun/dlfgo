# 배열 선언 (괄호와 공백 필요)
dataset=("beans")
decomp=("2d")
levels=("1")

# 반복문
for l in "${levels[@]}"; do
    for dc in "${decomp[@]}"; do
        for ds in "${dataset[@]}"; do
            echo "decomp: $dc, dataset: $ds, level: $l"
            python run_vid.py --datadir "/data/ysj/dataset/stanford_half/$ds" \
                            --basedir "/data/ysj/result/dlfgo_logs/250901" \
                            --dataset_name stanford \
                            --render_test \
                            --decomp "$dc" --levels "$l" --gpuid 0 --epoch 6000 \
                            --expname "$ds"_"$dc" --dump_images --grid_dim 16
            sleep 3
        done
    done
done
