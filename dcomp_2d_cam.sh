# 배열 선언 (괄호와 공백 필요)
dataset=("14_17")
decomp=("2d")
levels=("1")

# 반복문
for l in "${levels[@]}"; do
    for dc in "${decomp[@]}"; do
        for ds in "${dataset[@]}"; do
            echo "decomp: $dc, dataset: $ds, level: $l"
            python run_vid.py --datadir "/data/ysj/dataset/$ds/14_17_dataset" \
                            --basedir "/data/ysj/result/dlfgo_vid_logs/250917" \
                            --dataset_name cam \
                            --render_test \
                            --decomp "$dc" --levels "$l" --gpuid 1 --epoch 600 \
                            --expname "$ds"_"$dc" --dump_images --grid_dim 16
            sleep 3
        done
    done
done
