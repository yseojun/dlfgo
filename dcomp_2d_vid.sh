# 배열 선언 (괄호와 공백 필요)
dataset=("14_17_60s_2_1800f_960_540")
decomp=("2d")
levels=("1")

# 반복문
for l in "${levels[@]}"; do
    for dc in "${decomp[@]}"; do
        for ds in "${dataset[@]}"; do
            echo "decomp: $dc, dataset: $ds, level: $l"
            python run_vid_dataset_try.py --datadir "/data/ysj/dataset/lf_from_4dgs/$ds" \
                            --basedir "/data/ysj/result/dlfgo_vid_logs/251023_test" \
                            --dataset_name video \
                            --render_test \
                            --decomp "$dc" --levels "$l" --gpuid 1 --epoch 50 \
                            --expname "$ds"_"$dc" --dump_images --grid_dim 16 \
                            --grid_size_x 19 --grid_size_y 5 --frame_num 50 \
                            --ray_type "oneplane"
            sleep 3
        done
    done
done