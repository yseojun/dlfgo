# 배열 선언 (괄호와 공백 필요)
dataset=("ambushfight_1")
decomp=("2d")
levels=("1")

# 반복문
for l in "${levels[@]}"; do
    for dc in "${decomp[@]}"; do
        for ds in "${dataset[@]}"; do
            echo "decomp: $dc, dataset: $ds, level: $l"
            python run_vid.py --datadir "/data/ysj/dataset/LF_video_crop_half/$ds" \
                            --basedir "/data/ysj/result/dlfgo_vid_logs/250902" \
                            --dataset_name video \
                            --render_test \
                            --decomp "$dc" --levels "$l" --gpuid 0 --epoch 600 \
                            --expname "$ds"_"$dc" --dump_images --grid_dim 16 \
                            --render_only
            sleep 3
        done
    done
done
