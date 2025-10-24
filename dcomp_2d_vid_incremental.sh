#!/bin/bash

# 배열 선언 (괄호와 공백 필요)
dataset=("14_17_60s_2_1800f_960_540")
decomp=("2d")
levels=("1")

# 프레임 설정
FRAME_STEP=50
TOTAL_FRAMES=1800

# 반복문
for l in "${levels[@]}"; do
    for dc in "${decomp[@]}"; do
        for ds in "${dataset[@]}"; do
            # 0부터 1800까지 50프레임 단위로 반복
            for ((start_frame=0; start_frame<TOTAL_FRAMES; start_frame+=FRAME_STEP)); do
                end_frame=$((start_frame + FRAME_STEP))
                if [ $end_frame -gt $TOTAL_FRAMES ]; then
                    end_frame=$TOTAL_FRAMES
                fi
                
                echo "=========================================="
                echo "decomp: $dc, dataset: $ds, level: $l"
                echo "Training frames: $start_frame to $end_frame"
                echo "=========================================="
                
                python run_vid_dataset_try.py --datadir "/data/ysj/dataset/lf_from_4dgs/$ds" \
                                --basedir "/data/ysj/result/dlfgo_vid_logs/251023_test" \
                                --dataset_name video \
                                --render_test \
                                --decomp "$dc" --levels "$l" --gpuid 1 --epoch 50 \
                                --expname "$ds"_"$dc"_f"$start_frame"-"$end_frame" \
                                --dump_images --grid_dim 16 \
                                --grid_size_x 19 --grid_size_y 5 --frame_num $FRAME_STEP \
                                --start_frame $start_frame \
                                --ray_type "oneplane"
                
                sleep 3
            done
        done
    done
done

echo "All training sessions completed!"

