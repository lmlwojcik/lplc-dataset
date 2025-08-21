run_yolo_im() {
    for f in 0 1 2 3 4
    do
        python3 main.py \
            -c configs/config_${1}.json \
            -d 0 \
            -n ${1}${f}_1 \
            -dt configs/split_configs/config_classes_${2}.json \
            -f ${f}_2 \
            -p -pt test \
            -t configs/train_configs/train_cfg_yolo.json \
            -v configs/train_configs/test_cfg_yolo.json \

        python3 main.py \
            -c configs/config_${1}.json \
            -d 0 \
            -n ${1}${f}_2 \
            -dt configs/split_configs/config_classes_${2}.json \
            -f ${f}_2 \
            -p -pt test \
            -t configs/train_configs/train_cfg_yolo.json \
            -v configs/train_configs/test_cfg_yolo.json \

    done
}

run_torch_im() {
    for f in 0 1 2 3 4
    do
        python3 main.py \
            -c configs/config_${1}.json \
            -d 0 \
            -n ${1}${f}_1 \
            -dt configs/split_configs/config_classes_${2}.json \
            -f ${f}_2 \
            -p -pt test \
            -t configs/train_configs/train_cfg_torch.json \
            -v configs/train_configs/test_cfg_torch.json \

        python3 main.py \
            -c configs/config_${1}.json \
            -d 0 \
            -n ${1}${f}_2 \
            -dt configs/split_configs/config_classes_${2}.json \
            -f ${f}_2 \
            -p -pt test \
            -t configs/train_configs/train_cfg_torch.json \
            -v configs/train_configs/test_cfg_torch.json \

    done
}

run_yolo_im yolo_m base
run_yolo_im yolo_m 1
run_yolo_im yolo_m 2

run_torch_im resnet50 base
run_torch_im resnet50 1
run_torch_im resnet50 2

run_torch_im vitb16 base
run_torch_im vitb16 1
run_torch_im vitb16 2
