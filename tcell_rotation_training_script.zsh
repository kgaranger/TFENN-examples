#!/usr/bin/env zsh

# Keep sessions open after script execution
keep_open=0

name="tcell_rotation";
results_dir="results/${name}/"

# DATASET
dim=2;
rotations=(175 20 222 298 340 45)

# TRAINING PARAMETERS
epochs=1500;
batch_size=256;

kernel_init="he_normal";
loss="se";
optimizer="adamw";
typeset -A optimizer_opt
scheduler="coc";
typeset -A scheduler_opt
scheduler_opt[peak_value]=1e-3;
scheduler_opt[div_factor]=3;
gradient_clip=1;

# LOGGING
save_interval=100;
print_interval=50;

p=$(((dim + 1) * dim / 2));

# MODEL
typeset -A model
model[m]=2;
model[n]=37;
model[f]="tensor";
model[k]="cubic";

typeset -a seeds=(
    0
    1
    2
    3
    4
)

for rotation in $rotations; do
    for seed in $seeds; do
        case_name="${name}_${seed}_${rotation}"

        echo "Starting ${case_name}";

        tmux new -d -s "$case_name" \
            "python -m scripts.train \
            -d datasets/tcell/rotated/tcell_train_rotated_$rotation.csv \
            --val_dataset_files datasets/tcell/rotated/tcell_val_rotated_$rotation.csv \
            --seed $seed \
            --use_io_rotator \
            -s $results_dir$case_name/00 \
            -i $([ "$model[f]" = "tensor" ] && echo "1 ")$p \
            -o $([ "$model[f]" = "tensor" ] && echo "1 ")$p \
            -m ${model[m]} \
            -n ${model[n]} \
            --logging_level INFO \
            --save_interval $save_interval \
            --print_epoch_interval $print_interval \
            --kernel_initializer $kernel_init \
            --loss $loss \
            --optimizer $optimizer \
            $([ ${#optimizer_opt[@]} -ne 0 ] && echo "--optimizer_options $(for key val in "${(@kv)optimizer_opt}"; do echo -n "$key=$val "; done)") \
            --scheduler $scheduler \
            $([ ${#scheduler_opt[@]} -ne 0 ] && echo "--scheduler_options $(for key val in "${(@kv)scheduler_opt}"; do echo -n "$key=$val "; done)") \
            --gradient_clip $gradient_clip \
            --max_epochs $epochs \
            --batch_size $batch_size \
            --scale_per_feature \
            -f ${model[f]} \
            ${model[k]:+"-k $model[k]"}  \
            $([ "$model[f]" = "tensor" ] && echo "--data_type tensor --data_notation voigt --data_tensor_dim $dim") \
            --jax_config jax_debug_nans jax_enable_x64 \
            $([ $keep_open = 1 ] && echo "; exec zsh -i") 
            "

    done
done
