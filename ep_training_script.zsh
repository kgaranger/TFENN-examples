#!/usr/bin/env zsh

# Keep sessions open after script execution
keep_open=0

name="ep";
results_dir="results/${name}/"

# DATASET
dim=2;
train_set="datasets/elasto_plastic/msmat_200incs_0-003_e1e2_*.csv";
dataset_split=0.8;

# TRAINING PARAMETERS
epochs=1500;
batch_size=64;

kernel_init="he_normal";
loss="se";
activation="tanh";
optimizer="adamw";
typeset -A optimizer_opt
optimizer_opt[weight_decay]=1e-2;
scheduler="coc";
typeset -A scheduler_opt
scheduler_opt[peak_value]=1e-3;
scheduler_opt[pct_start]="0.5";
scheduler_opt[div_factor]=3;
scheduler_opt[final_div_factor]="1000";
gradient_clip=1e3;

# LOGGING
save_interval=100;
print_interval=10;

p=$(((dim + 1) * dim / 2));

# MODELS
typeset -A model1 model2 model3 model4;
model1[m]=2;
model1[n]=64;
model1[f]="scalar";
model2[m]=2;
model2[n]=49;
model2[f]="tensor";
model2[k]="cubic";
model3[m]=2;
model3[n]=100;
model3[f]="scalar";
model4[m]=2;
model4[n]=200;
model4[f]="scalar";

typeset -a models=(
    model1
    model2
    model3
    model4
)

# DATASETS MAX SAMPLES
max_datasets[1]=8000;
max_datasets[2]=5000;
max_datasets[3]=3000;
max_datasets[4]=1000;

for model in $models; do
    local -A model=("${(Pkv@)model}")
    for max_dataset in "${max_datasets[@]}"; do

        case_name="${name}_${model[m]}x${model[n]}_${model[f]}";
        ((${model[k]+1})) && case_name="${case_name}_${model[k]}";
        case_name="${case_name}_${max_dataset}";

        echo "Starting ${case_name}";

        tmux new -d -s "$case_name" \
            "python -m scripts.train \
            -d $train_set \
            ${val_set:+"--val_dataset_files $val_set"} \
            ${dataset_split:+"--dataset_split $dataset_split"} \
            --max_dataset_size $max_dataset \
            -s $results_dir$case_name/00 \
            --file_in_cols 2 3 4 --file_out_cols 5 6 7 \
            -i $([ "$model[f]" = "tensor" ] && echo "1 ")$p \
            -o $([ "$model[f]" = "tensor" ] && echo "1 ")$p \
            -m ${model[m]} \
            -n ${model[n]} \
            --logging_level INFO \
            --save_interval $save_interval \
            --print_epoch_interval $print_interval \
            --kernel_initializer $kernel_init \
            --loss $loss \
            --activation $activation \
            --optimizer $optimizer \
            $([ ${#optimizer_opt[@]} -ne 0 ] && echo "--optimizer_options $(for key val in "${(@kv)optimizer_opt}"; do echo -n "$key=$val "; done)") \
            --scheduler $scheduler \
            $([ ${#scheduler_opt[@]} -ne 0 ] && echo "--scheduler_options $(for key val in "${(@kv)scheduler_opt}"; do echo -n "$key=$val "; done)") \
            --gradient_clip $gradient_clip \
            --max_epochs $epochs \
            --batch_size $batch_size \
            --scale_per_feature \
            -t gru \
            -f ${model[f]} \
            ${model[k]:+"-k $model[k]"}  \
            $([ "$model[f]" = "tensor" ] && echo "--data_type tensor --data_notation voigt --data_tensor_dim $dim") \
            --jax_config jax_debug_nans jax_enable_x64 \
            $([ $keep_open = 1 ] && echo "; exec zsh -i") 
            "

    done
done
