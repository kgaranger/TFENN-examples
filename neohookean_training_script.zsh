#!/usr/bin/env zsh

# Keep sessions open after script execution
keep_open=0

name="neohookean";
results_dir="results/${name}/"

# DATASET
dim=3;
train_set="datasets/neohookean/neohookean_train.csv";
val_set="datasets/neohookean/neohookean_val.csv";

# TRAINING PARAMETERS
epochs=1500;
batch_size=256;

kernel_init="he_normal";
loss="se";
activation="leaky_relu";
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

# MODELS
typeset -A model1 model2 model3 model4 model5 model6;
model1[m]=2;
model1[n]=512;
model1[f]="scalar";
model2[m]=2;
model2[n]=365;
model2[f]="tensor";
model2[k]="isotropic";
model3[m]=2;
model3[n]=32;
model3[f]="scalar";
model4[m]=2;
model4[n]=23;
model4[f]="tensor";
model4[k]="isotropic";
model5[m]=3;
model5[n]=32;
model5[f]="scalar";
model6[m]=3;
model6[n]=23;
model6[f]="tensor";
model6[k]="isotropic";

typeset -a models=(
    model1
    model2
    model3
    model4
    model5
    model6
)

# DATASETS MAX SAMPLES
max_datasets[1]=80000;
max_datasets[2]=40000;
max_datasets[3]=20000;
max_datasets[4]=10000;
max_datasets[5]=5000;

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
            -f ${model[f]} \
            ${model[k]:+"-k $model[k]"}  \
            $([ "$model[f]" = "tensor" ] && echo "--data_type tensor --data_notation voigt --data_tensor_dim $dim") \
                --jax_config jax_debug_nans jax_enable_x64 \
            $([ $keep_open = 1 ] && echo "; exec zsh -i") 
            "

   done
done
