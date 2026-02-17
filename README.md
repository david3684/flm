# Text Generation (Base: [DUO](https://github.com/s-sahoo/duo))

In this project, we use OpenWebText dataset.

We used H100 4 gpus for training and inference.

Training for single rectification iteration takes 3 hours on our machine.

## Usage

To get started with this project, follow these steps:

1. Install requirement
    ```bash
    # We used docker image with torch==2.3.1+cu121
    pip install -r requirements.txt
    ```

2. Download Pretrained models (of DUO)
    ```bash
    # Finetuned models with ReDi
    # Download from Hugginface(https://huggingface.co/Ugness/ReDi)

    # Or
    # Pretrained models from origin DUO
    # Download origin DUO checkpoint from Google Drive folder(https://drive.google.com/drive/folders/1JpqFM8XRvifwIkjWPfMyuDvu41r1yk0t?usp=share_link).
    ```

3. Download OpenWebText dataset
    ```bash
    # The training code automatically downloads the OWT dataset onto your local machine.
    ```

4. Use ReDi method
    ```bash
    # Train
    ## Create Rectified Coupling (origin ReDi)
    bash scripts/train_owt_duo_reflow_greedy_gen.sh --checkpoint_path "CKPT_PATH" --ckpt "ReDi1" --dataset_path "DATASET_PATH"
    ## Create Rectified Coupling (perturbed ReDi)
    bash scripts/train_owt_duo_reflow_greedy_gen_perturbed.sh --checkpoint_path "CKPT_PATH" --ckpt "ReDi1" --dataset_path "DATASET_PATH" --owt_path "OWT_PATH"

    ## Add symbolic links to the cache_dir for training
    ln -s /PATH/TO/OWT/openwebtext "DATASET_PATH"/openwebtext
    ln -s /PATH/TO/OWT/openwebtext-train_train_bs1024_wrapped.dat "DATASET_PATH"/openwebtext-train_train_bs1024_wrapped.dat
    ln -s /PATH/TO/OWT/openwebtext-valid_validation_bs1024_wrapped.dat "DATASET_PATH"/openwebtext-valid_validation_bs1024_wrapped.dat

    ## Train a model (origin ReDi)
    bash scripts/train_owt_duo_reflow_train.sh --checkpoint_path "CKPT_PATH" --ckpt "ReDi1" --dataset_path "DATASET_PATH"
    ## Train a model (perturbed ReDi)
    bash scripts/train_owt_duo_reflow_train_perturbed.sh --checkpoint_path "CKPT_PATH" --ckpt "ReDi1" --dataset_path "DATASET_PATH"


    # Test
    ## Test by llama3.1 (default option)
    bash scripts/gen_ppl_owt_duo.sh --checkpoint_path "CKPT_PATH" --ckpt "ReDi1" --steps 32
    ## Test by gpt2
    bash scripts/gen_ppl_owt_duo_gpt2.sh --checkpoint_path "CKPT_PATH" --ckpt "ReDi1" --steps 32
    ## Test with di4c ckpt
    bash scripts/gen_ppl_owt_duo_di4c.sh --checkpoint_path "CKPT_PATH" --ckpt "ReDi1" --steps 32
    ## Test with TC
    bash scripts/gen_ppl_tc_owt_duo.sh --checkpoint_path "CKPT_PATH" --ckpt "ReDi1" --steps 32
    ```
