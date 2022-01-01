from sacred import Experiment

ex = Experiment("transform_and_tell")


def _loss_names(d):
    ret = {
        "itm": 0,
        "clm": 0,
        # "mpp": 0,
        # "vqa": 0,
        # "nlvr2": 0,
        # "irtr": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "transform_and_tell"
    seed = 0
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    max_text_len = 512
    # Decoder Setting
    embed_size = 1024
    embed_output_dim = 1024
    padding_idx = 1
    init_size = 512
    left_pad = False
    dropout = 0.1
    decoder_conv_dim = 1024
    decoder_glu = True
    decoder_conv_type = "dynamic"
    weight_softmax = True
    decoder_attention_heads = 16
    weight_dropout = 0.1
    relu_dropout = 0.0
    input_dropout = 0.1
    decoder_normalize_before = False
    attention_dropout = 0.1
    decoder_ffn_embed_dim = 4096
    decoder_kernel_size_list = [3, 7, 15, 31]
    decoder_layers = 4
    final_norm = False
    vocab_size = 50265

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    # get_recall_metric = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False


# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def env_dandelin():
    data_root = "/data2fast/users/vkhanh/data/"
    log_dir = "/data2/vilt/result"
    num_gpus = 1
    num_nodes = 1


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_mlm_itm():
    exp_name = "transform_and_tell_mlm_itm"
    loss_names = _loss_names({"itm": 1, "clm": 1})
    batch_size = 4
    max_epoch = 10
    max_text_len = 512

@ex.named_config
def task_mlm_itm_randaug():
    exp_name = "mlm_itm_randaug"
    datasets = ["coco", "vg", "sbu", "gcc"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_mpp():
    exp_name = "mlm_itm_mpp"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mpp": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


@ex.named_config
def task_finetune_vqa_randaug():
    exp_name = "finetune_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end


@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000

