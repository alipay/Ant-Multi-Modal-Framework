from sacred import Experiment

ex = Experiment("VLMo")


def _loss_names(d):
    ret = {
        "itm": 0,  # image-text matching loss
        "itc": 0,  # image-text contrastive loss
        "caption": 0,  # image captioning loss
        "mvlm": 0,  # masked language modeling loss
        "textmlm": 0,  # text-only masked language modeling
        "imagemlm": 0,  # image-only masked language modeling
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,  # retrieval task ft
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vlmo"
    seed = 1
    datasets = ["coco", "vg", "sbu", "gcc"]  # dataset name, the definition can refer to: vlmo/datamodules/__init__.py  # noqa
    loss_names = _loss_names({"itm": 0, "itc": 0, "mvlm": 0})  # training loss
    batch_size = 1024  # this is a desired batch size; pl trainer will accumulate gradients.

    # BEiT-v3 setting
    encoder_layers = 12  # the layer number of backbone
    encoder_embed_dim = 768  # the hidden size of tokenizer
    out_embed_dim = 768  # the hidden size of output embedding
    beit_version = "base"  # model size: base(0.4B)|large(1B)|huge(10B)
    beit3_vl_layers = 3  # the layer number of vl_backbone
    deepnorm_init = True  # init method
    share_layer = False  # if share the weight between layer within backbone
    share_attn = False  # if share the attention weight of different layer
    one_attn = False  # if share the attention weight of vision and language

    # Image setting
    train_transform_keys = ["square_transform_randaug"]  # train transform: refer to vlmo/transforms/__init__.py
    val_transform_keys = ["square_transform"]  # test transform: refer to refer to vlmo/transforms/__init__.py
    image_size = 224  # image size
    reclip_image_size = None  # reclip image size
    patch_size = 16  # patch size
    draw_false_image = 0  # if get negative image
    image_only = False  # only input image
    text_only = False  # # only input text

    # Video setting, video_num_frm is not None means video input
    video_num_frm = None

    # Visual tokenizer setting based on beit2
    tokenizer_model = "beit2_visual_tokenizer"
    codebook_size = 8192
    codebook_dim = 32
    visual_mask_size = 14
    visual_mask_num = 80

    # Text Setting
    lang = 'cn'  # language for zero-shot imagenet testing: cn|en
    vqav2_label_size = 3129
    max_text_len = 40  # the number of characters
    max_text_len_of_initckpt = 196
    tokenizer_type = "BertTokenizer"  # Chinese text
    vocab_size = 21128
    tokenizer = "./vocab.txt"
    whole_word_masking = True
    mlm_prob = 0.15  # language mask ratio
    draw_false_text = 0
    mvlm_prob = 0.50  # vision-langurage mlm task
    mask_ratio = 0  # flip: mask ratio for image

    # cap setting
    cap_onlytext = False  # default caption image to text

    # imagemlm setting
    split_data_for_imagemlm = False  # if True, split a batch data to two parts, and the first part for imagemlm.

    # itc setting
    itc_mask = False  # itc use masked token
    aggregate_nodes = -1  # aggregate nodes num for compute_itc, default -1 is for all nodes

    # Transformer Setting
    model_arch = "vlmo_base_patch16"
    drop_path_rate = 0.1

    # Downstream Setting
    get_recall_metric = False
    get_recall_rerank_metric = False
    get_zeroshot_metric = False
    get_muge_feat = False
    get_f30k_feat = False
    k_test = 32

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    use_sharded_training = False
    resume_during_training = False
    save_top_k = 10
    every_n_train_steps = 2000  # the step to save checkpoint
    log_metric_steps = 100  # the step to log metric

    # below params varies with the environment
    use_pcache = False  # data storage method: pcache or nas
    pcache_root = ""
    # main_site: pcache://multimodalproxyi-pool.cz50c.alipay.com:39999/mnt/
    # public_cloud: pcache://pcache_public_cloud.pcache.local:39999/mnt/abc7c88079a60b45ddfce7afa40720b7/
    gpu_env = "main_site"  # public_cloud or main_site
    data_root = ""  # data root for data list


    log_dir = "result"
    per_gpu_batchsize = 4  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16
    local_run = True
    flash_attn = False
    deepspeed_config = None  # "ds_config.json"
    coalesce_backbone = False
    mask_data = "v+l"  # 'v+l':choose input of imagemlm+textmlm task, 'vl': choose input of mvlm task.
    communication_benchmark = False
    checkpoint_activations = False

    # dataset setting
    single_cap = True  # if have only one caption
    random_one = False  # if choose one caption from caption list

    # ITC setting
    itc_feats_name = "cls_vlffn_feats"  # feat for itc loss
    itc_distill = ""
    itc_distill_dim = 1024
    itc_teacher_weights = ""

    # mup training setting
    mup = False
    base_encoder_embed_dim = 1
    delta_encoder_embed_dim = 2
    mup_encoder_attention_heads = 1
    base_encoder_ffn_embed_dim = 1
    delta_encoder_ffn_embed_dim = 2

    # atorch
    atorch_config = None
    compile_op = False
    optimizer_state_shard_save = False
    model_state_shard_save = False

    # itc loss
    local_loss = False
    use_dual_softmax = False

    num_frames = 1
# ----------------------- LMM pretraining config -----------------------

    # norm setting
    deepnorm = False

