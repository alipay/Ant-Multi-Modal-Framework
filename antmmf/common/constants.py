# Copyright (c) 2023 Ant Group and its affiliates.


imdb_version = 1
FASTTEXT_WIKI_URL = (
    "https://dl.fbaipublicfiles.com/pythia/pretrained_models/fasttext/wiki.en.bin"
)

CLEVR_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"

VISUAL_GENOME_CONSTS = {
    "imdb_url": "https://dl.fbaipublicfiles.com/pythia/data/imdb/visual_genome.tar.gz",
    "features_url": "https://dl.fbaipublicfiles.com/pythia/features/visual_genome.tar.gz",
    "synset_file": "vg_synsets.txt",
}

DOWNLOAD_CHUNK_SIZE = 1024 * 1024

# save keys that should passed from dataset to model.
REGISTRY_FOR_MODEL = "registry_for_model"
# save in registry indicate current dataset' name, so the model can be decoupling with
# specific dataset
DATASET_NAME = "dataset_name"


# modality
IMAGE_MODALITY = "image"
IMAGE_MODALITY_ID = 0
TEXT_MODALITY = "text"
TEXT_MODALITY_ID = 1
VISION_MODALITY = "vision"  # 视觉与图像的区别在于其存在时序信息，故而将其单独作为一个模态来标记
VISION_MODALITY_ID = 2

# strings related to image modality
IMAGES_STR = "images"
IMAGE_NAME_STR = "image_name"
# use these keys to indicate image_name in Sample
POSSIBLE_IMAGE_NAME_STRS = ["image_name", "img_path"]
IMG_TOKEN_INTERVAL_STR = "img_token_interval"  # number of separate-tokens between image tokens for attrim-MMBT

# result keys for image detection task
DETECTION_RESULTS_KEYS = ["pred_boxes", "pred_logits"]

# related to tokenization
CLS_ID_STR = "cls_id"
SEP_ID_STR = "sep_id"
LM_LABEL_IDS_STR = "lm_label_ids"
INTER_TOKEN_ID_STR = (
    "inter_token_id"  # separate-token between image tokens for attrim-MMBT
)


# constant strings
CONFIG_STR = "config"
INFO_STR = "info"
TRAINING_PARAMETERS_STR = "training_parameters"
NUM_WORKERS_STR = "num_workers"
BATCH_SIZE_STR = "batch_size"
REPORT_FOLDER_STR = "report_folder"
EXPERIMENT_NAME_STR = "experiment_name"
REPORT_FORMAT_STR = "report_format"
ID_STR = "id"
QUESTION_ID_STR = "question_id"
SAMPLER_STR = "sampler"
SHUFFLE_STR = "shuffle"
PRETRAINED_STR = "pretrained"

# state strings
STATE = "antmmf_state"
# disable loss calculation warning during serving
STATE_ONLINE_SERVING = "online_serving"
STATE_LOCAL = "local"
EVALAI_INFERENCE = "evalai_inference"

# special symbols
CLS_TOKEN_STR = "[CLS]"
SEP_TOKEN_STR = "[SEP]"
MASK_TOKEN_STR = "[MASK]"

USE_FEATURE_STR = "use_features"
FEATURES_STR = "features"
DATASETS_STR = "datasets"
FEATURE_KEY_STR = "feature_key"
FEATURE_PATH_STR = "feature_path"
MAX_FEATURES_STR = "max_features"
FEATURE_DIM_STR = "feature_dim"

DEPTH_FIRST_STR = "depth_first"

# feature name
IMAGE_FEAT_STR = "image_feat"
IMAGE_TEXT_STR = "image_text"
IMAGE_BBOXES_STR = "image_bboxes"
IMAGE_FEATURE_STR = "image_feature"
IS_OCR_STR = "is_ocr"
IMAGE_BBOX_SOURCE_STR = "image_bbox_source"
OUTPUT_REPRESENTATIONS = "output_representations"

# read & write
WRITER_STR = "writer"
FAST_READ_STR = "fast_read"

# return information
RETURN_FEATURES_INFO_STR = "return_features_info"

# database name
COCO_DATABASE_NAME = "coco"

# file extensions
LMDB_EXT_STR = ".lmdb"
NPY_EXT_STR = ".npy"

# environment variable name of bert pretrained models dir
BERT_PRETRAINED_MODELS_ENV_VAR = "PYTORCH_TRANSFORMERS_CACHE"

# environment variable name of antmmf pretrained models dir
ANTMMF_PRETRAINED_MODELS_ENV_VAR = "ANTMMF_PRETRAINED_CACHE"

# Knowledge graph target length
KGBERT_MAX_TARGET_LEN = 300

# environment variable name of torchvision models dir,
# try to find torchvision models  in $TORCH_HOME/checkpoints
TORCHVISION_PRETAINED_MODELS_ENV_VAR = "TORCH_HOME"

# environment variable name as root dir for online antmmf models
USER_MODEL_HOME = "USER_MODEL_HOME"

# LOSS NAMES
SS_CE_LOSS = "self-supervised-ce-loss"
SS_TRAIN = "self_supervised"

# for interpretation
SS_GRAD_INPUT = "grad_input"
SS_SALIENCE = "salience"

# Modeling
HIER_CLASS_SEP = "-"

# Numerical flooring etc
EPSILON = 1e-10

# NCLF related names
UNARY = "unary"
BINARY = "binary"
NODE_TYPE = "n"
EDGE_TYPE = "e"
LINK_TYPE = "l"
GROUND_TYPE = "g"
BLANKET_TYPE = "b"

# Doin constants
CHAR_LINK = 1
ROW_LINK = 2
COL_LINK = 3
