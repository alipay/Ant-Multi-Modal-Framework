import json
import os
import urllib
from tqdm import tqdm

from vlmo.config import config, _loss_names  # noqa
from vlmo.modules import VLMo
from vlmo.transforms import keys_to_transforms


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True, unit_divisor=1024
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def config_setting(custom_config: dict):
    cfg = eval("config")()
    for k, v in custom_config.items():
        cfg[k] = v
    return cfg


def load_from_config(model_config):
    if isinstance(model_config, str):
        model_config = json.loads(open(model_config, 'r').read())
    else:
        assert isinstance(model_config, dict)

    model_url = model_config.pop('model_url', None)

    if model_url:
        load_path = _download(model_url, os.path.expanduser("~/.cache/m2_encoder"))
    else:
        from modelscope import snapshot_download
        modelscope_cfg = model_config.pop('modelscope', None)
        model_dir = snapshot_download(**modelscope_cfg)
        load_path = os.path.join(model_dir, model_config.pop('model_file'))

    cfg = config_setting(model_config)
    cfg["load_path"] = load_path

    if cfg["flash_attn"]:
        from vlmo.utils.patch_utils import patch_torch_scale_with_flash_attn
        patch_torch_scale_with_flash_attn()

    model = VLMo(cfg)

    from vlmo.modules.vlmo_module import get_pretrained_tokenizer
    txt_processor = get_pretrained_tokenizer(cfg["tokenizer_type"], from_pretrained=cfg["tokenizer"])
    img_processor = keys_to_transforms(cfg["val_transform_keys"], size=cfg["image_size"])[0]

    return model, [txt_processor, img_processor]
