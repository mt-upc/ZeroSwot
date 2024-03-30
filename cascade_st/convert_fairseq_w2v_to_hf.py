# modified from transformers/src/transformers/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py

from collections import OrderedDict
from transformers import Wav2Vec2ForCTC, logging
import torch

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "adapter_layer": "encoder.layers.*.adapter_layer",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "quantizer.weight_proj": "quantizer.weight_proj",
    "quantizer.vars": "quantizer.codevectors",
    "project_q": "project_q",
    "final_proj": "project_hid",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
    "pooling_layer.linear": "projector",
    "pooling_layer.projection": "classifier",
}
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
    "projector",
    "classifier",
]


def read_txt_into_dict(filename):
    result = {}
    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            line = line.strip()
            if line:
                words = line.split()
                key = line_number
                value = words[0]
                result[key] = value
    return result


def set_recursively(key, value, full_name, weight_type, hf_pointer):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    hf_param_name = None
    for param_key in PARAM_MAPPING.keys():
        if full_name.endswith(param_key):
            hf_param_name = PARAM_MAPPING[full_name.split(".")[-1]]
            weight_type = "param"

    if weight_type is not None and weight_type != "param":
        hf_shape = getattr(hf_pointer, weight_type).shape
    elif weight_type is not None and weight_type == "param":
        shape_pointer = hf_pointer
        for attribute in hf_param_name.split("."):
            shape_pointer = getattr(shape_pointer, attribute)
        hf_shape = shape_pointer.shape

        # let's reduce dimension
        value = value[0]
    else:
        hf_shape = hf_pointer.shape

    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "param":
        for attribute in hf_param_name.split("."):
            hf_pointer = getattr(hf_pointer, attribute)
        hf_pointer.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")


def rename_dict(key, value, full_name, weight_type, hf_dict):
    hf_param_name = None
    for param_key in PARAM_MAPPING.keys():
        if full_name.endswith(param_key):
            hf_param_name = PARAM_MAPPING[full_name.split(".")[-1]]
            weight_type = "param"

    if weight_type is not None and weight_type != "param":
        full_key = ".".join([key, weight_type])
    elif weight_type is not None and weight_type == "param":
        full_key = ".".join([key, hf_param_name])
    else:
        full_key = key

    hf_dict[full_key] = value if "lm_head" in full_key else value[0]


PARAM_MAPPING = {
    "W_a": "linear_1.weight",
    "W_b": "linear_2.weight",
    "b_a": "linear_1.bias",
    "b_b": "linear_2.bias",
    "ln_W": "norm.weight",
    "ln_b": "norm.bias",
}


def load_wav2vec2_layer(name, value, hf_model=None, hf_dict=None):
    is_used = False
    for key, mapped_key in MAPPING.items():
        mapped_key = "wav2vec2." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
        if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
            is_used = True
            if "*" in mapped_key:
                layer_index = name.split(key)[0].split(".")[-2]
                mapped_key = mapped_key.replace("*", layer_index)
            if "weight_g" in name:
                weight_type = "weight_g"
            elif "weight_v" in name:
                weight_type = "weight_v"
            elif "bias" in name:
                weight_type = "bias"
            elif "weight" in name:
                # TODO: don't match quantizer.weight_proj
                weight_type = "weight"
            else:
                weight_type = None
            if hf_dict is not None:
                rename_dict(mapped_key, value, name, weight_type, hf_dict)
            else:
                set_recursively(mapped_key, value, name, weight_type, hf_model)
            return is_used
    return is_used


def recursively_load_weights(fairseq_dict, hf_model, is_headless):
    unused_weights = []

    feature_extractor = hf_model.wav2vec2.feature_extractor

    for name, value in fairseq_dict.items():
        is_used = False
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            is_used = load_wav2vec2_layer(name, value, hf_model)
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    if type_id == 0:
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)
        

def convert(checkpoint_path, hf_model_name="facebook/wav2vec2-large-960h-lv60-self", is_siamese=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hf_wav2vec = Wav2Vec2ForCTC.from_pretrained(hf_model_name).to(device)
    original_dict = torch.load(checkpoint_path, map_location=device)["model"]
    
    if is_siamese:
        w2v_dict = OrderedDict()
        for key in original_dict.keys():
            if "encoder.spch_encoder." in key:
                w2v_dict[key.replace("encoder.spch_encoder.", "")] = original_dict[key]
    else:
        w2v_dict = original_dict

    recursively_load_weights(w2v_dict, hf_wav2vec, False)
    
    if is_siamese:
        hf_wav2vec.lm_head.weight.data = original_dict["decoder.proj.weight"]
        hf_wav2vec.lm_head.bias.data = original_dict["decoder.proj.bias"]

    return hf_wav2vec