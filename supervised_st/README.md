## Supervised ST finetuning

Here you can find the scripts used to finetune a ZeroSwot model on supervised ST. The scripts are specifically for MuST-C En-De, but can be easily adapted to other datasets.

To run you need a ZeroSwot model and the MuST-C ST data. For example, to run `train_mustc_nllb600M_en-de.sh` you first need to train a speech encoder with `zs_st/exp_configs/mustc_nllb600M.yaml` and do checkpoint averaging.

```bash
bash ${ZS_ROOT}/supervised_st/train_mustc_nllb600M_en-de.sh
```

Due to the large number of parameters, we freeze the acoustic encoder, and embedding layer for the model based on nllb600M, while for nllb1.3B we also use LNA finetuning. Please refer to the appendix of the paper for more details. To change these settings check `examples/extended_siamese/models/siamese_zs_s2t_model.py` in the fairseq branch of ZeroSwot.
