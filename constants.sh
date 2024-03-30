#!/bin/bash

export MUSTC_NAME="MUSTC_v1.0"
export COVOST_NAME="CoVoST2"
export CV_NAME="CommonVoice"
export LS_NAME="LibriSpeech"

export DATASETS=("$MUSTC_NAME" "$COVOST_NAME", "$CV_NAME", "$LS_NAME")

declare -A SPLITS
SPLITS=([$MUSTC_NAME]="train dev tst-COMMON" [$COVOST_NAME]="train dev test")

export MODELS=("mbart-large-50-one-to-many-mmt" "nllb-200-distilled-600M" "nllb-200-1.3B" "nllb-200-distilled-1.3B" "nllb-200-3.3B")

declare -A SRC_LANGS
export SRC_LANGS=([$MUSTC_NAME]="en" [$COVOST_NAME]="en")

declare -A TGT_LANGS
export TGT_LANGS=([$MUSTC_NAME]="de es fr it nl pt ro ru" [$COVOST_NAME]="ar ca cy de et fa id ja lv mn sl sv-SE ta tr zh-CN")

declare -A NLLB_LANG_CODES
export NLLB_LANG_CODES=(["en"]="eng_Latn" ["de"]="deu_Latn" ["es"]="spa_Latn" ["fr"]="fra_Latn" ["it"]="ita_Latn" ["nl"]="nld_Latn" ["pt"]="por_Latn" ["ro"]="ron_Latn" ["ru"]="rus_Cyrl" ["tr"]="tur_Latn" ["fa"]="pes_Arab" ["sv-SE"]="swe_Latn" ["mn"]="khk_Cyrl" ["zh-CN"]="zho_Hans" ["cy"]="cym_Latn" ["ca"]="cat_Latn" ["sl"]="slv_Latn" ["et"]="est_Latn" ["id"]="ind_Latn" ["ar"]="arb_Arab" ["ta"]="tam_Taml" ["lv"]="lvs_Latn" ["ja"]="jpn_Jpan")

declare -A MBART_LANG_CODES
export MBART_LANG_CODES=(["en"]="en_XX" ["de"]="de_DE" ["es"]="es_XX" ["fr"]="fr_XX" ["it"]="it_IT" ["nl"]="nl_XX" ["pt"]="pt_XX" ["ro"]="ro_RO" ["ru"]="ru_RU" ["tr"]="tr_TR" ["fa"]="fa_IR" ["sv-SE"]="sv_SE" ["mn"]="mn_MN" ["zh-CN"]="zh_CN" ["sl"]="sl_SI" ["et"]="et_EE" ["id"]="id_ID" ["ar"]="ar_AR" ["ta"]="ta_IN" ["lv"]="lv_LV" ["ja"]="ja_XX")

declare -A TOKENIZERS
export TOKENIZERS=(["ja"]="char" ["zh-CN"]="char")

export FLEURS_LANG_CODES=(amh_Ethi arb_Arab asm_Beng azj_Latn bel_Cyrl ben_Beng bos_Latn bul_Cyrl cat_Latn ceb_Latn ces_Latn ckb_Arab zho_Hans cym_Latn dan_Latn deu_Latn ell_Grek est_Latn fin_Latn fra_Latn fuv_Latn gaz_Latn gle_Latn glg_Latn guj_Gujr heb_Hebr hin_Deva hrv_Latn hun_Latn hye_Armn ibo_Latn ind_Latn isl_Latn ita_Latn jav_Latn jpn_Jpan kan_Knda kat_Geor kaz_Cyrl khk_Cyrl khm_Khmr kir_Cyrl kor_Hang lao_Laoo lit_Latn lug_Latn luo_Latn lvs_Latn mal_Mlym mar_Deva mkd_Cyrl mlt_Latn mya_Mymr nld_Latn nob_Latn npi_Deva nya_Latn ory_Orya pan_Guru pbt_Arab pes_Arab pol_Latn por_Latn ron_Latn rus_Cyrl slk_Latn slv_Latn sna_Latn snd_Arab som_Latn spa_Latn srp_Cyrl swe_Latn swh_Latn tam_Taml tel_Telu tgk_Cyrl tgl_Latn tha_Thai tur_Latn ukr_Cyrl urd_Arab uzn_Latn vie_Latn yor_Latn zho_Hant zsm_Latn zul_Latn)
