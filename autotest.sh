if [ -n "$2" ]; then
    gpu_device=$2
else
    gpu_device=-1
fi

if [ "$1"x = "Y"x ]; then
    for path in '--conf_path=autotest/conf/conf_text_classification_bilstm_attn_autotest.json --force=True' '--conf_path=autotest/conf/conf_chinese_text_matching_emb_char_autotest.json --force=True' '--conf_path=autotest/conf/conf_question_pairs_bilstm_attn_autotest.json --force=True' '--conf_path=autotest/conf/conf_kdtm_match_linearAttn_autotest.json --force=True'
    do
        (
         CUDA_VISIBLE_DEVICES=$gpu_device python train.py $path
         CUDA_VISIBLE_DEVICES=-1 python test.py $path
        )&
    done
    wait
else
   CUDA_VISIBLE_DEVICES=$gpu_device python train.py --conf_path=autotest/conf/conf_text_classification_bilstm_attn_autotest.json --force=True
   CUDA_VISIBLE_DEVICES=-1 python test.py --conf_path=autotest/conf/conf_text_classification_bilstm_attn_autotest.json --force=True
   CUDA_VISIBLE_DEVICES=$gpu_device python train.py --conf_path=autotest/conf/conf_chinese_text_matching_emb_char_autotest.json --force=True
   CUDA_VISIBLE_DEVICES=-1 python test.py --conf_path=autotest/conf/conf_chinese_text_matching_emb_char_autotest.json --force=True
   CUDA_VISIBLE_DEVICES=$gpu_device python train.py --conf_path=autotest/conf/conf_question_pairs_bilstm_attn_autotest.json --force=True
   CUDA_VISIBLE_DEVICES=-1 python test.py --conf_path=autotest/conf/conf_question_pairs_bilstm_attn_autotest.json --force=True
   CUDA_VISIBLE_DEVICES=$gpu_device python train.py --conf_path=autotest/conf/conf_kdtm_match_linearAttn_autotest.json --force=True
   CUDA_VISIBLE_DEVICES=-1 python test.py --conf_path=autotest/conf/conf_kdtm_match_linearAttn_autotest.json --force=True
fi
python ./autotest/tools/get_results.py
