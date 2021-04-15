# --train-json dump/train/deltafalse/data.json --valid-json dump/dev/deltafalse/data.json --dict data/lang_1char/train_chars.txt --LFR_m 4 --LFR_n 3 --d_input 40 --n_layers_enc 6 --n_head 8 --d_k 64 --d_v 64 --d_model 512 --d_inner 1024 --dropout 0.1 --pe_maxlen 5000 --d_word_vec 512 --n_layers_dec 6 --tgt_emb_prj_weight_sharing 1 --label_smoothing 0.1 --epochs 10 --shuffle 1 --batch-size 8 --batch_frames 7500 --maxlen-in 800 --maxlen-out 150 --k 0.2 --warmup_steps 4000 --save-folder exp/train_m4_n3_in40_elayer6_head8_k64_v64_model512_inner1024_drop0.1_pe5000_emb512_dlayer6_share1_ls0.1_epoch10_shuffle1_bs8_bf7500_mli800_mlo150_k0.2_warm4000 --checkpoint 0 --continue-from "" --print-freq 10 --visdom 0 --visdom_lr 0 --visdom_epoch 0 --visdom-id "Transformer Training" 
# Started at Mon Apr  5 21:41:07 UTC 2021
#
bash: line 1: --train-json: command not found
# Accounting: time=0 threads=1
# Ended (code 127) at Mon Apr  5 21:41:07 UTC 2021, elapsed time 0 seconds
