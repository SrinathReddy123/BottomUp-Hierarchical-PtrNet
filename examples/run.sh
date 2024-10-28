#!/usr/bin/env bash

LOGDIR="$(pwd)/logs"
LOGPATH="$LOGDIR/log_file.log"

# ensure the logs directory exists
mkdir -p "$LOGDIR"

MODELNAME="HPtrNetPSTGate"
echo "running model $MODELNAME"
echo "log saved to $LOGPATH"

# Disable GPU usage by setting CUDA_VISIBLE_DEVICES to an empty value
CUDA_VISIBLE_DEVICES=0 python examples/HL2RParser.py --mode FastLSTM --num_epochs 50 --batch_size 4 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
 --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
 --schedule 20 --double_schedule_decay 5 \
 --p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --char --beam 10 --prior_order inside_out \
 --word_embedding word2vec --word_path "embs/GoogleNews-vectors-negative300.bin" --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train "data/n_ewt-ud-train.conllu" \
 --dev "data/n_ewt-ud-dev.conllu" \
 --test "data/n_ewt-ud-test.conllu" \
 --model_path "models/" --model_name 'network.pt' \
 --mymodel "$MODELNAME" --proj --sgate > "$LOGPATH"
