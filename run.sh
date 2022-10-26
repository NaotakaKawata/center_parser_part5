python run_race.py \
--data_dir=data/RACE \
--bert_model=bert-base-uncased \
--output_dir=base_models \
--max_seq_length=380  \
--do_train \
--do_eval \
--do_lower_case \
--train_batch_size=8 \
--eval_batch_size=4 \
--learning_rate=5e-5 \
--num_train_epochs=3 \
--gradient_accumulation_steps=8 \
--loss_scale=128


