python eval.py \
--data_dir=RACE \
--bert_model=bert-base-uncased \
--output_dir=base_models \
--question_file=data/center-2017-2015/dev/Center-2017--Main-Eigo_hikki.xml \
--answer_file=data/center-2017-2015/answer/Center-2017--Main-Eigo_hikki_part5.json \
--max_seq_length=380 \
--do_eval \
--do_lower_case \
--train_batch_size=1 \
--eval_batch_size=1 \
--learning_rate=5e-5 \
--num_train_epochs=3 \
--gradient_accumulation_steps=8 \
--loss_scale=128