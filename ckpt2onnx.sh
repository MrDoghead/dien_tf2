python -m tf2onnx.convert --checkpoint ./dnn_best_model/512/savemodel_noshuffDIEN3.meta --output onnx/dien_512.onnx --inputs Inputs/mid_his_batch_ph:0,Inputs/cat_his_batch_ph:0,Inputs/uid_batch_ph:0,Inputs/mid_batch_ph:0,Inputs/cat_batch_ph:0,Inputs/mask:0,Inputs/seq_len_ph:0,Inputs/target_ph:0,Inputs/Placeholder:0,Inputs/noclk_mid_batch_ph:0,Inputs/noclk_cat_batch_ph:0 --outputs dien/fcn/add_6:0 --opset 11
