import tensorflow as tf
print(tf.__version__)
import numpy as np
from tensorflow.python.saved_model import tag_constants
import pdb

bs = 512
sl = 827
# pb_path = "./dien_frozen_fix.pb"
model_path = "./dien_frozen"
    
"""
#导入pb文件到graph中
with tf.gfile.GFile(pb_path,'rb') as f:
    # 复制定义好的计算图到新的图中，先创建一个空的图.
    graph_def = tf.GraphDef()
    # 加载proto-buf中的模型
    graph_def.ParseFromString(f.read())
    # 最后复制pre-def图的到默认图中.
    _ = tf.import_graph_def(graph_def, name='')
"""
with tf.Session() as sess:
    meta_graph_def = tf.saved_model.loader.load(sess,[tag_constants.SERVING],model_path)

    uid_batch_ph = tf.get_default_graph().get_tensor_by_name("Inputs/uid_batch_ph:0")
    mid_batch_ph = tf.get_default_graph().get_tensor_by_name("Inputs/mid_batch_ph:0")
    cat_batch_ph = tf.get_default_graph().get_tensor_by_name("Inputs/cat_batch_ph:0")
    mid_his_batch_ph = tf.get_default_graph().get_tensor_by_name("Inputs/mid_his_batch_ph:0")
    cat_his_batch_ph = tf.get_default_graph().get_tensor_by_name("Inputs/cat_his_batch_ph:0")
    mask_ph = tf.get_default_graph().get_tensor_by_name("Inputs/mask:0")
    sl_ph = tf.get_default_graph().get_tensor_by_name("Inputs/seq_len_ph:0")
    pred = tf.get_default_graph().get_tensor_by_name("dien/fcn/add_6:0")
    # pred = tf.get_default_graph().get_tensor_by_name("dien/rnn_1/gru1/while/gru_cell/MatMul:0")

    # fake inputs
    uids = np.array([0 for _ in range(bs)])
    mids = np.array([0 for _ in range(bs)])
    cats = np.array([0 for _ in range(bs)])
    sls = np.array([sl for _ in range(bs)])
    mid_his = np.zeros((bs, sl)).astype('int64')
    cat_his = np.zeros((bs, sl)).astype('int64')
    mid_mask = np.ones((bs, sl)).astype('int64')

    # op info
    # graph = sess.graph
    # for op in graph.get_operations():
        # print(op.name, op.values)
    # exit(0)

    # inference
    feed_dict={
        uid_batch_ph:uids, 
        mid_batch_ph:mids, 
        cat_batch_ph:cats, 
        mid_his_batch_ph:mid_his, 
        cat_his_batch_ph:cat_his, 
        mask_ph:mid_mask,
        sl_ph:sls
        }
    out = sess.run(pred,feed_dict)
    print("out: ", out, out.shape)

