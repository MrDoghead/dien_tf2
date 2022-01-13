import tensorflow as tf

def run():
    with tf.Graph().as_default():
        output_graph_path = './dien_frozen/saved_model.pb'
     
        with open(output_graph_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name="")
     
        with tf.Session() as sess:
            # tf.global_variables_initializer()
            test = sess.graph.get_tensor_by_name("Inputs/mid_his_batch_ph:0")
            print(test)

def run2():
    saved_model_dir = "./dien_frozen"
    sess = tf.InteractiveSession()
    meta_graph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
    print(meta_graph_def)

def run3():
    saved_path = "./dien_frozen"
    with tf.Session() as sess:
        loaded = tf.saved_model.load(sess,export_dir=saved_path,tags=[tf.saved_model.tag_constants.SERVING])
        print(list(loaded.signatures.keys()))

if __name__=="__main__":
    run3()
