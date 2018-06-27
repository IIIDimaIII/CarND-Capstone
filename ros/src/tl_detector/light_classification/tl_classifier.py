import tensorflow as tf
import numpy as np
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        #DONE load classifier
        tf.reset_default_graph()
        graph_file = 'frozen_model.pb'
        jit_level = 0
        config = tf.ConfigProto()
        use_xla = False
        if use_xla:
            jit_level = tf.OptimizerOptions.ON_1
            config.graph_options.optimizer_options.global_jit_level = jit_level

        sess = tf.Session(config=config)
        gd = tf.GraphDef()
        #---------------------remove---------------------------
        with open('classifier_saved_here.txt','w') as f:
            f.write("ok")
        #---------------------remove---------------------------
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')        
        self.image_input = sess.graph.get_tensor_by_name("raw_rgb_image:0")
        self.is_training = sess.graph.get_tensor_by_name("is_training:0")    
        self.pred = sess.graph.get_tensor_by_name("logits:0")                
        self.tf_sess = sess 
        self.class_to_tl = {
                            0: TrafficLight.RED,
                            1: TrafficLight.YELLOW,
                            2: TrafficLight.GREEN,
                            3: TrafficLight.UNKNOWN,
                            }

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        im_4d = np.expand_dims(image, axis=0)
        predict = self.tf_sess.run(self.pred, feed_dict = {self.image_input: im_4d,
                                                           self.is_training:False})
        
        predicted_class = np.argmax(predict[0])
        return self.class_to_tl[predicted_class]