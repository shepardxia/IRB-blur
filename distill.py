import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import model_from_json
from src.loss import iwpodnet_loss, iwpodnet_distill_loss
from src.data_generator_tf2 import ALPRDataGenerator
from src.utils import image_files_from_folder
from src.label import readShapes, Shape
from os.path import isfile, isdir, splitext
import cv2
from create_model_iwpodnet import create_lite_iwpodnet, create_model_iwpodnet


class Distiller(keras.Model):
    
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, 
        optimizer, 
        metrics, 
        student_loss_fn, 
        distillation_loss_fn, 
        alpha=0.1, 
        temperature=3):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
    
    def train_step(self, data):
        x, y = data

        teacher_pred = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)

            student_loss = self.student_loss_fn(y, student_pred)
            print(teacher_pred.shape)
            print(student_pred.shape)
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_pred / self.temperature, axis=1),
                    tf.nn.softmax(student_pred / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, student_pred)

        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )

        return results


    def test_step(self, data):
        x, y = data

        y_pred = self.student(x, training=False)

        student_loss = self.student_loss_fn(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

    def call(self, inputs, *args, **kwargs):
        return self.student(inputs)

    



if __name__ == "__main__":

    # load pre_trained teacher model
    with open('./weights/teacher_5000.json') as json_file:
        model_json = json_file.read()
    teacher = model_from_json(model_json, custom_objects={})
    teacher.load_weights('./weights/teacher_5000.h5')

    opt = keras.optimizers.Adam(learning_rate=0.001)

    teacher.compile(
	    loss = iwpodnet_loss,
	    optimizer = opt,
	    )

    # create student model
    student = create_lite_iwpodnet()
    student.compile(
	    loss = iwpodnet_loss,
	    optimizer = opt,
	    )


    # load dataset for student training
    files = image_files_from_folder('./data_1')

    fakepts = np.array([[0.5, 0.5001, 0.5001, 0.5], [0.5, 0.5, 0.5001, 0.5001]])
    fakeshape = Shape(fakepts)
    Data = []
    ann_files = 0
    for file in files:
        labfile = splitext(file)[0] + '.txt'
        if isfile(labfile):
            ann_files += 1
            L = readShapes(labfile)
            I = cv2.imread(file)
            if len(L) > 0:
                Data.append([I, L])
        else:
        	#
        	#  Appends a "fake"  plate to images without any annotation
        	#
            I = cv2.imread(file)
            Data.append(  [I, [fakeshape] ]  )

    print ('%d images with labels found' % len(Data) )
    print ('%d annotation files found' % ann_files )

    train_generator = ALPRDataGenerator(
        Data,
        batch_size=64,
        dim=208,
        stride=16,
        shuffle=True,
        OutputScale=1.0
    )

    distiller = Distiller(student, teacher)


    distiller.compile(
        optimizer=opt, 
        metrics=None, 
        student_loss_fn=iwpodnet_loss, 
        distillation_loss_fn=iwpodnet_distill_loss, 
        alpha=0.1, 
        temperature=3
    )

    

    distiller.fit(train_generator, epochs=3)

    #distiller.evaluate(train_generator)


    print(teacher.summary())
