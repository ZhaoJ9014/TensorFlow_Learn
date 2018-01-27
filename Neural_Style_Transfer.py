from keras.preprocessing import image
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

target_image_path = './portrait.jpg'
style_reference_image_path = './transfer_style_reference.jpg'
width, height = image.load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size = (img_height, img_width, 3))
    img = image.img_to_array(img)
    img = img[np.newaxis, :, :, :]
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def content_loss(base, combination):
    return K.sum(K.square(base - combination))

def gram_matrix(x):
    features= K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S-C)) / (4. * (channels ** 2) * (size ** 2))

def total_varitation_loss(x):
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    return K.sum(K.pow(a + b, 1.25))

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
combination_image = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis = 0)
model = vgg19.VGG19(input_tensor = input_tensor, weights = 'imagenet', include_top = False)
print('model loaded.')

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layer = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]
total_varitation_weight = 1e-4
style_weight = 1.
content_weight = 5e-5
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_features)
for layer_name in style_layer:
    layer_features = outputs_dict[layer_name]
    style_reference_image_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += (style_weight / len(style_layer)) * style_loss(style_reference_image_features, combination_features)
loss += total_varitation_weight * total_varitation_loss(combination_image)

grads = K.gradients(loss, combination_image)[0]
fetch_loss_and_grads = K.function([combination_image], [loss, grads])
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grad_value = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_value = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_value = grad_value
        return self.loss_value

    def grads(self, x):
        assert self.grad_value is not None
        grad_value = np.copy(self.grad_value)
        self.loss_value = None
        self.grad_value = None
        return grad_value
evaluator = Evaluator()
result_prefix = 'my_result'
iterations = 20
x = preprocess_image(target_image_path)
x = x.flatten()
for i in range(iterations):
    print('start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime = evaluator.grads, maxfun = 20)
    print('current loss value:', min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    print('image saved as', fname)
    end_time = time.time()
    print('iteration %d completed in %ds' % (i, end_time - start_time))
