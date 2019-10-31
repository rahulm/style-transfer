import argparse
import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf


def readArgs():
  parser = argparse.ArgumentParser(
    description = "Style Transfer"
  )
  
  parser.add_argument(
    "--content",
    required = True, type = str,
    help = "Content image."
  )
  
  parser.add_argument(
    "--style",
    required = True, type = str,
    help = "Style image."
  )
  
  parser.add_argument(
    "--outdir",
    required = True, type = str,
    help = "Output directory for generated images."
  )
  
  parser.add_argument(
    "--iters",
    required = True, type = int,
    help = "Number of iterations to run."
  )
  
  parser.add_argument(
    "--interval",
    required = False, type = int, default = 10,
    help = "Iteration interval when image checkpoints should be saved.\
            Ex: a value of 1 saves an image at every iteration."
  )
  
  args = parser.parse_args()
  
  # make outdir
  if not os.path.isdir(args.outdir):
    if os.path.isfile(args.outdir):
      parser.error("{} is a file. --outdir cannot be an existing file.".format(args.outdir))
    else:
      os.makedirs(args.outdir)
  
  return args
  

def loadImage(imgPath):
  img = tf.io.read_file(imgPath)
  img = tf.image.decode_image(img, channels = 3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  return img

def saveImage(img, path):
  imgNp = (img * 255).numpy().astype(np.uint8)
  Image.fromarray(imgNp).save(path)


def gramMatrix(input):
  res = tf.linalg.einsum("bijc,bijd->bcd", input, input)
  inshape = tf.shape(input)
  numLocs = tf.cast(inshape[1] * inshape[2], tf.float32)
  return res / numLocs


# losses
def customLoss(target, gen):
  return tf.reduce_mean(
    [
      tf.reduce_mean((gen[layerName] - target[layerName]) ** 2)
      for layerName in gen.keys()
    ],
    0
  )
def totalLoss(targetValues, genOutputs, lossWeights):
  loss = (lossWeights["content"] * customLoss(targetValues["content"], genOutputs["content"]))
  loss += (lossWeights["style"] * customLoss(targetValues["style"], genOutputs["style"]))
  return loss


class StyleTransferModel(tf.keras.Model):
  def __init__(self, contentLayers, styleLayers):
    super(StyleTransferModel, self).__init__()
    
    self.customLayers = {
      "content" : contentLayers,
      "style" : styleLayers
    }
    
    # create custom model that returns appropriate layers
    self.customModel = tf.keras.applications.VGG19(include_top = False, weights = "imagenet")
    self.customModel = tf.keras.Model(
      [self.customModel.input],
      {
        customName : {
          layerName : self.customModel.get_layer(layerName).output
          for layerName in self.customLayers[customName]
        }
        for customName in self.customLayers.keys()
      }
    )
    self.customModel.trainable = False
  
  def call(self, inputs):
    inputs = inputs * 255.0
    inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
    
    outputs = self.customModel(inputs)
    
    # apply gram matrix to styles
    outputs["style"] = {
      styleLayerName : gramMatrix(outputs["style"][styleLayerName])
      for styleLayerName in self.customLayers["style"]
    }
    
    return outputs


def clip01(img):
  return tf.clip_by_value(img, clip_value_min = 0.0, clip_value_max = 1.0)

@tf.function()
def genStep(model, opt, targets, weights, img):
  with tf.GradientTape() as tape:
    outputs = model(img)
    loss = totalLoss(targets, outputs, weights)
  grad = tape.gradient(loss, img)
  opt.apply_gradients([(grad, img)])
  img.assign(clip01(img))

def run():
  print("Style Transfer")
  
  # parse arguments
  args = readArgs()
  outDir = args.outdir
  numIters = args.iters
  saveInterval = args.interval
  
  # load images
  contentImg, styleImg = loadImage(args.content), loadImage(args.style)
  saveImage(contentImg, os.path.join(outDir, "_content.png"))
  saveImage(styleImg, os.path.join(outDir, "_style.png"))
  
  # set content layers and style layers for VGG19
  contentLayers = ["block5_conv2"]
  styleLayers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1"
  ]
  
  # create model
  model = StyleTransferModel(contentLayers, styleLayers)
  opt = tf.optimizers.Adam(learning_rate = 0.02, beta_1 = 0.99, epsilon = 1e-1)
  
  # get the initial style and content layer values
  targets = {
    "content" : model(tf.expand_dims(contentImg, 0))["content"],
    "style" : model(tf.expand_dims(styleImg, 0))["style"]
  }
  weights = {
    "content" : 1e4,
    "style" : 1e-2
  }
  
  print(">>> STARTING GENERATION")
  
  # define the image to generate, starting with the content image
  genImage = tf.Variable(tf.expand_dims(contentImg, 0))
  
  
  startTime = time.time()
  
  # generate image
  numDigits = np.log10(numIters) + 1
  formatString = "gen-{:0" + str(int(numDigits)) + "d}.png"
  for iter in range(numIters):
    print("Iter: {}/{}".format(iter, numIters))
    genStep(model, opt, targets, weights, genImage)
    if ((iter % saveInterval) == 0):
      saveImage(genImage[0], os.path.join(outDir, formatString.format(iter)))
  
  # save final image
  saveImage(genImage[0], os.path.join(outDir, "_generated.png"))
  
  endTime = time.time()
  print("Total gen time (s): {:.1f}".format(endTime - startTime))
  


if __name__ == "__main__":
  run()

