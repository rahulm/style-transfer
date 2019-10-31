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
  
  parser.add_argument(
    "--model",
    required = False, type = str, default = "vgg19",
    choices = ("vgg19", "vgg16"),
    help = "Pretrained model to use during generation. Select from: %(choices)s"
  )
  
  parser.add_argument(
    "--csratio",
    required = False, type = float, default = None,
    help = "The ratio of loss weights, =content/style"
  )
  
  parser.add_argument(
    "--varloss",
    required = False, type = float, default = None,
    help = "The weight to place on the total variation loss (on high frequency artifacts)."
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

def writeBlockToFile(file, block):
  file.write("--- {} ---\n".format(block["title"]))
  for lineFormat, lineArgs in block["lines"]:
    file.write(lineFormat.format(lineFormat, **lineArgs))
  file.write("--- ---\n\n")

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
  def __init__(self, modelInfo):
    super(StyleTransferModel, self).__init__()
    
    self.modelInfo = modelInfo
    
    # create custom model that returns appropriate layers
    if self.modelInfo["name"] == "vgg16":
      self.customModel = tf.keras.applications.VGG16(include_top = False, weights = "imagenet")
    elif self.modelInfo["name"] == "vgg19":
      self.customModel = tf.keras.applications.VGG19(include_top = False, weights = "imagenet")
    else:
      print("Model {} not supported.".format(self.modelInfo["name"]))
      exit(1)
    
    self.customModel = tf.keras.Model(
      [self.customModel.input],
      {
        customName : {
          layerName : self.customModel.get_layer(layerName).output
          for layerName in self.modelInfo["layers"][customName]
        }
        for customName in self.modelInfo["layers"].keys()
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
      for styleLayerName in self.modelInfo["layers"]["style"]
    }
    
    return outputs


def clip01(img):
  return tf.clip_by_value(img, clip_value_min = 0.0, clip_value_max = 1.0)

@tf.function()
def genStep(model, opt, weights, targets, img):
  loss = 0.0
  with tf.GradientTape() as tape:
    outputs = model(img)
    loss = totalLoss(targets, outputs, weights)
    loss += (weights["variation"] * tf.image.total_variation(img))
  grad = tape.gradient(loss, img)
  opt.apply_gradients([(grad, img)])
  img.assign(clip01(img))
  return loss

def run():
  print("Style Transfer")
  
  # parse arguments
  args = readArgs()
  outDir = args.outdir
  numIters = args.iters
  saveInterval = args.interval
  
  ALLMODELS = {
    "vgg19" : {
      "name" : "vgg19",
      "layers" : {
        "content" : ["block5_conv2"],
        "style" : [
          "block1_conv1",
          "block2_conv1",
          "block3_conv1",
          "block4_conv1",
          "block5_conv1"
        ]
      },
      "weights" : {
        "content" : 0,
        "style" : 1,
        "variation" : 0
      }
      
      # "weights" : {
        # "content" : 1e4,
        # "style" : 1e-2,
        # "variation" : 30
      # }
    },
    
    "vgg16" : {
      "name" : "vgg16",
      "layers" : {
        "content" : ["block2_conv2"],
        "style" : [
          "block1_conv2",
          "block2_conv2",
          "block3_conv3",
          "block4_conv3",
          "block5_conv3"
        ]
      },
      "weights" : {
        "content" : 0,
        "style" : 1,
        "variation" : 0
      }
      
      # "weights" : {
        # "content" : 0.02,
        # "style" : 4.5,
        # "variation" : 1e-5
      # }
    }
  }
  
  modelInfo = ALLMODELS[args.model]
  if args.csratio is not None:
    modelInfo["weights"]["content"] = float(args.csratio) / float(1 + args.csratio)
    modelInfo["weights"]["style"] = (1 - modelInfo["weights"]["content"])
  if args.varloss is not None:
    modelInfo["weights"]["variation"] = float(args.varloss)
  
  # create model
  model = StyleTransferModel(modelInfo)
  opt = tf.optimizers.Adam(learning_rate = 0.02, beta_1 = 0.99, epsilon = 1e-1)
  # opt = tf.optimizers.Adam(learning_rate = 1e-3, beta_1 = 0.99, epsilon = 1e-1)
  
  # save arguments, hyper-parameters, etc.
  with open(os.path.join(outDir, "_args.txt"), "w") as argFile:
    argFile.write("Style Transfer\n\n")
    argFile.write(str(args) + "\n\n")
    blocks = []
    
    # image information
    b = {
      "title" : "Image Information",
      "lines" : [
        ("{name: <16} | {val}\n", {"name" : "content image", "val" : args.content}),
        ("{name: <16} | {val}\n", {"name" : "style image", "val" : args.style}),
        ("{name: <16} | {val}\n", {"name" : "output directory", "val" : outDir}),
      ]
    }
    blocks.append(b)
    
    b = {
      "title" : "Model Information",
      "lines" : [
        ["", {"name" : "model name", "val" : str(modelInfo["name"])}],
        ["", {"name" : "content layers", "val" : list(modelInfo["layers"]["content"]) }],
        ["", {"name" : "style layers", "val" : list(modelInfo["layers"]["style"]) }],
        ["", {"name" : "weights", "val" : modelInfo["weights"] }],
      ]
    }
    maxLen = max(len(bargs["name"]) for _, bargs in b["lines"])
    formatString = "{name: <" + str(maxLen) + "} | {val}\n"
    for bline in b["lines"]:
      bline[0] = formatString
    blocks.append(b)
    
    b = {
      "title" : "Generation Information",
      "lines" : [
        ["", {"name" : "iterations", "val" : numIters}],
        ["", {"name" : "save interval", "val" : saveInterval}],
        ["", {"name" : "optimizer", "val" : "{}: {}".format(opt._name, opt._hyper)}],
      ]
    }
    maxLen = max(len(bargs["name"]) for _, bargs in b["lines"])
    formatString = "{name: <" + str(maxLen) + "} | {val}\n"
    for bline in b["lines"]:
      bline[0] = formatString
    blocks.append(b)
    
    for block in blocks:
      writeBlockToFile(argFile, block)
    argFile.flush()
  
  # load images
  contentImg, styleImg = loadImage(args.content), loadImage(args.style)
  saveImage(contentImg, os.path.join(outDir, "_content.png"))
  saveImage(styleImg, os.path.join(outDir, "_style.png"))  
  
  # get the initial style and content layer values
  targets = {
    "content" : model(tf.expand_dims(contentImg, 0))["content"],
    "style" : model(tf.expand_dims(styleImg, 0))["style"]
  }
  
  print(">>> STARTING GENERATION")
  
  # define the image to generate, starting with the content image
  genImage = tf.Variable(tf.expand_dims(contentImg, 0))
  
  
  startTime = time.time()
  
  # generate image
  numDigits = np.log10(numIters) + 1
  fileFormatString = "gen-{:0" + str(int(numDigits)) + "d}.png"
  numCharsInPrefix = (2 * numDigits) + 1
  prefixFormatString = "Iter {0: <" + str(numCharsInPrefix) + "}: "
  for iter in range(numIters):
    # print("Iter {}/{}:".format(iter + 1, numIters), end = "")
    print(prefixFormatString.format("{}/{}".format(iter + 1, numIters)), end = "")
    loss = genStep(model, opt, modelInfo["weights"], targets, genImage)
    if ((iter % saveInterval) == 0):
      saveImage(genImage[0], os.path.join(outDir, fileFormatString.format(iter + 1)))
    print(str(loss.numpy()[0]))
  
  # save final image
  saveImage(genImage[0], os.path.join(outDir, "_generated.png"))
  
  endTime = time.time()
  print("Total gen time (s): {:.1f}".format(endTime - startTime))
  
  with open(os.path.join(outDir, "_args.txt"), "a") as argFile:
    argFile.write("\n\nTotal Generation Time (s): {:.1f}\n\n".format(endTime - startTime))
    argFile.flush()


if __name__ == "__main__":
  run()

