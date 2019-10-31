import argparse
import os

# import tensorflow as tf

# print("Testing tensorflow.")

# mnist = tf.keras.datasets.mnist

# (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
# xTrain, xTest = xTrain / 255.0, xTest / 255.0

# model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten(input_shape=(28, 28)),
  # tf.keras.layers.Dense(128, activation='relu'),
  # tf.keras.layers.Dropout(0.2),
  # tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
              # loss='sparse_categorical_crossentropy',
              # metrics=['accuracy'])


# model.fit(xTrain, yTrain, epochs=5)
# model.evaluate(xTest,  yTest, verbose=2)

# print("Done")

# maybe try ResNet-18 or ResNet-34 because they are smaller in size than the VGGs? (due to hardware limitations)


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
    "--iterations",
    required = True, type = int,
    help = "Number of iterations to run."
  )
  
  args = parser.parse_args()
  
  # make outdir
  if not os.path.isdir(args.outdir):
    if os.path.isfile(args.outdir):
      parser.error("{} is a file. --outdir cannot be an existing file.".format(args.outdir))
    else:
      os.makedirs(args.outdir)
  
  return args
  

def readImage(imgPath):
  return None


def run():
  print("Style Transfer")
  
  args = readArgs()
  print(args)
  contentImg, styleImg = readImage(args.content), readImage(args.style)
  
  
  print("Done")
  


if __name__ == "__main__":
  run()

