import arraymancer

proc load_preprocess_mnist*(): tuple[xTrain, yTrain, xTest, yTest: Tensor[float32]] =
  var xTrain = read_mnist_images("train-images-idx3-ubyte.gz").asType(float32) / 255
  var xTest = read_mnist_images("t10k-images-idx3-ubyte.gz").asType(float32) / 255
  let yTrain = read_mnist_labels("train-labels-idx1-ubyte.gz")
  let yTest = read_mnist_labels("t10k-labels-idx1-ubyte.gz")

  # One-hot encode
  var yTrain_onehot = zeros[float32](yTrain.shape[0], 10)
  for i in 0 ..< yTrain.shape[0]:
    let class = yTrain[i].int
    yTrain_onehot[i, class] = 1
  var yTest_onehot = zeros[float32](yTest.shape[0], 10)
  for i in 0 ..< yTest.shape[0]:
    let class = yTest[i].int
    yTest_onehot[i, class] = 1

  # Normalize
  let mean = xTrain.mean(axis=0)
  var std = xTrain.std(axis=0)
  let zero = 0'f32
  std[std ==. zero] = 1
  xTrain = (xTrain -. mean) /. std 
  xTest = (xTest -. mean) /. std

  result = (xTrain: xTrain.reshape(xTrain.shape[0], 28*28), yTrain: yTrain_onehot, xTest: xTest.reshape(xTest.shape[0], 28*28), yTest: yTest_onehot)
  