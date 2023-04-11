import std/[random, tables, math, sequtils, strformat]
import arraymancer except Optimizer
import benchy
import ./models, ./mnist

# Input is of the form: (nBatch, nFeatures)

type
  Linear* = ref object of Layer
    x*: Tensor[float32]
  ReLU* = ref object of Layer
    mask*: Tensor[float32]
  SoftmaxCrossEntropy* = ref object of LossFunction
    soft*, yTrue*: Tensor[float32]
  MeanSquareError* = ref object of LossFunction
    diff*: Tensor[float32]

proc `$`*(lay: Layer): string =
  $lay[]

method forward*(lay: Layer, x: Tensor[float32]): Tensor[float32] {.base.} = doAssert false, "All layers need to implement `forward` method"
method backward*(lay: Layer, deriv: Tensor[float32]): Tensor[float32] {.base.} = doAssert false, "All layers need to implement `backward` method"

# Layers

method forward*(lay: Linear, x: Tensor[float32]): Tensor[float32] =
  result = x * lay.weights["w"] +. lay.weights["b"]
  lay.x = x

method backward*(lay: Linear, deriv: Tensor[float32]): Tensor[float32] =
  lay.gradients["w"] += (lay.x.transpose * deriv)
  lay.gradients["b"] += deriv.sum(axis=0)
  result = deriv * lay.weights["w"].transpose

proc newLinear*(inputSize, outputSize: int): Linear =
  new result
  result.weights["w"] = randomNormalTensor[float32]([inputSize, outputSize], 0, 1)
  result.weights["b"] = randomNormalTensor[float32]([1, outputSize], 0, 1)
  result.gradients["w"] = zeros_like(result.weights["w"])
  result.gradients["b"] = zeros_like(result.weights["b"])


method forward*(lay: ReLU, x: Tensor[float32]): Tensor[float32] =
  result = relu(x)
  lay.mask = (result !=. 0).asType(float32)

method backward*(lay: ReLU, deriv: Tensor[float32]): Tensor[float32] =
  result = lay.mask *. deriv

# Loss functions

method forward*(loss: LossFunction, yPred, yTrue: Tensor[float32]): float32 {.base.} = doAssert false, "All losses need to implement `forward` method"
method backward*(loss: LossFunction): Tensor[float32] {.base.} =
  ## Return the derivative of the loss w.r.t yPred
  doAssert false, "All losses need to implement `backward` method"

method forward*(loss: MeanSquareError, yPred, yTrue: Tensor[float32]): float32 =
  result = mean_squared_error(yPred, yTrue)
  loss.diff = yPred - yTrue

method backward*(loss: MeanSquareError): Tensor[float32] =
  result = 2 / loss.diff.size.float32 * loss.diff

method forward*(loss: SoftmaxCrossEntropy, yPred, yTrue: Tensor[float32]): float32 =
  let soft = softmax(yPred)
  result = softmax_cross_entropy(yPred, yTrue)
  loss.soft = soft
  loss.yTrue = yTrue

method backward*(loss: SoftmaxCrossEntropy): Tensor[float32] =
  result = (loss.soft - loss.yTrue) / loss.soft.shape[0].float32


# Model

proc forward*(model: Model, x: Tensor[float32]): Tensor[float32] =
  result = x
  for lay in model.layers:
    result = lay.forward(result)

proc backward*(model: Model) =
  var deriv = model.loss.backward()
  for i in countdown(model.layers.high, 0):
    deriv = model.layers[i].backward(deriv)

proc zeroGrad*(model: Model) =
  for lay in model.layers:
    for w in lay.gradients.keys:
      lay.gradients[w] *= 0

proc train*(model: Model, x, y, xVal, yVal: Tensor[float32], optimizer: Optimizer, epochs: int, batchSize: int) =
  let nX = x.shape[0]
  let nBatches = nX div batchSize
  for epoch in 0 ..< epochs:
    var train_loss: float32
    for iter in 0 ..< nBatches:
      let batchX = x[iter*batchSize ..< min(nX, (iter+1)*batchSize), _]
      let batchY = y[iter*batchSize ..< min(nX, (iter+1)*batchSize), _]
      #echo batchX.shape, batchY.shape
      let pred = model.forward(batchX)
      #echo pred.shape
      let loss = model.loss.forward(pred, batchY)
      train_loss += loss
      model.backward()
      optimizer.update()
      model.zeroGrad()
    let predVal = model.forward(xVal)
    let lossVal = model.loss.forward(predVal, yVal)
    let predInd = predVal.argmax(axis=1)
    let trueInd = yVal.argmax(axis=1)
    let accuracy = mean((predInd ==. trueInd).asType(float32))
    echo &"Epoch {epoch}: val loss = {lossVal:.2f}, val acc = {accuracy:.2f}, train loss = {train_loss / nBatches.float32:.2f}"

let hidden = 2000
var model = Model()
model.loss = SoftmaxCrossEntropy()
model.layers.add newLinear(28*28, hidden)
model.layers.add ReLU()
model.layers.add newLinear(hidden, hidden)
model.layers.add ReLU()
model.layers.add newLinear(hidden, hidden)
model.layers.add ReLU()
model.layers.add newLinear(hidden, 10)


let optimizer = newSGD(model, 1e-5, reg=1e-3)

#[ let N = 10000
let x = randomTensor[float32]([N, 2], -50'f32..50'f32) + randomNormalTensor[float32]([N, 2], 0, 0.1)

#[ let w1 = 3.14'f32
let w2 = 2.71'f32
let b = 1.42'f32
let y = w1 * x[_, 0] + w2 * x[_, 1] +. b + sin(x[_, 0]) ]#
var y = zeros[float32](N, 5)
for i in 0 ..< N:
  y[i, rand(0..4)] = 1

echo "Before:"
echo model.layers[0].weights["w"]
echo model.layers[0].weights["b"] ]#

let (xTrain, yTrain, xTest, yTest) = load_preprocess_mnist()

#timeIt "Train 1 epoch":
model.train(xTrain, yTrain, xTest, yTest, optimizer, epochs=3, batchSize=64)

echo "After:"
echo model.layers[0].weights["w"].mean()
#echo model.layers[0].weights["b"]

#[ let lay = newLinear(100, 2000)
let lay2 = newLinear(2000, 2000)
timeIt "Dense":
  keep lay2.forward(lay.forward(randomNormalTensor[float32]([1, 100], 0, 1))) ]#

#[ let pred = model.forward([[1.0'f32, 2.0]].toTensor)
echo "Pred: ", pred
echo "Loss: ", model.loss.forward(pred, [[1.0'f32]].toTensor)
echo "Weight before: ", model.layers[0].weights["w"]
model.backward()
echo "Gradients after: ", model.layers[0].gradients["w"]
optimizer.update()
echo "Weight after update: ", model.layers[0].weights["w"]
model.zeroGrad()
echo "Gradient after zero: ", model.layers[0].gradients["w"] ]#


