import std/[random, tables, math, sequtils, strformat, intsets, times]
import arraymancer except Optimizer
import benchy
import models
import ./lsh, ./mnist

let v = newSparseVector(@[0'f32, 1, 2, 0, 4, 0].toTensor)
#echo v[]

#echo v.dot([1.0'f32, 1, 1, 1, 1, 1].toTensor)

type
  Linear* = ref object of Layer
    input*: SparseVector
    output*: SparseVector
    inputSize*, outputSize*: int
    lshTabs*: seq[LSHTable]
    sparsity*: float32
  ReLU* = ref object of Layer
    input*: SparseVector
    inputSize*, outputSize*: int

  MeanSquareError* = ref object of LossFunction
    diff*: Tensor[float32]
  SoftmaxCrossEntropy* = ref object of LossFunction
    soft*, yTrue*: Tensor[float32]
  

method forward*(lay: Layer, x: QueryVector): SparseVector {.base.} = doAssert false, "All layers need to implement `forward` method"
method backward*(lay: Layer, deriv: SparseVector): SparseVector {.base.} = doAssert false, "All layers need to implement `backward` method"

method forward*(lay: Linear, x: QueryVector): SparseVector =
  # Find most probable winning activations
  # Vanilla sampling, pick random tables until we get enough nodes or have picked all
  #assert x.len == lay.inputSize, &"{x.len}, {lay.inputSize}"
  let expectedNodes = round(lay.sparsity * lay.outputSize.float32).int
  lay.lshTabs.shuffle()
  #var indices = newSeqOfCap[int](2*expectedNodes)
  var indices = initIntSet()
  let queryVector = x#x.toQueryVector
  for i, tab in lay.lshTabs:
    if indices.len >= expectedNodes: break
    let queryIndices = tab.query(queryVector)
    #indices.add queryIndices
    for ind in queryIndices:
      indices.incl ind
  #echo indices.len / lay.outputSize 
  # TODO: use multiprobing if we don't reach expectedNodes. (save all fingerprints in a list)
  #echo &"Expected: {expectedNodes}, Found: {indices.len}"
  # Calculate activations
  result = SparseVector(len: lay.outputSize, data: newSeq[tuple[index: int, el: float32]](indices.len))
  let weights = lay.weights["w"]
  let bias = lay.weights["b"]
  var i = 0
  for index in indices:
    result.data[i].index = index
    let wx =
      if x.isDense:
        dot(x.dense, weights[_, index].squeeze)
      else:
        dot(x.sparse, weights[_, index].squeeze)
    result.data[i].el = wx + bias[0, index]
    i += 1
  lay.input = if x.isDense: newSparseVector(x.dense) else: x.sparse
  lay.output = result

method backward*(lay: Linear, deriv: SparseVector): SparseVector =
  # no batching, yippie! 
  # y1 = x1*w1 + x2*w2 + b
  # dy1/dw1 = x1 (once!)
  # dy1/db = 1
  # dy1/dx1 = w1 (multiple!)
  # we only have to update the weights which has non-zero xi and non-zero y.
  var biasGrad = lay.gradients["b"]
  var weightGrad = lay.gradients["w"]
  for (outIndex, deriv) in deriv.data:
    # Update bias
    biasGrad[0, outIndex] += deriv
    # Update weight
    for (inIndex, inValue) in lay.input.data:
      weightGrad[inIndex, outIndex] += inValue * deriv
  
  result = newSparseVector(lay.inputSize, lay.input.data.len)
  let weights = lay.weights["w"]
  for i, (inIndex, inValue) in lay.input.data:
    result.data[i].index = inIndex
    for (outIndex, deriv) in deriv.data:
      result.data[i].el += deriv * weights[inIndex, outIndex]

proc findKL*(s, d: float32, c1: float32 = 1, c2: float32 = 0.1): tuple[k, l: int] =
  let startK = ceil(log2(1/(c1*s))).int
  result = (k: startK, l: round(c1 * s * 2'f32 ^ startK).int)
  for K in startK .. 256:
    let L = c1 * s * (2.0'f32 ^ K)
    #echo &"L: {round(L)}, K: {K}"
    if L > 256: break
    if K.float32 * L + s*d > c2*d: break
    result = (k: K, l: round(L).int)

proc newLinear*(inputSize, outputSize: int, sparsity: float32): Linear =
  new result
  result.inputSize = inputSize
  result.outputSize = outputSize
  result.sparsity = sparsity
  result.weights["w"] = randomNormalTensor[float32]([inputSize, outputSize], 0, sqrt(2 / (inputSize + outputSize)))
  result.weights["b"] = randomNormalTensor[float32]([1, outputSize], 0, sqrt(2 / (inputSize + outputSize)))
  result.gradients["w"] = zeros_like(result.weights["w"])
  result.gradients["b"] = zeros_like(result.weights["b"])
  let (k, l) = findKL(sparsity, outputSize.float32)
  # Create L hash tables with K bits each
  for i in 0 ..< l:
    result.lshTabs.add initLSHTable(result.weights["w"].transpose, k=k)

method forward*(lay: ReLU, x: QueryVector): SparseVector =
  let x = x.sparse
  result = newSparseVector(x.len, x.data.len)
  # Prune zeros, only keep elements which are non-zero!
  var i = 0
  for (index, value) in x.data:
    if value > 0:
      result.data[i] = (index: index, el: value)
      i += 1
  result.data.setLen(i)
  #lay.input = result # we want to keep the pruning of zeros

method backward*(lay: ReLU, deriv: SparseVector): SparseVector =
  # Because all zeros got pruned in the forward-pass, all derivatives arriving will just pass through!
  # We only backprop gradients to nodes that was passed forward so it will match nicely!
  result = deriv

# Loss functions
method forward*(loss: LossFunction, yPred, yTrue: Tensor[float32]): float32 {.base.} = doAssert false, "All losses need to implement `forward` method"
method backward*(loss: LossFunction): SparseVector {.base.} =
  ## Return the derivative of the loss w.r.t yPred
  doAssert false, "All losses need to implement `backward` method"

method forward*(loss: MeanSquareError, yPred, yTrue: Tensor[float32]): float32 =
  result = mean_squared_error(yPred, yTrue)
  loss.diff = yPred - yTrue

method backward*(loss: MeanSquareError): SparseVector =
  result = (2 / loss.diff.size.float32 * loss.diff).newSparseVector()

method forward*(loss: SoftmaxCrossEntropy, yPred, yTrue: Tensor[float32]): float32 =
  let soft = softmax(yPred.reshape(1, yPred.size.int))
  result = softmax_cross_entropy(yPred.reshape(1, yPred.size.int), yTrue.reshape(1, yPred.size.int))
  loss.soft = soft.squeeze
  loss.yTrue = yTrue

method backward*(loss: SoftmaxCrossEntropy): SparseVector =
  result = newSparseVector((loss.soft - loss.yTrue))

# Model
proc forward*(model: Model, x: Tensor[float32]): Tensor[float32] =
  var temp = x.toQueryVector()
  for lay in model.layers:
    temp = lay.forward(temp).toQueryVector()
  result = temp.sparse.toTensor

proc backward*(model: Model, batchSize: int) =
  var deriv = model.loss.backward() / batchSize.float32
  for i in countdown(model.layers.high, 0):
    deriv = model.layers[i].backward(deriv)

proc zeroGrad*(model: Model) =
  for lay in model.layers:
    for w in lay.gradients.keys:
      lay.gradients[w] *= 0

proc updateTables*(model: Model) =
  for lay in model.layers:
    if lay of Linear:
      for tab in lay.Linear.lshTabs.mitems:
        tab.updateTable()

proc train*(model: Model, x, y, xVal, yVal: Tensor[float32], optimizer: Optimizer, scheduler: LRSchedule, epochs: int, batchSize: int) =
  let nX = x.shape[0]
  var iter = 0
  var nUpdates = 0
  let n0 = 10
  let lambda = 0.5
  var nextTableUpdate = n0
  for epoch in 0 ..< epochs:
    for batchIter in 0 ..< nX div batchSize:
      let batchX = x[batchIter*batchSize ..< min(nX, (batchIter+1)*batchSize), _]
      let batchY = y[batchIter*batchSize ..< min(nX, (batchIter+1)*batchSize), _]
      let bs = batchX.shape[0]
      for i in 0 ..< bs:
        let pred = model.forward(batchX[i, _].squeeze(0))
        let loss = model.loss.forward(pred, batchY[i, _].squeeze(0))
        model.backward(bs)
      optimizer.update()
      model.zeroGrad()
      scheduler.step()
      iter += 1
      if iter == nextTableUpdate:
        #echo "Updating tables"
        model.updateTables()
        nUpdates += 1
        nextTableUpdate = nextTableUpdate + round(n0.float * exp(lambda * nUpdates.float)).int
    var valLoss, valAcc: float32
    for i in 0 ..< xVal.shape[0]:
      let pred = model.forward(xVal[i, _].squeeze(0))
      let predIndex = pred.argmax(0)[0]
      let trueIndex = yVal[i, _].squeeze.argmax(0)[0]
      if predIndex == trueIndex: valAcc += 1
      let loss = model.loss.forward(pred, yVal[i, _].squeeze(0))
      valLoss += loss
    valLoss /= xVal.shape[0].float32
    valAcc /= xVal.shape[0].float32
    echo &"Epoch {epoch}: val loss = {valLoss :.2f}, val acc = {valAcc:.2f}"
      # TODO: update the hashes of all weight-vectors that has been updated!
      # Change to updating the weights on the fly! HOGWILD!
    
    
let (xTrain, yTrain, xTest, yTest) = load_preprocess_mnist()

#randomize()

#[ echo findKL(0.1, 100)
let lay = newLinear(100, 2000, 0.05)
let lay2 = newLinear(2000, 2000, 0.05) ]#
let sparsity = 0.075
let hidden = 800
var model = Model()
model.loss = SoftmaxCrossEntropy()
model.layers.add newLinear(28*28, hidden, sparsity)
model.layers.add ReLU()
model.layers.add newLinear(hidden, hidden, sparsity)
model.layers.add ReLU()
model.layers.add newLinear(hidden, 10, 1)

let epochs = 10
let bs = 64
let iters = epochs * xTrain.shape[0] div bs

#let optimizer = newSGD(model, 2e-2, reg=0)
let optimizer = newAdamW(model, lr=1e-4, reg=1e-2)

let scheduler = newOneCycle(optimizer, maxLR=3e-4'f32, divFactor=25'f32, cycleLength=iters)

#[ let N = 10000
let x = randomTensor[float32]([N, 2], -50'f32..50'f32) + randomNormalTensor[float32]([N, 2], 0, 0.1)

let w1 = 3.14'f32
let w2 = 2.71'f32
let b = 1.42'f32
let y = w1 * x[_, 0] + w2 * x[_, 1] +. b #+ sin(x[_, 0]) ]#

#[ echo "Before:"
echo model.layers[0].weights["w"]
echo model.layers[0].weights["b"] ]#



#timeIt "Train 10 epochs":

model.train(xTrain, yTrain, xTest, yTest, optimizer, scheduler, epochs=epochs, batchSize=bs)

#echo "After:"
echo model.layers[0].weights["w"].mean()
#echo model.layers[0].weights["b"]

#[ let x = [1.0'f32, 2.0].toTensor
echo "Before:"
echo model.layers[0].gradients["w"]
echo model.layers[0].gradients["b"]
let pred = model.forward(x)
echo pred
echo model.loss.forward([0.0'f32].toTensor, pred)
model.backward(1)
echo "After:"
echo model.layers[0].gradients["w"]
echo model.layers[0].gradients["b"] ]#

#[ timeIt "Sparse":
  keep lay2.forward(lay.forward(randomNormalTensor[float32]([100], 0, 1).newSparseVector)) ]#

# only update tables if norm is larger than max-norm
# SLIDE section 4.2 discuss this topic
# Keep track of active nodes before and after layer and only update the weights connecting them


