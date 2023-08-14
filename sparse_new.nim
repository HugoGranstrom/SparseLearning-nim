import std/[random, tables, math, sequtils, strformat, intsets, times]
import arraymancer except Optimizer
import ./models_sparse, ./mnist

proc train*(model: var Model, x, y, xVal, yVal: Tensor[float32], optimizer: Optimizer, epochs: int, batchSize: int) =
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
      #scheduler.step()
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
let sparsity = 0.1
let hidden = 800
var model = Model()
model.loss = newSoftmaxCrossEntropy()
model.layers.add newLinear(28*28, hidden, sparsity)
model.layers.add newReLU()
model.layers.add newLinear(hidden, hidden, sparsity)
model.layers.add newReLU()
model.layers.add newLinear(hidden, 10, 1)

let epochs = 1
let bs = 64 
let iters = epochs * xTrain.shape[0] div bs

#let optimizer = newSGD(model, 2e-2, reg=0)
let optimizer = newAdamW(model, lr=1e-4, reg=1e-2)

#let scheduler = newOneCycle(optimizer, maxLR=3e-4'f32, divFactor=25'f32, cycleLength=iters)

model.train(xTrain, yTrain, xTest, yTest, optimizer, epochs=epochs, batchSize=bs)

#echo "After:"
echo model.layers[0].weights["w"].mean()