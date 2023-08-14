import std/[random, tables, math, sequtils, strformat, intsets, times]
import arraymancer except Optimizer
import weave, loony
import ./models_sparse, ./mnist

proc zeroGradParallel*(model: var Model) =
  let model = model # copy
  let modelPtr = model.addr
  parallelFor i in 0 .. model.layers.high:
    captures: {modelPtr}
    var lay = modelPtr[].layers[i] # copy
    for w in lay.gradients.keys:
      lay.gradients[w] *= 0
  syncRoot(Weave)

proc updateAdamWParallel*(opt: Optimizer) =
  let lr = opt.lr
  let reg = opt.regularization
  let beta1 = opt.beta1
  let beta2 = opt.beta2
  let paramGradPtr = opt.paramGrads.addr
  let momentsPtr = opt.moments.addr
  parallelFor i in 0 .. opt.paramGrads.high:
    captures: {lr, reg, beta1, beta2, paramGradPtr, momentsPtr}
    var (w, grad) = paramGradPtr[][i] # copy
    var (m, v) = momentsPtr[][i] # copy
    if reg > 0:
      w.apply_inline:
        x - reg * lr * x
    m.apply2_inline(grad):
      beta1 * x + (1 - beta1) * y
    v.apply2_inline(grad):
      beta2 * x + (1 - beta2) * y*y
    w.apply3_inline(m, v):
      x - lr * y  / (sqrt(z) + 1e-8)
  syncRoot(Weave)

proc updateParallel*(opt: Optimizer) =
  case opt.kind
  of StochGradDescent:
    discard
  of AdamW:
    updateAdamWParallel(opt)

proc updateTablesParallel*(model: var Model) =
  let model = model # copy
  let modelPtr = model.addr
  parallelFor i in 0 .. model.layers.high:
    captures: {modelPtr}
    var lay = modelPtr[].layers[i] # copy
    case lay.kind
    of Linear:
      for tab in lay.lshTabs.mitems:
        tab.updateTable()
    of ReLU:
      discard
  syncRoot(Weave)

proc train*(model: var Model, x, y, xVal, yVal: Tensor[float32], optimizer: Optimizer, epochs: int, batchSize: int) =
  #init(Weave)
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
      let modelTemp = model
      let modelPtr = modelTemp.addr
      let batchXPtr = batchX.addr
      let batchYPtr = batchY.addr
      #let before = model.layers[0].weights["w"].abs.mean()
      # something is wrong with this!
      for i in 0 ..< bs:
        #captures: {modelPtr, batchXPtr, batchYPtr, bs}
        var model = modelPtr[] # copy
        let pred = model.forward(batchXPtr[][i, _].squeeze(0))
        let loss = model.loss.forward(pred, batchYPtr[][i, _].squeeze(0))
        #let pred = model.forward(batchX[i, _].squeeze(0))
        #let loss = model.loss.forward(pred, batchY[i, _].squeeze(0))
        model.backward(bs)
      #syncRoot(Weave)
      optimizer.update()
      #let after = model.layers[0].weights["w"].abs.mean()
      #echo "Before: ", before, " After: ", after
      model.zeroGrad()
      #scheduler.step()
      iter += 1
      if iter == nextTableUpdate:
        #echo "Updating tables"
        model.updateTables()
        nUpdates += 1
        nextTableUpdate = nextTableUpdate + round(n0.float * exp(lambda * nUpdates.float)).int
    #[ var valLoss, valAcc: float32
    for i in 0 ..< xVal.shape[0]:
      let pred = model.forward(xVal[i, _].squeeze(0))
      let predIndex = pred.argmax(0)[0]
      let trueIndex = yVal[i, _].squeeze.argmax(0)[0]
      if predIndex == trueIndex: valAcc += 1
      let loss = model.loss.forward(pred, yVal[i, _].squeeze(0))
      valLoss += loss
    valLoss /= xVal.shape[0].float32
    valAcc /= xVal.shape[0].float32
    echo &"Epoch {epoch}: val loss = {valLoss :.2f}, val acc = {valAcc:.2f}" ]#
      # TODO: update the hashes of all weight-vectors that has been updated!
      # Change to updating the weights on the fly! HOGWILD!
  #exit(Weave)
    
    
let (xTrain, yTrain, xTest, yTest) = load_preprocess_mnist()

#randomize()

#[ echo findKL(0.1, 100)
let lay = newLinear(100, 2000, 0.05)
let lay2 = newLinear(2000, 2000, 0.05) ]#
let sparsity = 0.075
let hidden = 1000
var model = Model()
model.loss = newSoftmaxCrossEntropy()
model.layers.add newLinear(28*28, hidden, sparsity)
model.layers.add newReLU()
model.layers.add newLinear(hidden, hidden, sparsity)
model.layers.add newReLU()
model.layers.add newLinear(hidden, hidden, sparsity)
model.layers.add newReLU()
model.layers.add newLinear(hidden, hidden, sparsity)
model.layers.add newReLU()
model.layers.add newLinear(hidden, hidden, sparsity)
model.layers.add newReLU()
model.layers.add newLinear(hidden, 10, 1)

let epochs = 1
let bs = 64*10*2
let iters = epochs * xTrain.shape[0] div bs

#let optimizer = newSGD(model, 2e-2, reg=0)
let optimizer = newAdamW(model, lr=1e-4*0, reg=0)

#let scheduler = newOneCycle(optimizer, maxLR=3e-4'f32, divFactor=25'f32, cycleLength=iters)

model.train(xTrain, yTrain, xTest, yTest, optimizer, epochs=epochs, batchSize=bs)

#echo "After:"
echo model.layers[0].weights["w"].mean()
