import std / [tables, sequtils, algorithm, intsets, math, random]
import arraymancer
import ./lsh_new
export lsh_new

type
  LayerKind* = enum
    Linear
    ReLU

  Layer* = object
    weights*: Table[string, Tensor[float32]]
    gradients*: Table[string, Tensor[float32]]

    case kind*: LayerKind
    of Linear:
      # must be assigned in the forward pass!
      input*: SparseVector
      output*: SparseVector
      inputSize*, outputSize*: int
      lshTabs*: seq[LSHTable]
      sparsity*: float32
    of ReLU:
      discard

  LossKind* = enum
    SoftmaxCrossEntropy

  LossFunction* = object
    case kind*: LossKind
    of SoftmaxCrossEntropy:
      soft*, yTrue*: Tensor[float32]

  Model* = object
    layers*: seq[Layer]
    loss*: LossFunction

  OptimizerKind* = enum
    StochGradDescent, AdamW
  Optimizer* = ref object
    lr*: float32
    paramGrads*: seq[tuple[param, grad: Tensor[float32]]]
    regularization*: float32
    case kind*: OptimizerKind
    of StochGradDescent:
      momentum*: float32
      velocity*: seq[Tensor[float32]]
    of AdamW:
      beta1*, beta2*: float32
      moments*: seq[tuple[m, v: Tensor[float32]]]

# Linear

proc forwardLinear*(lay: var Layer, x: QueryVector): SparseVector =
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
  # echo toSeq(indices) # not in order!
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

proc backwardLinearUnsafe*(lay: var Layer, deriv: SparseVector): SparseVector =
  # no batching, yippie! 
  # y1 = x1*w1 + x2*w2 + b
  # dy1/dw1 = x1 (once!)
  # dy1/db = 1
  # dy1/dx1 = w1 (multiple!)
  # we only have to update the weights which has non-zero xi and non-zero y.
  var biasGrad = lay.gradients["b"]
  var weightGrad = lay.gradients["w"]
  let weightWidth = weightGrad.shape[1]
  var wGradBuffer = weightGrad.toUnsafeView
  for (outIndex, deriv) in deriv.data:
    # Update bias
    biasGrad[0, outIndex] += deriv
  
  # Update weight
  # flip loop order for memory efficiency:
  for (inIndex, inValue) in lay.input.data.mitems:
    for (outIndex, deriv) in deriv.data:
      #weightGrad[inIndex, outIndex] += inValue * deriv
      wGradBuffer[inIndex * weightWidth + outIndex] += inValue * deriv
  
  result = newSparseVector(lay.inputSize, lay.input.data.len)
  let weights = lay.weights["w"]
  let wBuffer = weights.toUnsafeView
  for i, (inIndex, inValue) in lay.input.data.mpairs:
    result.data[i].index = inIndex
    var el: float32
    for (outIndex, deriv) in deriv.data:
      #result.data[i].el += deriv * weights[inIndex, outIndex]
      el += deriv * wBuffer[inIndex * weightWidth + outIndex]
    result.data[i].el = el

proc backwardLinear*(lay: var Layer, deriv: SparseVector): SparseVector =
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
  # flip loop order for memory efficiency:
  for (inIndex, inValue) in lay.input.data.mitems:
    for (outIndex, deriv) in deriv.data:
      weightGrad[inIndex, outIndex] += inValue * deriv
  
  result = newSparseVector(lay.inputSize, lay.input.data.len)
  let weights = lay.weights["w"]
  for i, (inIndex, inValue) in lay.input.data.mpairs:
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

proc newLinear*(inputSize, outputSize: int, sparsity: float32, denseOutput=false): Layer =
  # TODO: dense output for output layers. All layers should return QueryVector instead of SparseVector
  result = Layer(kind: Linear)
  result.inputSize = inputSize
  result.outputSize = outputSize
  result.sparsity = sparsity
  result.weights["w"] = randomNormalTensor[float32]([inputSize, outputSize], 0, sqrt(2 / (inputSize + outputSize)))
  result.weights["b"] = randomNormalTensor[float32]([1, outputSize], 0, sqrt(2 / (inputSize + outputSize)))
  result.gradients["w"] = zeros_like(result.weights["w"])
  result.gradients["b"] = zeros_like(result.weights["b"])
  let (k, l) = findKL(sparsity, outputSize.float32)
  echo "k: ", k, " L: ", l
  # Create L hash tables with K bits each
  for i in 0 ..< l:
    result.lshTabs.add initLSHTable(result.weights["w"].transpose, k=k)

# ReLU

proc forwardReLU*(lay: var Layer, x: QueryVector): SparseVector =
  let x = x.sparse
  result = newSparseVector(x.len, x.data.len)
  # Prune zeros, only keep elements which are non-zero!
  var i = 0
  for (index, value) in x.data:
    if value > 0:
      result.data[i] = (index: index, el: value)
      i += 1
  result.data.setLen(i)

proc backwardReLU*(lay: var Layer, deriv: SparseVector): SparseVector =
  # Because all zeros got pruned in the forward-pass, all derivatives arriving will just pass through!
  # We only backprop gradients to nodes that was passed forward so it will match nicely!
  result = deriv

proc newReLU*(): Layer =
  Layer(kind: ReLU)

# Layer

proc forward*(lay: var Layer, x: QueryVector): SparseVector =
  case lay.kind
  of Linear:
    forwardLinear(lay, x)
  of ReLU:
    forwardReLU(lay, x)

proc backward*(lay: var Layer, deriv: SparseVector): SparseVector =
  case lay.kind
  of Linear:
    backwardLinearUnsafe(lay, deriv)
  of ReLU:
    backwardReLU(lay, deriv)

proc parameters*(lay: Layer): seq[tuple[param, grad: Tensor[float32]]] =
  case lay.kind
  of Linear, ReLU:
    for key in lay.weights.keys:
      result.add (param: lay.weights[key], grad: lay.gradients[key])

# Loss functions

proc forwardSoftmaxCrossEntropy*(loss: var LossFunction, yPred, yTrue: Tensor[float32]): float32 =
  let soft = softmax(yPred.reshape(1, yPred.size.int))
  result = softmax_cross_entropy(yPred.reshape(1, yPred.size.int), yTrue.reshape(1, yPred.size.int))
  loss.soft = soft.squeeze
  loss.yTrue = yTrue

proc backwardSoftmaxCrossEntropy*(loss: var LossFunction): SparseVector =
  result = newSparseVector((loss.soft - loss.yTrue))

proc forward*(loss: var LossFunction, yPred, yTrue: Tensor[float32]): float32 =
  case loss.kind
  of SoftmaxCrossEntropy:
    forwardSoftmaxCrossEntropy(loss, yPred, yTrue)

proc backward*(loss: var LossFunction): SparseVector =
  case loss.kind
  of SoftmaxCrossEntropy:
    backwardSoftmaxCrossEntropy(loss)

proc newSoftmaxCrossEntropy*(): LossFunction =
  LossFunction(kind: SoftmaxCrossEntropy)

# Model
proc forward*(model: var Model, x: Tensor[float32]): Tensor[float32] =
  var temp = x.toQueryVector()
  for lay in model.layers.mitems:
    temp = lay.forward(temp).toQueryVector()
  result = temp.sparse.toTensor

proc backward*(model: var Model, batchSize: int) =
  var deriv = model.loss.backward()
  deriv /= batchSize.float32
  for i in countdown(model.layers.high, 0):
    deriv = model.layers[i].backward(deriv)

proc parameters*(model: Model): seq[tuple[param, grad: Tensor[float32]]] =
  for lay in model.layers:
    result.add lay.parameters()

proc zeroGrad*(model: var Model) =
  for lay in model.layers.mitems:
    for w in lay.gradients.keys:
      lay.gradients[w] *= 0

proc updateTables*(model: var Model) =
  for lay in model.layers.mitems:
    case lay.kind
    of Linear:
      for tab in lay.lshTabs.mitems:
        tab.updateTable()
    of ReLU:
      discard

# Optimizer

proc updateAdamW*(opt: Optimizer) =
  let lr = opt.lr
  let reg = opt.regularization
  let beta1 = opt.beta1
  let beta2 = opt.beta2
  for i, (w, grad) in opt.paramGrads.mpairs:
    var (m, v) = opt.moments[i]
    if reg > 0:
      w.apply_inline:
        x - reg * lr * x
    m.apply2_inline(grad):
      beta1 * x + (1 - beta1) * y
    v.apply2_inline(grad):
      beta2 * x + (1 - beta2) * y*y
    w.apply3_inline(m, v):
      x - lr * y  / (sqrt(z) + 1e-8)

proc newAdamW*(model: Model, lr: float32 = 1e-3, reg: float32 = 1e-2, beta1: float32 = 0.9, beta2: float32 = 0.999): Optimizer =
  result = Optimizer(kind: AdamW, lr: lr, regularization: reg, beta1: beta1, beta2: beta2)
  let parameters = model.parameters()

  # Change:
  for (w, grad) in parameters:
    result.moments.add (zeros_like(w), zeros_like(w))

  result.paramGrads = parameters

proc update*(opt: Optimizer) =
  case opt.kind
  of StochGradDescent:
    discard
  of AdamW:
    updateAdamW(opt)