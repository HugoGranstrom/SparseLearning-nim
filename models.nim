import std/[tables]
import arraymancer

type
  Layer* = ref object of RootObj
    weights*: Table[string, Tensor[float32]]
    gradients*: Table[string, Tensor[float32]]

  LossFunction* = ref object of RootObj

  Model* = ref object
    layers*: seq[Layer]
    loss*: LossFunction

type
  Optimizer* = ref object of RootObj
    alpha*: float32
    model*: Model
    regularization*: float32
  StochGradDescent* = ref object of Optimizer
    momentum*: float32
    velocity*: Table[Layer, Table[string, Tensor[float32]]]

method update*(opt: Optimizer) {.base.} = doAssert false, "All optimizers need to implement `update` method"

method update*(opt: StochGradDescent) =
  for lay in opt.model.layers:
    for weight in lay.weights.keys:
      #if weight == "b": continue
      var vel = opt.velocity[lay][weight]
      let grad = 
        if opt.regularization != 0:
          lay.gradients[weight] + opt.regularization * lay.weights[weight]
        else:
          lay.gradients[weight]
      # update vel
      apply2_inline(vel, grad):
        opt.momentum * x + y
      # update weight
      apply2_inline(lay.weights[weight], vel):
        x - opt.alpha * y

proc newSGD*(model: Model, alpha: float32, momentum: float32 = 0.9, reg: float32 = 0.0): StochGradDescent =
  result = StochGradDescent(model: model, alpha: alpha, momentum: momentum, regularization: reg)
  for lay in model.layers:
    result.velocity[lay] = initTable[string, Tensor[float32]]()
    for weight in lay.weights.keys:
      result.velocity[lay][weight] = zeros_like(lay.weights[weight])