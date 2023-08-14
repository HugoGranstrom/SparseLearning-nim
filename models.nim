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
    lr*: float32
    model*: Model
    regularization*: float32
  StochGradDescent* = ref object of Optimizer
    momentum*: float32
    velocity*: Table[Layer, Table[string, Tensor[float32]]]
  AdamW* = ref object of Optimizer
    beta1*, beta2*: float32
    moments*: Table[Layer, Table[string, tuple[m, v: Tensor[float32]]]]

  LRSchedule* = ref object of RootObj
    opt*: Optimizer
    iter*: int
  OneCycle* = ref object of LRSchedule
    cycleLength*: int
    divFactor*: float32
    maxLR*: float32

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
        x - opt.lr * y

proc newSGD*(model: Model, lr: float32, momentum: float32 = 0.9, reg: float32 = 0.0): StochGradDescent =
  result = StochGradDescent(model: model, lr: lr, momentum: momentum, regularization: reg)
  for lay in model.layers:
    result.velocity[lay] = initTable[string, Tensor[float32]]()
    for weight in lay.weights.keys:
      result.velocity[lay][weight] = zeros_like(lay.weights[weight])

method update*(opt: AdamW) =
  let lr = opt.lr
  let reg = opt.regularization
  let beta1 = opt.beta1
  let beta2 = opt.beta2
  for lay in opt.model.layers:
    for weight in lay.weights.keys:
      var (m, v) = opt.moments[lay][weight]
      let grad = lay.gradients[weight]
      var w = lay.weights[weight]
      if reg > 0:
        w.apply_inline:
          x - reg * lr * x
      m.apply2_inline(grad):
        beta1 * x + (1 - beta1) * y
      v.apply2_inline(grad):
        beta2 * x + (1 - beta2) * y*y
      w.apply3_inline(m, v):
        x - lr * y  / (sqrt(z) + 1e-8)

proc newAdamW*(model: Model, lr: float32 = 1e-3, reg: float32 = 1e-2, beta1: float32 = 0.9, beta2: float32 = 0.999): AdamW =
  result = AdamW(model: model, lr: lr, regularization: reg, beta1: beta1, beta2: beta2)
  for lay in model.layers:
    result.moments[lay] = initTable[string, tuple[m, v: Tensor[float32]]]()
    for weight in lay.weights.keys:
      result.moments[lay][weight] = (zeros_like(lay.weights[weight]), zeros_like(lay.weights[weight]))

proc lerp*(p1, p2: tuple[x, y: float32], x: float32): float32 =
  let k = (p2.y - p1.y) / (p2.x - p1.x)
  let m = p1.y - k*p1.x
  result = k*x + m

method step*(sched: LRSchedule) {.base.} = doAssert false, "All Schedulers need to implement `step` method"

method step*(sched: OneCycle) =
  let progress = float32(sched.iter / sched.cycleLength)
  if progress < 0.3: # increasing phase
    sched.opt.lr = lerp((0'f32, sched.maxLR / sched.divFactor), (0.3'f32, sched.maxLR), progress)#sched.maxLR / sched.divFactor + k / 0.3 * progress
  elif progress < 0.6: # decreasing phase
    sched.opt.lr = lerp((0.3'f32, sched.maxLR), (0.6'f32, sched.maxLR / sched.divFactor), progress)
  else: # last phase
    sched.opt.lr = lerp((0.6'f32, sched.maxLR / sched.divFactor), (1'f32, sched.maxLR * 1e-4'f32), progress)
  #if sched.iter mod 100 == 0: echo sched.opt.lr
  sched.iter += 1

proc newOneCycle*(opt: Optimizer, maxLR, divFactor: float32, cycleLength: int): OneCycle =
  OneCycle(opt: opt, iter: 0, cycleLength: cycleLength, maxLR: maxLR, divFactor: divFactor)