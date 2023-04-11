import std/[random, tables, math, sequtils, strformat]
import arraymancer, benchy

type
  SparseVector* = ref object
    data*: seq[tuple[index: int, el: float32]]
    len*: int

proc `/=`*(v: SparseVector, x: float32) =
  for (index, el) in v.data.mitems:
    el /= x

proc `/`*(v: SparseVector, x: float32): SparseVector =
  new result
  result.data = v.data
  result.len = v.len
  result /= x

proc dot*(v1: SparseVector, v2: Tensor[float32]): float32 =
  assert v1.len == v2.size, &"{v1.len}, {v2.size}"
  assert v2.rank == 1
  for (index, el) in v1.data:
    result += el * v2[index]

proc dot*(v1: Tensor[float32], v2: SparseVector): float32 =
  dot(v2, v1)

proc norm2*(v: SparseVector): float32 =
  for (index, el) in v.data:
    result += el*el

proc norm*(v: SparseVector): float32 =
  sqrt(norm2(v))

proc newSparseVector*(len: int, allocLen: int): SparseVector =
  new result
  result.len = len
  result.data = newSeq[tuple[index: int, el: float32]](allocLen)

proc newSparseVector*(x: seq[float32] or Tensor[float32]): SparseVector =
  new result
  result.len = when x is seq: x.len else: x.size
  for i in 0 ..< result.len:
    if x[i] != 0:
      result.data.add (index: i, el: x[i])

proc toTensor*(v: SparseVector): Tensor[float32] =
  result = newTensor[float32](v.len)
  for (index, value) in v.data:
    result[index] = value