import std/[random, tables, math, sequtils]
import arraymancer, benchy
import ./sparsevector
export sparsevector

type
  LSHTable* = ref object
    #tabs*: Table[BitArray, seq[int]]
    buckets*: seq[seq[int]] # length 2^k
    randomVectors*: Tensor[float32]
    vectors*: Tensor[float32]
    k*: int # number of bits to use
    maxNorm*: float32
  QueryVector* = ref object
    isDense*: bool
    sparse*: SparseVector
    dense*: Tensor[float32]
    lshEl*: float32

proc toQueryVector*(t: Tensor[float32]): QueryVector =
  new result
  result.isDense = true
  result.dense = t

proc toQueryVector*(v: SparseVector): QueryVector =
  new result
  result.isDense = false
  result.sparse = v

proc norm2(v: Tensor[float32]): float32 =
  dot(v, v)

proc norm(v: Tensor[float32]): float32 =
  sqrt(norm2(v))

proc norm2*(v: QueryVector): float32 =
  if v.isDense: norm2(v.dense) else: norm2(v.sparse)

#[ proc `/`(v: QueryVector, x: float32): QueryVector =
  new result
  if v.isDense:
    result.isDense = true
    result.dense = v.dense / x
  else:
    result.isDense = false
    result.sparse = v.sparse / x ]#

proc `/=`(v: var QueryVector, x: float32) =
  if v.isDense:
    v.dense /= x
  else:
    v.sparse /= x

proc lshProjection*(p: QueryVector, a: Tensor[float32]): bool =
  var x =
    if p.isDense:
      p.dense.dot(a[0..^2])
    else:
      p.sparse.dot(a[0..^2])
  x += p.lshEl * a[a.size.int-1]
  result = not signbit(x)

proc simpleLsh*(vec: QueryVector, a: Tensor[float32], doNormalize: bool): int =
  let v_norm2 = norm2(vec)
  var p = vec
  if doNormalize:
    p /= sqrt(v_norm2)
    p.lshEl = 0.0
  else:
    p.lshEl = sqrt(max(0, 1 - v_norm2))
  
  #let p = concat(vec, [sqrt(max(0, 1 - v_norm2))].toTensor, axis=0)
  let k = a.shape[0]

  #result = newSeqOfCap[bool](n)
  result = 0
  var step = 2 ^ k
  for i in 0 ..< k:
    let s = lshProjection(p, a[i, _].squeeze)
    step = step div 2
    if s:
      result += step
    # else: nada


proc query*(tab: LSHTable, vec: QueryVector, multiprobe = false): lent seq[int] =
  var fingerprint = simpleLsh(vec, tab.randomVectors, true)
  result = tab.buckets[fingerprint]
  # TODO:
  #[ if multiprobe:
    for i in 0 ..< tab.k:
      fingerprint[i] = not fingerprint[i]
      result.add tab.tabs.getOrDefault(fingerprint, @[])
      fingerprint[i] = not fingerprint[i] ]#

proc updateTable*(tab: var LSHTable) =
  # set all buckets to zero
  for b in tab.buckets.mitems:
    b.setLen(0)
  # normalize according to largest norm
  let maxNorm = sqrt(max(sum((tab.vectors *. tab.vectors), axis=1)))
  tab.maxNorm = maxNorm
  #let maxes = reduce_axis_inline(tab.vectors, 1):
  #  x += y *. y

  #let maxNorm = sqrt(max(maxes))

  let normVectors = tab.vectors / maxNorm
  let nVectors = normVectors.shape[0]
  for i in 0 ..< nVectors:
    let fingerprint = simpleLsh(normVectors[i, _].squeeze.toQueryVector, tab.randomVectors, false)
    #if i == 100: echo fingerprint
    tab.buckets[fingerprint].add(i)

proc initLSHTable*(vectors: Tensor[float32], k: int): LSHTable =
  new result
  result.vectors = vectors
  result.k = k
  # TODO: handle k = 0 (dense output)
  result.randomVectors = randomNormalTensor[float32]([k, vectors.shape[1] + 1], 0, 1)
  result.buckets = newSeq[seq[int]](2^k)
  result.updateTable()

when isMainModule:
  randomize()

  let N = 10000

  let x = randomNormalTensor[float32]([10000, N], 0, 1) # 1000 points with N elements each
  #let p = randomNormalTensor[float32]([N], 0, 1)
  let p = x[100, _].squeeze + randomNormalTensor[float32]([N], 0, 0.01)

  var dots: seq[float32]
  for i in 0 ..< 1000:
    dots.add p.dot(x[i, _].squeeze)

  echo argmax_max(dots.toTensor, 0)
  echo dots[100]

  var lshTab = initLSHTable(x, 6)

  timeIt "Init":
    keep initLSHTable(x, 6)

  timeIt "query":
    keep lshTab.query(p.toQueryVector)

  let weights = randomNormalTensor[float32]([1000, N], 0, 1)

  var lshTabWeights = initLSHTable(weights, 8)

  timeIt "Dense":
    keep weights * p.reshape(N, 1)

  timeIt "LSH":
    for _ in 0 .. 10:
      let indices = lshTabWeights.query(p.toQueryVector)
    let indices = lshTabWeights.query(p.toQueryVector)
    keep weights[indices, _] * p.reshape(N, 1)

  timeIt "updateTable":
    lshTabWeights.updateTable()