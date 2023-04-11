import std/[random, tables, math, sequtils]
import arraymancer, benchy, bitty
import ./sparsevector
export sparsevector

type
  LSHTable* = ref object
    tabs*: Table[BitArray, seq[int]]
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

proc `/`*(v: QueryVector, x: float32): QueryVector =
  new result
  if v.isDense:
    result.isDense = true
    result.dense = v.dense / x
  else:
    result.isDense = false
    result.sparse = v.sparse / x

proc lshProjection*(p: QueryVector, a: Tensor[float32]): bool =
  var x =
    if p.isDense:
      p.dense.dot(a[0..^2])
    else:
      p.sparse.dot(a[0..^2])
  x += p.lshEl * a[a.size.int-1]
  result = not signbit(x)

proc simpleLsh*(vec: QueryVector, a: Tensor[float32], doNormalize: bool): BitArray =
  let v_norm2 = norm2(vec)
  var p = vec
  if doNormalize:
    p = p / sqrt(v_norm2)
    p.lshEl = 0.0
  else:
    p.lshEl = sqrt(max(0, 1 - v_norm2))
  
  #let p = concat(vec, [sqrt(max(0, 1 - v_norm2))].toTensor, axis=0)
  let n = a.shape[0]
  #result = newSeqOfCap[bool](n)
  result = newBitArray(n)
  for i in 0 ..< n:
    result[i] = lshProjection(p, a[i, _].squeeze)

proc query*(tab: LSHTable, vec: QueryVector, multiprobe = false): seq[int] =
  var fingerprint = simpleLsh(vec, tab.randomVectors, true)
  result = tab.tabs.getOrDefault(fingerprint, @[])
  if multiprobe:
    for i in 0 ..< tab.k:
      fingerprint[i] = not fingerprint[i]
      result.add tab.tabs.getOrDefault(fingerprint, @[])
      fingerprint[i] = not fingerprint[i]

proc updateTable*(tab: var LSHTable) =
  tab.tabs = initTable[BitArray, seq[int]]()
  # normalize according to largest norm
  let maxNorm = sqrt(max(sum((tab.vectors ^. 2), axis=1)))
  tab.maxNorm = maxNorm
  #let maxes = reduce_axis_inline(tab.vectors, 1):
  #  x += y *. y

  #let maxNorm = sqrt(max(maxes))

  let normVectors = tab.vectors / maxNorm
  let nVectors = normVectors.shape[0]
  for i in 0 ..< nVectors:
    let fingerprint = simpleLsh(normVectors[i, _].squeeze.toQueryVector, tab.randomVectors, false)
    #if i == 100: echo fingerprint
    tab.tabs.mgetOrPut(fingerprint, @[]).add(i)

proc initLSHTable*(vectors: Tensor[float32], k: int): LSHTable =
  new result
  result.vectors = vectors
  result.k = k
  result.randomVectors = randomNormalTensor[float32]([k, vectors.shape[1] + 1], 0, 1)
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

  #[ echo lshTab.query(p)
  echo lshTab.query(p).len / 10000
  echo (x[lshTab.query(p), _] * p.reshape(N, 1)).mean()
  echo (x * p.reshape(N, 1)).mean()

  var s: int
  var cTab: CountTable[int]

  let nIters = 5

  for c in 0 ..< nIters:
    lshTab = initLSHTable(x, 9)
    #echo "x[100]: ", simpleLsh(x[100, _].squeeze, lshTab.randomVectors, true)
    #echo "p:      ", simpleLsh(p, lshTab.randomVectors, true)
    #echo "------------------"
    #echo lshTab.tabs
    let indices = lshTab.query(p)
    cTab.merge indices.toCountTable
    if 100 in indices:
      #echo "Match! ", c
      s += 1

  echo "Percent with 100: ", s / nIters
  echo "Density: ", cTab.len / 10000
  cTab.sort()
  #echo cTab ]#

    #echo dots.max()
    #echo dots.sum() / dots.len.float32
    #echo indices
    #for i in indices:
    #  echo p.dot(x[i, _].squeeze)


  #[ let r = 2.5
  let U = 0.83
  let b = rand(r)
  let m = 3 ]#
  #[ let a = randomNormalTensor[float32]([N+m], 0, 1) # hashing
  let x = randomNormalTensor[float32]([N], 0, 1) # query
  let p = randomNormalTensor[float32]([N], 0, 1) # "weights" ]#



  #[ proc lsh(vec: Tensor[float32], isQuery=true): int =
    let x =
      if isQuery:
        let vec = vec / norm(vec)
        concat(vec, zeros[float32](m) +. 0.5, axis=0)
      else:
        let vec = vec / norm(vec) * U
        concat(vec, norm(vec) ^. (2.0 ^. arange[float32](1.0, m.float32 + 0.1)), axis=0)
        
    int(floor((a.dot(x) + b) / r))

  echo "Weight:", lsh(p, isQuery=false)

  for i in 0 .. 10:
    let t = randomNormalTensor[float32]([N], 0, 1)#p + randomTensor[float32]([N], 5.0) -. 2.5
    echo lsh(t), " ", t.dot(p) / (norm(t)*norm(p)) ]#