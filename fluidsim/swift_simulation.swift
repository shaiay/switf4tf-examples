import TensorFlow
import Foundation

#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

let pil = Python.import("PIL.Image")
let np  = Python.import("numpy")

func load_image(_ path:String) -> array {
    let img = pil.open(path)
    var image = Tensor<UInt8>(numpy: np.array(img))!
    while image.rank > 2 {
        image = image[TensorRange.ellipsis, 0]
    }
    return array(image)
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @differentiable(wrt: (self, value))
  public mutating func update(_ i: Int, _ j: Int, to value: Self) {
    self[i, j] = value
  }

  @derivative(of: update, wrt: (self, value))
  @usableFromInline
  mutating func vjpUpdate(_ i: Int, _ j: Int, to value: Self) -> (
    value: (),
    pullback: (inout Self) -> Self
  ) {
    update(i, j, to: value)
    let valueShape = value.shape
    func pullback(tSelf: inout Self) -> Self {
      let tValue = tSelf[i, j]
      tSelf[i, j] = Tensor(zeros: valueShape)
      return tValue
    }
    return ((), pullback)
  }
}

typealias array = Tensor<Float>
struct Fluid: Differentiable, KeyPathIterable {
    var ivx: array
    var ivy: array
    @noDerivative var num_time_steps: Int
    
    @differentiable(wrt: (vx, vy))
    init(vx: array, vy:array, num_time_steps: Int) {
        self.num_time_steps = num_time_steps
        ivx = vx
        ivy = vy
    }
    
    func advect(f0: array, vx: array, vy: array) -> array {
        var f1 : array = Tensor(zeros: vx.shape)
        // rows -> y, cols -> x
        for r in 0..<f0.shape[0] {
            for c in 0..<f0.shape[1] {
                let center_y = Float(r) - vy[r, c].scalarized()
                let center_x = Float(c) - vx[r, c].scalarized()
                let left_ix = floor(center_x)
                let top_ix = floor(center_y)
                let rw = center_x - left_ix 
                let bw = center_y - top_ix
                let lx = Int(left_ix)     % f0.shape[0]
                let rx = Int(left_ix + 1) % f0.shape[0]
                let tx = Int(top_ix)      % f0.shape[1]
                let bx = Int(top_ix  + 1) % f0.shape[1]
                let left =  (1 - bw) * f0[withoutDerivative(at: lx), withoutDerivative(at: tx)] + bw * f0[withoutDerivative(at: lx), withoutDerivative(at: bx)]
                let right = (1 - bw) * f0[withoutDerivative(at: rx), withoutDerivative(at: tx)] + bw * f0[withoutDerivative(at: rx), withoutDerivative(at: bx)]
                f1.update(r, c, to: (1 - rw) * left + rw * right)
            }
        }
        return f1
    }
    
    @differentiable
    func callAsFunction(source: array) -> array {
        var vx = ivx
        var vy = ivy
        var smoke = source
        for _ in 0..<num_time_steps {
            vx = advect(f0: vx, vx: vx, vy: vy)
            vy = advect(f0: vy, vx: vx, vy: vy)
            smoke = advect(f0: smoke, vx: vx, vy: vy)
        }
        return smoke
    }
    
    @differentiable
    func call(source: array) -> array {
        callAsFunction(source: source)
    }
}

let source = load_image("init_smoke.png")
let y_full = load_image("skull.png")
let y = y_full[TensorRange.range(0..<y_full.shape[0], stride:2) ,TensorRange.range(0..<y_full.shape[1], stride:2)]
let num_time_steps = 5

var model = Fluid(vx: Tensor(zeros: source.shape), vy: Tensor(zeros: source.shape), num_time_steps: 50)
let optimizer = SGD(for: model, learningRate: 0.02)

print("Running model with \(num_time_steps) time steps and grid size \(source.shape)")
for i in 0..<10 {
    print("optimizing iteration \(i) - start")
    let (loss, 𝛁model) = valueWithGradient(at: model) { model -> Tensor<Float> in
        let ŷ = model(source: source)
        return (ŷ - y).squared().mean()
    }
    print("optimizing iteration \(i), loss: \(loss)")
    optimizer.update(&model, along: 𝛁model)
}