import sys,os; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from ai_edge_quantizer import quantizer, recipe, qtyping
from ai_edge_quantizer.algorithm_manager import AlgorithmName
src,dst=sys.argv[1],sys.argv[2]
q=quantizer.Quantizer(src); q.load_quantization_recipe(recipe.dynamic_wi8_afp32())
# Keep the baked output limiter in fp32: int8 on its 9-tap Hann filter makes the smoothed
# gain no longer sum to 1, breaking the brickwall ceiling (w8a8 peaked 1.015 > 1.0). The
# limiter is tiny, so fp32 there costs ~nothing. No-op on encoders (no OutputLimiter ops).
q.update_quantization_recipe(regex=".*OutputLimiter.*",
                             operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                             algorithm_key=AlgorithmName.NO_QUANTIZE)
q.quantize().export_model(dst)
print(f"{os.path.basename(dst)} {os.path.getsize(dst)/1e6:.0f}MB")
