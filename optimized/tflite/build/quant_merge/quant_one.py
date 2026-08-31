import sys,os; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from ai_edge_quantizer import quantizer, recipe
src,dst=sys.argv[1],sys.argv[2]
q=quantizer.Quantizer(src); q.load_quantization_recipe(recipe.dynamic_wi8_afp32())
q.quantize().export_model(dst)
print(f"{os.path.basename(dst)} {os.path.getsize(dst)/1e6:.0f}MB")
