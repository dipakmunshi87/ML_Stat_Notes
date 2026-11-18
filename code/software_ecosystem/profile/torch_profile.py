"""PyTorch's built-in profiler: which operators take the time (CPU here; add ProfilerActivity.CUDA on a GPU)."""
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

model = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 512))
x = torch.randn(256, 512)
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("training_step"):
        for _ in range(20):
            loss = model(x).pow(2).mean()
            loss.backward()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=6, max_name_column_width=36))
prof.export_chrome_trace("trace.json")   # open in https://ui.perfetto.dev or chrome://tracing
