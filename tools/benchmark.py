"""Parameters, MACs and GPU latency of P2Echo (Table VII). Prompt embeddings are precomputed."""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2echo import P2Echo  # noqa: E402


class MainOutput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, prompts):
        return self.model(image, prompts)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    model = P2Echo().eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M "
          "(frozen text encoder excluded)")

    try:
        from thop import profile
        macs, _ = profile(MainOutput(model).eval(),
                          inputs=(torch.randn(1, 3, 256, 256), torch.randn(1, args.num_prompts, 1024)),
                          verbose=False)
        print(f"MACs (thop): {macs / 1e9:.2f} G")
    except ImportError:
        print("Install thop to count MACs.")

    if not torch.cuda.is_available():
        return
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    model.to(device)
    g = torch.Generator().manual_seed(42)
    prompts = torch.nn.functional.normalize(torch.randn(1, args.num_prompts, 1024, generator=g), dim=-1).to(device)
    image = torch.randn(1, 3, 256, 256, generator=g).to(device)

    times = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image, prompts)
        for _ in range(args.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                model(image, prompts)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) / args.iterations)
    times = np.array(times)
    print(f"Latency ({torch.cuda.get_device_name()}, FP32, batch 1): "
          f"{times.mean():.5f} ± {times.std(ddof=1):.5f} s/image ({1 / times.mean():.2f} FPS)")


if __name__ == "__main__":
    main()
