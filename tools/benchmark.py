import time

from src.pipeline import stack_target


def benchmark_pipeline(frames, output_path, args):
    start_time = time.time()
    stack_target(frames, output_path, args)
    end_time = time.time()
    print(f"Pipeline completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs="+", required=True, help="List of frame paths")
    parser.add_argument("--output", required=True, help="Output path")
    args = parser.parse_args()
    benchmark_pipeline(args.frames, args.output, args)
