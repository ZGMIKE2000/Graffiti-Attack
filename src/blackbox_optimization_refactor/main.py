import argparse
from models import GraffitiGenerator, ObjectDetector
from optimizer import BlackBoxOptimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", required=True, help="Path to graffiti patch PNG (RGBA)")
    parser.add_argument("--img", required=True, help="Path to target image")
    parser.add_argument("--bbox", required=False, help="Path to YOLO bbox .txt (same basename)", default=None)
    parser.add_argument("--yolo", required=True, nargs='+', help="Path(s) to yolo .pt model(s)")
    parser.add_argument("--target-class", type=int, required=True)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=8)
    args = parser.parse_args()

    gen = GraffitiGenerator(patch_path=args.patch)
    det = ObjectDetector(args.yolo)
    opt = BlackBoxOptimizer(
        generator=gen, detector=det,
        image_paths=[args.img], bbox_paths=[args.bbox],
        target_class_id=args.target_class,
        num_generations=args.generations, population_size=args.population
    )
    best = opt.run()
    print("Best found:", best)

if __name__ == "__main__":
    main()