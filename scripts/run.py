import argparse
from perceptx.detect import Detector
from perceptx.segment import Segmentor
from perceptx.track import Tracker
from perceptx.pipeline import PerceptXPipeline

def main():
    parser = argparse.ArgumentParser(description="PerceptX Pipeline")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument("--sam-checkpoint", type=str, default="sam_vit_b_01ec64.pth")
    args = parser.parse_args()
    detector = Detector(device=args.device)

    segmentor = None
    if args.segment:
        segmentor = Segmentor(
            checkpoint_path=args.sam_checkpoint,
            device=args.device
        )

    tracker = Tracker()

    pipeline = PerceptXPipeline(
        detector=detector,
        segmentor=segmentor,
        tracker=tracker
    )

    output = pipeline.process_video(args.video)
    print(f"Done! Saved to {output}--->PerceptX XO")

if __name__ == "__main__":
    main()