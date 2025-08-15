import argparse
import ambient_utils
import numpy as np
import json
import os
parser = argparse.ArgumentParser(description='Evaluate FID score')
parser.add_argument("--gen_path", type=str, help="Path to generated images", required=True)
parser.add_argument("--ref_path", type=str, help="Path to reference images", default=None)
parser.add_argument("--ref_stats", type=str, help="Path to reference stats", default=None)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for FID calculation")
parser.add_argument("--out_path", type=str, default="fid_out.json", help="Path to json file to save results")

def main(args):
    assert args.ref_path is not None or args.ref_stats is not None, "Either reference path or reference stats should be provided"
    if args.ref_stats is not None:
        ref_stats = np.load(args.ref_stats)
        ref_mu = ref_stats['mu']
        ref_sigma = ref_stats['sigma']
    else:
        ref_mu, ref_sigma, _ = ambient_utils.eval_utils.calculate_inception_stats(args.ref_path, max_batch_size=args.batch_size)
    mu, sigma, inception_score = ambient_utils.eval_utils.calculate_inception_stats(args.gen_path, max_batch_size=args.batch_size)
    fid_score = ambient_utils.eval_utils.calculate_fid_from_inception_stats(mu, sigma, ref_mu, ref_sigma)

    print(f"FID score: {fid_score}")

    # save results to json file
    results = {"inception_score": inception_score, "fid_score": fid_score}
    with open(os.path.join(args.gen_path, args.out_path), "w") as f:
        json.dump(results, f)


    


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)