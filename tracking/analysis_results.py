import argparse
import _init_paths
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist


def parse_plot_types(s):
    allowed = {'success', 'norm_prec', 'prec'}
    parts = [p.strip() for p in s.split(',') if p.strip()]
    bad = [p for p in parts if p not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(
            'invalid plot type(s) %s; allowed: success, norm_prec, prec' % bad)
    return tuple(parts) if parts else ('success', 'norm_prec', 'prec')


def main():
    p = argparse.ArgumentParser(
        description='Analyze saved tracking results (reads txt under results_path / tracker / param / dataset).')
    p.add_argument('--dataset_name', type=str, default='MUSTHSI', help='Dataset key, e.g. MUSTHSI, hsitrack')
    p.add_argument('--tracker_name', type=str, default='sstrack', help='Tracker module name')
    p.add_argument('--tracker_param', type=str, default='baseline_must',
                   help='YAML name without .yaml under experiments/<tracker_name>/')
    p.add_argument('--runid', type=int, default=None,
                   help='If set, reads results under .../param_{runid:03d}/ (same as test --runid)')
    p.add_argument('--num_searches', type=int, default=2, help='Passed to trackerlist for SSTrack')
    p.add_argument('--display_name', type=str, default=None,
                   help='Legend / table display name; default uses name_param')
    p.add_argument('--plot_types', type=str, default='success,norm_prec,prec',
                   help='Comma-separated: success, norm_prec, prec')
    p.add_argument('--no_merge', action='store_true',
                   help='Disable averaging multiple random runs (merge_results=False)')
    args = p.parse_args()

    run_ids = args.runid
    trackers = trackerlist(
        name=args.tracker_name,
        parameter_name=args.tracker_param,
        dataset_name=args.dataset_name,
        run_ids=run_ids,
        num_searches=args.num_searches,
        display_name=args.display_name,
    )

    dataset = get_dataset(args.dataset_name)
    plot_types = parse_plot_types(args.plot_types)
    print_results(
        trackers,
        dataset,
        args.dataset_name,
        merge_results=not args.no_merge,
        plot_types=plot_types,
    )


if __name__ == '__main__':
    main()
