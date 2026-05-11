from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.musthsi_path = '/data3/PublicDataset/Public/MUST-BIT/HSICV/'

    settings.hsitrack_path = '/data/users/fengtao/SSTrack/data/HSICV/'
    settings.network_path = '/data/users/qinhaolin01/SSTrack-fengtao/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/data/users/qinhaolin01/SSTrack-fengtao'
    settings.result_plot_path = '/data/users/qinhaolin01/SSTrack-fengtao/output/test/result_plots'
    settings.results_path = '/data/users/qinhaolin01/SSTrack-fengtao/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/users/qinhaolin01/SSTrack-fengtao/save'
    settings.segmentation_path = '/data/users/qinhaolin01/SSTrack-fengtao/output/test/segmentation_results'

    return settings

