import logging
import yaml

from scor_dm.evaluate_sbgm.evaluation_main import evaluation_main
from scor_dm.evaluate_sbgm.plot_utils import make_publication_outputs

from scor_dm.evaluate.evaluate_main import evaluation_main as evaluation_main_new

logger = logging.getLogger(__name__)

def run_evaluation(cfg, make_plots=True):
    fe = cfg.get("full_gen_eval", {})
    use_new = bool(fe.get("use_new_eval", False))


    # Launch the evaluation process
    if use_new:
        logger.info("[launch_evaluation] Using NEW evaluation main.")
        evaluation_main_new(cfg)
    else:
        evaluation_main(cfg)

        # Make publication-ready plots
        if make_plots:
            make_publication_outputs(cfg)
