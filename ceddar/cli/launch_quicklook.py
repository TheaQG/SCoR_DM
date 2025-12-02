import logging
import yaml

from scor_dm.generate.quicklook import quicklook_from_runner


logger = logging.getLogger(__name__)

def run_quicklook(cfg):
    quicklook_from_runner(cfg)