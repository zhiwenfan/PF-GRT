from gnt.model import GNTModel
import config
from gnt.projection import Projector
from gnt.sample_ray import RaySamplerSingleImage
from eval import log_view

parser = config.config_parser()
args = parser.parse_args()
device = "cuda:{}".format(args.local_rank)
args.ckpt_path = "model_720000.pth"

model = GNTModel(
    args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
)

