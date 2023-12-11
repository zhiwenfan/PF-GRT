from models.model import GRTModel
import config

parser = config.config_parser()
args = parser.parse_args()
device = "cuda:{}".format(args.local_rank)
args.ckpt_path = "model_720000.pth"

model = GRTModel(
    args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
)

