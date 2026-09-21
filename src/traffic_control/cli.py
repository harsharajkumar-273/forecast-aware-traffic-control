import argparse,json
from .config import Config
from .evaluate import evaluate
def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True); s=sub.add_parser("smoke"); s.add_argument("--seed",type=int,default=7); e=sub.add_parser("evaluate"); e.add_argument("--episodes",type=int,default=1); e.add_argument("--steps",type=int,default=240); e.add_argument("--output",default="runs/v1.json"); i=sub.add_parser("ingest-aikosh"); i.add_argument("--input",required=True); i.add_argument("--output",required=True)
 a=p.parse_args()
 if a.cmd=="ingest-aikosh":
  from .aikosh import write_profile; print(json.dumps(write_profile(a.input,a.output),indent=2)); return
 cfg=Config(seed=getattr(a,"seed",7),episode_steps=getattr(a,"steps",240)); result=evaluate(cfg,a.episodes if a.cmd=="evaluate" else 1); print(json.dumps(result,indent=2));
 if a.cmd=="evaluate":
  import pathlib; pathlib.Path(a.output).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(a.output).write_text(json.dumps({"config":cfg.to_dict(),"results":result},indent=2))

if __name__ == "__main__":
 main()
