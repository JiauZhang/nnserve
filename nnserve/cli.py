import argparse

import uvicorn

from nnserve.app import create_app
from nnserve.config import ServeConfig


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nnserve")
    parser.add_argument("--config", required=True, help="path to the serve config JSON")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)

    config = ServeConfig.from_json(args.config)
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
