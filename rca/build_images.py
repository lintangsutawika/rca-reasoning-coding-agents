#!/usr/bin/env python3
"""
Build agent-server images for all unique SWE-Bench base images in a dataset split.

Example:
  uv run rca/build_images.py \
    --dataset princeton-nlp/SWE-bench_Verified --split test \
    --image ghcr.io/all-hands-ai/agent-server --target source
"""

import argparse
import contextlib
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

# from benchmarks.utils.args_parser import get_parser
# from benchmarks.utils.dataset import get_dataset
from openhands.agent_server.docker.build import BuildOptions, build, SDK_VERSION, _base_slug
from openhands.sdk import get_logger

from rca.utils.environment import get_docker_image_name

logger = get_logger(__name__)

valid_env = [
    'jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536',
    'jyangballin/swesmith.x86_64.pytest-dev_1776_iniconfig.16793ead',
    # 'jyangballin/swesmith.x86_64.jd_1776_tenacity.0d40e76f',
    # 'jyangballin/swesmith.x86_64.cog-creators_1776_red-discordbot.33e0eac7',
    # 'jyangballin/swesmith.x86_64.agronholm_1776_typeguard.b6a7e438',
    # 'jyangballin/swesmith.x86_64.pdfminer_1776_pdfminer.six.1a8bd2f7',
    # 'jyangballin/swesmith.x86_64.cknd_1776_stackprinter.219fcc52',
    # 'jyangballin/swesmith.x86_64.pudo_1776_dataset.5c2dc8d3',
    # 'jyangballin/swesmith.x86_64.seatgeek_1776_thefuzz.8a05a3ee',
    # 'jyangballin/swesmith.x86_64.madzak_1776_python-json-logger.5f85723f',
    # 'jyangballin/swesmith.x86_64.suor_1776_funcy.207a7810',
    # 'jyangballin/swesmith.x86_64.pygments_1776_pygments.27649ebb',
    # 'jyangballin/swesmith.x86_64.bottlepy_1776_bottle.a8dfef30',
    # 'jyangballin/swesmith.x86_64.agronholm_1776_exceptiongroup.0b4f4937',
    # 'jyangballin/swesmith.x86_64.pycqa_1776_flake8.cf1542ce',
    # 'jyangballin/swesmith.x86_64.luozhouyang_1776_python-string-similarity.115acaac',
    # 'jyangballin/swesmith.x86_64.mahmoud_1776_boltons.3bfcfdd0',
    # 'jyangballin/swesmith.x86_64.joke2k_1776_faker.8b401a7d',
    # 'jyangballin/swesmith.x86_64.cantools_1776_cantools.0c6a7871',
    # 'jyangballin/swesmith.x86_64.pyupio_1776_safety.7654596b',
    # 'jyangballin/swesmith.x86_64.knio_1776_dominate.9082227e',
    # 'jyangballin/swesmith.x86_64.jaraco_1776_inflect.c079a96a',
    # 'jyangballin/swesmith.x86_64.marshmallow-code_1776_webargs.dbde72fe',
    # 'jyangballin/swesmith.x86_64.keleshev_1776_schema.24a30457',
    # 'jyangballin/swesmith.x86_64.matthewwithanm_1776_python-markdownify.6258f5c3',
    # 'jyangballin/swesmith.x86_64.google_1776_textfsm.c31b6007',
    # 'jyangballin/swesmith.x86_64.paramiko_1776_paramiko.23f92003',
    # 'jyangballin/swesmith.x86_64.rsalmei_1776_alive-progress.35853799',
    # 'jyangballin/swesmith.x86_64.chardet_1776_chardet.9630f238',
    # 'jyangballin/swesmith.x86_64.andialbrecht_1776_sqlparse.e57923b3',
    # 'jyangballin/swesmith.x86_64.lepture_1776_mistune.bf54ef67',
    # 'jyangballin/swesmith.x86_64.python-openxml_1776_python-docx.0cf6d71f',
    # 'jyangballin/swesmith.x86_64.python_1776_mypy.e93f06ce',
    # 'jyangballin/swesmith.x86_64.hukkin_1776_tomli.443a0c1b',
    # 'jyangballin/swesmith.x86_64.pandas-dev_1776_pandas.95280573',
    # 'jyangballin/swesmith.x86_64.prettytable_1776_prettytable.ca90b055',
    # 'jyangballin/swesmith.x86_64.aio-libs_1776_async-timeout.d0baa9f1',
    # 'jyangballin/swesmith.x86_64.kennethreitz_1776_records.5941ab27',
    # 'jyangballin/swesmith.x86_64.pyasn1_1776_pyasn1.0f07d724',
    # 'jyangballin/swesmith.x86_64.pndurette_1776_gtts.dbcda4f3',
    # 'jyangballin/swesmith.x86_64.msiemens_1776_tinydb.10644a0e',
    # 'jyangballin/swesmith.x86_64.hips_1776_autograd.ac044f0d',
    # 'jyangballin/swesmith.x86_64.seperman_1776_deepdiff.ed252022',
    # 'jyangballin/swesmith.x86_64.kurtmckee_1776_feedparser.cad965a3',
    # 'jyangballin/swesmith.x86_64.gruns_1776_furl.da386f68',
    # 'jyangballin/swesmith.x86_64.tobymao_1776_sqlglot.036601ba',
    # 'jyangballin/swesmith.x86_64.instagram_1776_monkeytype.70c3acf6',
    # 'jyangballin/swesmith.x86_64.rubik_1776_radon.54b88e58',
    # 'jyangballin/swesmith.x86_64.datamade_1776_usaddress.a42a8f0c',
    # 'jyangballin/swesmith.x86_64.buriy_1776_python-readability.40256f40',
    # 'jyangballin/swesmith.x86_64.pexpect_1776_ptyprocess.1067dbda',
    # 'jyangballin/swesmith.x86_64.life4_1776_textdistance.c3aca916',
    # 'jyangballin/swesmith.x86_64.jawah_1776_charset_normalizer.1fdd6463',
    # 'jyangballin/swesmith.x86_64.tweepy_1776_tweepy.91a41c6e',
    # 'jyangballin/swesmith.x86_64.graphql-python_1776_graphene.82903263',
    # 'jyangballin/swesmith.x86_64.adrienverge_1776_yamllint.8513d9b9',
    # 'jyangballin/swesmith.x86_64.pyparsing_1776_pyparsing.533adf47',
    # 'jyangballin/swesmith.x86_64.rustedpy_1776_result.0b855e1e',
    # 'jyangballin/swesmith.x86_64.sunpy_1776_sunpy.f8edfd5c',
    # 'jyangballin/swesmith.x86_64.marshmallow-code_1776_marshmallow.9716fc62',
    # 'jyangballin/swesmith.x86_64.pylint-dev_1776_astroid.b114f6b5',
    # 'jyangballin/swesmith.x86_64.tkrajina_1776_gpxpy.09fc46b3',
    # 'jyangballin/swesmith.x86_64.arrow-py_1776_arrow.1d70d009',
    # 'jyangballin/swesmith.x86_64.jsvine_1776_pdfplumber.02ff4313',
    # 'jyangballin/swesmith.x86_64.mimino666_1776_langdetect.a1598f1a',
    # 'jyangballin/swesmith.x86_64.encode_1776_starlette.db5063c2',
    # 'jyangballin/swesmith.x86_64.tox-dev_1776_pipdeptree.c31b6418',
    # 'jyangballin/swesmith.x86_64.django_1776_channels.a144b4b8',
    # 'jyangballin/swesmith.x86_64.cookiecutter_1776_cookiecutter.b4451231',
    # 'jyangballin/swesmith.x86_64.getnikola_1776_nikola.0f4c230e',
    # 'jyangballin/swesmith.x86_64.pyca_1776_pyopenssl.04766a49',
    # 'jyangballin/swesmith.x86_64.gweis_1776_isodate.17cb25eb',
    # 'jyangballin/swesmith.x86_64.mido_1776_mido.a0158ff9',
    # 'jyangballin/swesmith.x86_64.martinblech_1776_xmltodict.0952f382',
    # 'jyangballin/swesmith.x86_64.scrapy_1776_scrapy.35212ec5',
    # 'jyangballin/swesmith.x86_64.benoitc_1776_gunicorn.bacbf8aa',
    # 'jyangballin/swesmith.x86_64.pallets_1776_markupsafe.620c06c9',
    # 'jyangballin/swesmith.x86_64.davidhalter_1776_parso.338a5760',
    # 'jyangballin/swesmith.x86_64.pydantic_1776_pydantic.acb0f10f',
    # 'jyangballin/swesmith.x86_64.django-money_1776_django-money.835c1ab8',
    # 'jyangballin/swesmith.x86_64.pallets_1776_click.fde47b4b',
    # 'jyangballin/swesmith.x86_64.iterative_1776_dvc.1d6ea681',
    # 'jyangballin/swesmith.x86_64.mozillazg_1776_python-pinyin.e42dede5',
    # 'jyangballin/swesmith.x86_64.pyutils_1776_line_profiler.a646bf0f',
    # 'jyangballin/swesmith.x86_64.django_1776_daphne.32ac73e1',
    # 'jyangballin/swesmith.x86_64.alecthomas_1776_voluptuous.a7a55f83',
    # 'jyangballin/swesmith.x86_64.stanfordnlp_1776_dspy.651a4c71',
    # 'jyangballin/swesmith.x86_64.borntyping_1776_python-colorlog.dfa10f59',
    # 'jyangballin/swesmith.x86_64.gruns_1776_icecream.f76fef56',
    # 'jyangballin/swesmith.x86_64.facelessuser_1776_soupsieve.a8080d97',
    # 'jyangballin/swesmith.x86_64.lincolnloop_1776_python-qrcode.456b01d4',
    # 'jyangballin/swesmith.x86_64.project-monai_1776_monai.a09c1f08',
    # 'jyangballin/swesmith.x86_64.kayak_1776_pypika.1c9646f0',
    # 'jyangballin/swesmith.x86_64.un33k_1776_python-slugify.872b3750',
    # 'jyangballin/swesmith.x86_64.weaveworks_1776_grafanalib.5c3b17ed',
    # 'jyangballin/swesmith.x86_64.termcolor_1776_termcolor.3a42086f',
    # 'jyangballin/swesmith.x86_64.john-kurkowski_1776_tldextract.3d1bf184',
    # 'jyangballin/swesmith.x86_64.burnash_1776_gspread.a8be3b96',
    # 'jyangballin/swesmith.x86_64.dbader_1776_schedule.82a43db1',
    # 'jyangballin/swesmith.x86_64.amueller_1776_word_cloud.ec24191c',
    # 'jyangballin/swesmith.x86_64.erikrose_1776_parsimonious.0d3f5f93',
    # 'jyangballin/swesmith.x86_64.pallets_1776_jinja.ada0a9a6',
    # 'jyangballin/swesmith.x86_64.tornadoweb_1776_tornado.d5ac65c1',
    # 'jyangballin/swesmith.x86_64.scanny_1776_python-pptx.278b47b1',
    # 'jyangballin/swesmith.x86_64.r1chardj0n3s_1776_parse.30da9e4f',
    # 'jyangballin/swesmith.x86_64.vi3k6i5_1776_flashtext.b316c7e9',
    # 'jyangballin/swesmith.x86_64.python-jsonschema_1776_jsonschema.93e0caa5',
    # 'jyangballin/swesmith.x86_64.gawel_1776_pyquery.811cd048',
    # 'jyangballin/swesmith.x86_64.theskumar_1776_python-dotenv.2b8635b7',
    # 'jyangballin/swesmith.x86_64.cloudpipe_1776_cloudpickle.6220b0ce',
    # 'jyangballin/swesmith.x86_64.sloria_1776_environs.73c372df',
    # 'jyangballin/swesmith.x86_64.pydata_1776_patsy.a5d16484',
    # 'jyangballin/swesmith.x86_64.mahmoud_1776_glom.fb3c4e76',
    # 'jyangballin/swesmith.x86_64.modin-project_1776_modin.8c7799fd',
    # 'jyangballin/swesmith.x86_64.dask_1776_dask.5f61e423',
    # 'jyangballin/swesmith.x86_64.markedjs_1776_marked.dbf29d91',
    # 'jyangballin/swesmith.x86_64.alanjds_1776_drf-nested-routers.6144169d',
    # 'jyangballin/swesmith.x86_64.conan-io_1776_conan.86f29e13',
    # 'jyangballin/swesmith.x86_64.python-trio_1776_trio.cfbbe2c1',
    # 'jyangballin/swesmith.x86_64.sqlfluff_1776_sqlfluff.50a1c4b6',
    # 'jyangballin/swesmith.x86_64.mewwts_1776_addict.75284f95',
    # 'jyangballin/swesmith.x86_64.facebookresearch_1776_hydra.0f03eb60',
    # 'jyangballin/swesmith.x86_64.pwaller_1776_pyfiglet.f8c5f35b',
    # 'jyangballin/swesmith.x86_64.mozilla_1776_bleach.73871d76',
    # 'jyangballin/swesmith.x86_64.cool-rr_1776_pysnooper.57472b46',
    'jyangballin/swesmith.x86_64.facebookresearch_1776_fvcore.a491d5b9',
    'jyangballin/swesmith.x86_64.getmoto_1776_moto.694ce1f4',
    'jyangballin/swesmith.x86_64.python-hyper_1776_h11.bed0dd4a',
    'jyangballin/swesmith.x86_64.marshmallow-code_1776_apispec.8b421526',
    'jyangballin/swesmith.x86_64.spulec_1776_freezegun.5f171db0',
    'jyangballin/swesmith.x86_64.pydicom_1776_pydicom.7d361b3d',
    'jyangballin/swesmith.x86_64.stanfordnlp_1776_string2string.c4a72f59'
 ]

@contextlib.contextmanager
def capture_output(base_name: str, out_dir: Path):
    """
    Capture stdout/stderr during a block and stream them to:
      <out_dir>/<base_name>/build-<timestamp>.log

    Keeps redirect_* semantics; writes are realtime (line-buffered + flush).
    Yields the log_path.
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    log_path = Path(out_dir) / base_name / f"build-{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # tell the user where we’re logging, without being swallowed by the redirect
    # (goes to the original stderr so it’s visible immediately)
    logger.info(f"Logging build output to {log_path}")

    # Open line-buffered so writes flush on newlines;
    # also wrap to hard-flush every write.
    f = log_path.open("w", encoding="utf-8", buffering=1)

    class _FlushOnWrite(io.TextIOBase):
        encoding = f.encoding

        def __init__(self, sink):
            self._sink = sink

        def write(self, s):
            n = self._sink.write(s)
            self._sink.flush()
            return n

        def flush(self):
            self._sink.flush()

        def fileno(self):
            # allow libs that try to detect fileno()
            return self._sink.fileno()

    sink = _FlushOnWrite(f)

    # Redirect stdout/stderr to the same realtime sink.
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):  # type: ignore[arg-type]
        try:
            yield log_path
        finally:
            # make sure everything is on disk
            sink.flush()
            f.close()


# def extend_parser() -> argparse.ArgumentParser:
#     """Reuse benchmark parser and extend with build-related options."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", required=True, help="Dataset name")
#     parser.add_argument("--split", required=True, help="Dataset split")
#     parser.add_argument("--n-limit", type=int, help="Limit number of instances")
#     parser.add_argument(
#         "--docker-image-prefix",
#         default="docker.io/swebench/",
#         help="Prefix for SWE-Bench images",
#     )
#     parser.add_argument(
#         "--image",
#         default="ghcr.io/all-hands-ai/agent-server",
#         help="Target repo/name for built image",
#     )
#     parser.add_argument(
#         "--target",
#         default="binary-minimal",
#         help="Build target (binary | binary-minimal | source | source-minimal)",
#     )
#     parser.add_argument(
#         "--platforms", default="linux/amd64", help="Comma-separated platforms"
#     )
#     parser.add_argument("--custom-tags", default="", help="Comma-separated custom tags")
#     parser.add_argument(
#         "--push", action="store_true", help="Push via buildx instead of load locally"
#     )
#     parser.add_argument(
#         "--max-workers", type=int, default=1, help="Concurrent builds (be cautious)"
#     )
#     parser.add_argument(
#         "--dry-run", action="store_true", help="List base images only, don’t build"
#     )
#     parser.description = "Build all agent-server images for SWE-Bench base images."
#     return parser


# def collect_unique_base_images(dataset, split, prefix, n_limit):
#     df = get_dataset(
#         dataset_name=dataset, split=split, eval_limit=n_limit if n_limit else None
#     )
#     return sorted(
#         {
#             get_docker_image_name(str(row["instance_id"]), prefix)
#             for _, row in df.iterrows()
#         }
#     )


class BuildOutput(BaseModel):
    time: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    base_image: str
    tags: list[str]
    error: str | None = None
    log_path: str | None = None


def build_one(base_image: str) -> BuildOutput:
    opts = BuildOptions(
        base_image=base_image,
        # custom_tags=args.custom_tags,
        image="ghcr.io/all-hands-ai/agent-server",
        target="source-minimal",
        platforms=["linux/amd64"],
        push=False,
    )
    tags = build(opts)
    return BuildOutput(base_image=base_image, tags=tags, error=None)


def _default_build_output_dir(
    dataset: str, split: str, base_dir: Path | None = None
) -> Path:
    """
    Default: ./builds/<dataset>/<split>
    Keeps build outputs in one predictable place, easy to .gitignore.
    """
    root = (base_dir or Path.cwd()) / "builds" / dataset / split
    root.mkdir(parents=True, exist_ok=True)
    return root


def _update_pbar(
    pbar: tqdm,
    successes: int,
    failures: int,
    running: int,
    sample: str | None,
    last_event: str | None,
):
    postfix = f"✅ {successes}  ❌ {failures}  🏃 {running}"
    if sample:
        postfix += f" ({sample})"
    if last_event:
        pbar.set_description(last_event)
    pbar.set_postfix_str(postfix, refresh=True)


def main(argv: list[str]) -> int:
    # parser = extend_parser()
    # args = parser.parse_args(argv)
    bases = valid_env
    # Decide manifest path under ./builds/<dataset>/<split>/
    # BUILD_DIR = _default_build_output_dir(args.dataset, args.split)
    dataset = "SWE-Smith"
    split = "train"
    BUILD_DIR = _default_build_output_dir(dataset, split)
    BUILD_LOG_DIR = BUILD_DIR / "logs"
    manifest_path = BUILD_DIR / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def build_one_fn(base: str) -> BuildOutput:
        with capture_output(base, BUILD_LOG_DIR) as log_path:
            # result = build_one(base, args)
            result = build_one(base)
            result.log_path = str(log_path)
            return result

    # if args.dry_run:
    #     print("\n".join(bases))
    #     return 0

    successes = 0
    failures = 0
    in_progress: set[str] = set()
    mu = Lock()

    with (
        manifest_path.open("w") as writer,
        tqdm(total=len(bases), desc="Building agent-server images", leave=True) as pbar,
    ):
        _update_pbar(pbar, successes, failures, 0, None, "Queueing")

        # Single unified path: ThreadPoolExecutor( max_workers = args.max_workers ),
        # even if it's 1
        with ThreadPoolExecutor(max_workers=1) as ex:
            futures = {}
            for base in bases:
                in_progress.add(base)
                fut = ex.submit(build_one_fn, base)
                futures[fut] = base

            _update_pbar(
                pbar,
                successes,
                failures,
                len(in_progress),
                next(iter(in_progress), None),
                "Running",
            )

            for fut in as_completed(futures):
                base = futures[fut]
                try:
                    result: BuildOutput = fut.result()
                    writer.write(result.model_dump_json() + "\n")
                    writer.flush()
                    with mu:
                        successes += 1
                    _update_pbar(
                        pbar, successes, failures, len(in_progress), base, "✅ Done"
                    )
                except Exception as e:
                    logger.error("Build failed for %s: %r", base, e)
                    # Write a failure line to manifest; keep going.
                    writer.write(
                        BuildOutput(
                            base_image=base, tags=[], error=repr(e)
                        ).model_dump_json()
                        + "\n"
                    )
                    writer.flush()
                    with mu:
                        failures += 1
                    _update_pbar(
                        pbar, successes, failures, len(in_progress), base, "❌ Failed"
                    )
                finally:
                    with mu:
                        in_progress.discard(base)
                    pbar.update(1)
                    _update_pbar(
                        pbar,
                        successes,
                        failures,
                        len(in_progress),
                        next(iter(in_progress), None),
                        None,
                    )

    # Optional: write a tiny summary JSON next to the manifest for quick reads
    summary_path = manifest_path.with_name("summary.json")
    summary_path.write_text(
        (
            "{"
            f'"dataset":"{dataset}",'
            f'"split":"{split}",'
            f'"total_unique_base_images":{len(bases)},'
            f'"built":{successes},'
            f'"failed":{failures}'
            "}"
        ),
        encoding="utf-8",
    )

    logger.info(
        "Done. Built=%d  Failed=%d  Manifest=%s  Summary=%s",
        successes,
        failures,
        str(manifest_path),
        str(summary_path),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    # sys.exit(main(sys.argv[1:]))
    sys.exit(main([]))
