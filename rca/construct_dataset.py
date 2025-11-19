"""
Preprocess the SWEBench dataset to SkyRL format
"""

import argparse
import os

import datasets

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/swe_smith")

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "SWE-bench/SWE-smith"
    # eval_data_source = "SumanthRH/SWE-bench_Verified"

    def filter_empty_examples(example):
        return example["problem_statement"] != ""
    
    def filter_valid_envs(example):
        return example["image_name"] in valid_env

    # Use filter to remove empty problem statements
    raw_data = datasets.load_dataset(data_source, "default", split="train")
    # Shuffle
    raw_data = raw_data.shuffle(seed=42)
    # Select 10000
    raw_data = raw_data.select([i for i in list(range(5000))])
    # Filter out empty examples
    raw_data = raw_data.filter(filter_empty_examples)
    raw_data = raw_data.filter(filter_valid_envs)

    print(len(raw_data), "examples after filtering")

    # train_dataset = raw_data.select(range(1000))
    # val_dataset = raw_data.select(range(1000, 1500))
    train_dataset = raw_data.select(range(35))
    val_dataset = raw_data.select(range(35))
    # val_dataset = raw_data.select(range(40, 50))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                # "data_source": data_source if split == "train" else eval_data_source,
                "data_source": "swe-smith",
                "prompt": [
                    {
                        "role": "user",
                        "content": example["problem_statement"],
                    }
                ],
                "env_extras": {
                    "data_source": "swe-smith",
                },
                "env_class": "null",
                "instance": example,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("test"), with_indices=True)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))
