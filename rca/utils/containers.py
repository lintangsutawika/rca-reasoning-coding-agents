# from openhands.agent_server.docker.build import SDK_VERSION, _base_slug
from openhands.agent_server.docker.build import _base_slug

def extract_custom_tag(base_image: str) -> str:
    """
    Extract SWE-Bench instance ID from official SWE-Bench image name.

    Example:
        docker.io/swebench/sweb.eval.x86_64.django_1776_django-12155:latest
        -> sweb.eval.x86_64.django_1776_django-12155
    """
    name_tag = base_image.split("/")[-1]
    name = name_tag.split(":")[0]
    return name

def get_agent_server_docker_image(
    image_source: str,
    image_name: str,
    target: str = "source-minimal",
    slug: str = "e485bba",
) -> str:
    return (
        image_source + f":{slug}-{extract_custom_tag(image_name)}-{target}"
    )

# def apply_git_overrides(workspace, instance, cwd):
