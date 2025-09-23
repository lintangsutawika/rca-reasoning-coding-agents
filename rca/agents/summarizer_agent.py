"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model

from minisweagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded


class SummarizerAgent(DefaultAgent):
    def __init__(self, deliberator_model: Model, summarizer_model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.deliberator_model = deliberator_model
        self.summarizer_model = summarizer_model
        self.env = env
        self.extra_template_vars = {}

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.deliberator_model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        summary = self.summarizer_model.query(self.messages)
        response = self.deliberator_model.query([
            {"role": "system", "content": self.render_template(self.config.system_template)},
            {"role": "user", "content": self.render_template(self.config.instance_template)+f"Summary:\n{summary}"}
            ])
        self.add_message("assistant", **response)
        return response
