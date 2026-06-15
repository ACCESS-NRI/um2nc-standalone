import os
import sys
from jinja2 import Template, StrictUndefined

template_file=sys.argv[1]

template_str = open(template_file).read()

rendered = Template(template_str, undefined=StrictUndefined).render(env=os.environ)

print(rendered)