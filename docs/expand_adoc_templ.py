
# coding: utf-8

import sys
import re


f = open("transforms.md", "r")
contents = f.read()


t = "hello {{method Jeremy}} there {{method close}}"


def do_method(ps): return str(ps)


fn_lu = {  # 'class': sub_class,
    #'arguments': sub_arguments,
    #'arg': sub_arg,
    #'xref': sub_xref,
    #'methods': sub_methods,
    'method': do_method}


def sub_methods():
    """<h3 class="methods">Methods</h3>
        <ul class="methodlist">"""


def sub_method(ps):
    return f'<li id="Transform-set_state" class="method">'


def do_tmpl(s):
    inner = s.group(1)
    fn_name, *params = inner.split(' ')
    fn = fn_lu[fn_name]
    return fn(params)


re.sub(r"{{(.*?)}}", do_tmpl, t)
